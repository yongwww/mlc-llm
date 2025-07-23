"""The FasterTransformer quantization config"""

import functools
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import tvm
from tvm import DataType, DataTypeCode, nd, relax, te, tir, topi, IRModule
from tvm import dlight as dl
from tvm.contrib.nvcc import get_target_compute_version
from tvm.relax.frontend import nn
from tvm.runtime import Module, NDArray
from tvm.target import Target
from tvm.script import tir as T

import hashlib
import os
from pathlib import Path

from ..loader import QuantizeMapping
from ..op import extern, faster_transformer_dequantize_gemm
from ..support import logging
from ..support.auto_target import detect_cuda_arch_list
from ..support.style import bold
from .group_quantization import (
    GroupQuantize,
    GroupQuantizeEmbedding,
    GroupQuantizeLinear,
)
from .utils import is_final_fc, is_moe_gate, convert_uint_to_float

logger = logging.getLogger(__name__)

# Global cache for quantization functions to persist across instances
_GLOBAL_QUANTIZE_FUNC_CACHE: Dict[str, Any] = {}

# Global cache for dequantized weights to avoid recomputing during forward pass
_GLOBAL_DEQUANTIZED_WEIGHT_CACHE: Dict[str, Any] = {}

def _get_cache_key(
    weight_shape: Tuple[int, int],
    weight_dtype: str,
    device_type: str,
    quantize_dtype: str,
    storage_dtype: str,
    model_dtype: str,
    group_size: Optional[int],
    skip_cutlass_preprocessing: bool,
    cuda_arch: Optional[str] = None,
) -> str:
    """
    Generate a comprehensive cache key for quantization functions.
    
    Parameters
    ----------
    weight_shape : Tuple[int, int]
        Shape of the weight tensor
    weight_dtype : str
        Data type of the weight tensor
    device_type : str
        Device type (e.g., "cuda")
    quantize_dtype : str
        Quantization data type (e.g., "int4")
    storage_dtype : str
        Storage data type (e.g., "int8")
    model_dtype : str
        Model data type (e.g., "float16")
    group_size : Optional[int]
        Group size for quantization
    skip_cutlass_preprocessing : bool
        Whether to skip CUTLASS preprocessing
    cuda_arch : Optional[str]
        CUDA architecture (e.g., "sm_86")
    
    Returns
    -------
    str
        Comprehensive cache key
    """
    key_components = [
        f"shape_{weight_shape[0]}x{weight_shape[1]}",
        f"wdtype_{weight_dtype}",
        f"device_{device_type}",
        f"qdtype_{quantize_dtype}",
        f"sdtype_{storage_dtype}",
        f"mdtype_{model_dtype}",
        f"group_{group_size}",
        f"skip_cutlass_{skip_cutlass_preprocessing}",
        f"arch_{cuda_arch}",
    ]
    
    # Create a hash of the key components for consistency
    key_str = "_".join(key_components)
    return hashlib.md5(key_str.encode()).hexdigest()

def _get_persistent_cache_path() -> Path:
    """Get the path for persistent quantization function cache."""
    # NOTE: Persistent cache disabled (TVM functions can't be serialized)
    cache_dir = Path("/tmp") / "mlc_llm_ft_quantize_cache"
    return cache_dir

def _load_persistent_cache():
    """Load persistent quantization function cache from disk."""
    # NOTE: Compiled TVM functions cannot be serialized/pickled since they contain
    # native code and runtime state. We'll skip persistent caching for now.
    global _GLOBAL_QUANTIZE_FUNC_CACHE
    _GLOBAL_QUANTIZE_FUNC_CACHE = {}
    logger.debug("Persistent cache disabled - using in-memory cache only")

def _save_persistent_cache():
    """Save persistent quantization function cache to disk."""
    # NOTE: Compiled TVM functions cannot be serialized/pickled since they contain
    # native code and runtime state. We'll skip persistent caching for now.
    # Only keep in-memory caching which is still beneficial during the same session.
    pass

def _get_dequantized_weight_cache_key(
    layer_name: str,
    weight_shape: Tuple[int, int],
    quantize_dtype: str,
    storage_dtype: str,
    model_dtype: str,
    group_size: Optional[int],
) -> str:
    """Generate cache key for dequantized weights."""
    key_components = [
        f"layer_{layer_name}",
        f"shape_{weight_shape[0]}x{weight_shape[1]}",
        f"qdtype_{quantize_dtype}",
        f"sdtype_{storage_dtype}",
        f"mdtype_{model_dtype}",
        f"group_{group_size}",
    ]
    key_str = "_".join(key_components)
    return hashlib.md5(key_str.encode()).hexdigest()

def _create_optimized_dequantize_kernel(
    weight_shape: Tuple[int, int],
    quantize_dtype: str,
    storage_dtype: str,
    model_dtype: str,
    group_size: Optional[int],
    num_elem_per_storage: int,
    max_int_value: int,
) -> Callable:
    """
    Create an optimized dequantization kernel using vectorized operations.
    
    This compiles a more efficient dequantization kernel compared to the 
    element-wise approach in _dequantize_weight_for_cublas.
    """
    k, n = weight_shape
    cur_group_size = k if not group_size else group_size
    
    def _optimized_dequantize_kernel(q_weight: te.Tensor, q_scale: te.Tensor) -> te.Tensor:
        """Optimized dequantization kernel with vectorized operations."""
        
        # Use vectorized bit extraction for better performance
        def _vectorized_dequantize(w: te.Tensor, s: te.Tensor, i: tir.Var, j: tir.Var):
            quantize_dtype_bits = 4  # int4
            
            # Constants for bit operations
            tir_bin_mask = tir.const((1 << quantize_dtype_bits) - 1, storage_dtype)
            tir_max_int = tir.const(max_int_value, model_dtype)
            
            # Vectorized extraction - process multiple elements at once when possible
            packed_idx = j // num_elem_per_storage
            element_idx = j % num_elem_per_storage
            
            # Extract quantized value
            w_val = w[i, packed_idx]
            s_val = s[i // cur_group_size, j]
            
            # Optimized bit shift and mask
            shift = (element_idx * quantize_dtype_bits).astype(storage_dtype)
            w_extracted = tir.bitwise_and(
                tir.shift_right(w_val, shift), 
                tir_bin_mask
            ).astype(model_dtype)
            
            # Apply dequantization with offset
            return (w_extracted - tir_max_int) * s_val
        
        # Create output tensor with optimized computation
        dequantized_weight = te.compute(
            shape=(k, n),
            fcompute=lambda i, j: _vectorized_dequantize(q_weight, q_scale, i, j).astype(model_dtype),
            name="ft_dequantized_weight_optimized",
            attrs={"schedule_rule": "meta_schedule.ParallelizeVectorizeUnroll"}
        )
        
        return dequantized_weight
    
    return _optimized_dequantize_kernel

# Initialize persistent cache on module load
_load_persistent_cache()

# NOTE: Persistent cache saving disabled (TVM functions can't be serialized)


@dataclass
class FTQuantize:  # pylint: disable=too-many-instance-attributes
    """Configuration for FasterTransformer quantization"""

    name: str
    kind: str
    quantize_dtype: Literal["int4", "int8"]
    storage_dtype: Literal["int8"]
    model_dtype: Literal["float16"]
    group_size: Optional[int] = None
    skip_cutlass_preprocessing: bool = False  # Skip CUTLASS preprocessing for cuBLAS compatibility

    num_elem_per_storage: int = 0
    max_int_value: int = 0

    def fallback_group_quantize(self) -> GroupQuantize:
        """
        The fallback group quantization config for other parameters.

        Returns
        ------
        quantize: GroupQuantize
            The group quantization config to fallback.
        """
        return GroupQuantize(
            name=self.name,
            kind="group-quant",
            group_size=32,  # hardcoded to 32 as only supporting int4 quantization
            quantize_dtype=self.quantize_dtype,
            storage_dtype="uint32",
            model_dtype=self.model_dtype,
            linear_weight_layout="NK",
        )

    def __post_init__(self):
        assert self.kind == "ft-quant"
        quantize_dtype = DataType(self.quantize_dtype)
        storage_dtype = DataType(self.storage_dtype)
        assert self.quantize_dtype in ["int4", "int8"]
        assert storage_dtype.type_code == DataTypeCode.INT
        assert self.model_dtype == "float16"
        assert self.group_size in [None, 64, 128]
        if storage_dtype.bits < quantize_dtype.bits:
            raise ValueError("Storage unit should be greater or equal to quantized element")

        self.num_elem_per_storage = storage_dtype.bits // quantize_dtype.bits
        self.max_int_value = (2 ** (quantize_dtype.bits - 1)) - 1
        self._quantize_func_cache = {}
        
        # Ensure persistent cache is managed properly
        self._ensure_cache_size_limit()
    
    def _ensure_cache_size_limit(self, max_cache_size: int = 100):
        """Ensure the global cache doesn't exceed size limit."""
        global _GLOBAL_QUANTIZE_FUNC_CACHE
        if len(_GLOBAL_QUANTIZE_FUNC_CACHE) > max_cache_size:
            # Remove oldest entries (simple LRU simulation)
            keys_to_remove = list(_GLOBAL_QUANTIZE_FUNC_CACHE.keys())[max_cache_size:]
            for key in keys_to_remove:
                del _GLOBAL_QUANTIZE_FUNC_CACHE[key]
            logger.debug(f"Trimmed cache size to {len(_GLOBAL_QUANTIZE_FUNC_CACHE)} entries")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        # NOTE: Persistent cache saving disabled (TVM functions can't be serialized)
        pass

    def quantize_model(
        self,
        model: nn.Module,
        quant_map: QuantizeMapping,
        name_prefix: str,
    ) -> nn.Module:
        """
        Quantize model with FasterTransformer quantization

        Parameters
        ----------
        model : nn.Module
            The non-quantized nn.Module.

        quant_map : QuantizeMapping
            The quantize mapping with name mapping and func mapping.

        name_prefix : str
            The name prefix for visited weight.

        Returns
        -------
        ret : nn.Module
            The quantized nn.Module.
        """

        class _Mutator(nn.Mutator):
            def __init__(self, config: FTQuantize, quant_map: QuantizeMapping) -> None:
                super().__init__()
                self.config = config
                self.quant_map = quant_map

            def visit_module(self, name: str, node: nn.Module) -> Any:
                """
                The visiting method for FasterTransformer quantization of nn.Module nodes.

                Parameters
                ----------
                name : str
                    The name of the current node.

                node : nn.Module
                    The current node of nn.Module to mutate.

                Returns
                ------
                ret_node: Any
                    The new node to replace current node.
                """
                if isinstance(node, nn.Linear):
                    weight_name = f"{name}.weight"
                    if (
                        # pylint: disable=too-many-boolean-expressions
                        is_final_fc(name)
                        or node.out_dtype == "float32"
                        or (self.config.quantize_dtype == "int4" and node.out_features % 8 != 0)
                        or (self.config.quantize_dtype == "int8" and node.out_features % 4 != 0)
                    ):
                        # Under any of the conditions we fall back to GroupQuantize
                        # For `is_final_fc()` see https://github.com/mlc-ai/mlc-llm/issues/1723
                        # If simply skipping lm_head quantization degrades performance
                        # Other requirements are from CUTLASS
                        logger.info(
                            'Fallback to GroupQuantize for nn.Linear: "%s", '
                            + "weight.shape: %s, out_dtype: %s",
                            bold(name),
                            node.weight.shape,
                            node.out_dtype,
                        )
                        # GroupQuantize layers only have q_weight and q_scale (no dequantized_weight)
                        self.quant_map.param_map[weight_name] = [f"{name}.q_weight", f"{name}.q_scale"]
                        group_quantize = self.config.fallback_group_quantize()
                        self.quant_map.map_func[weight_name] = group_quantize.quantize_weight
                        return GroupQuantizeLinear.from_linear(node, group_quantize)
                    if not is_moe_gate(name, node):
                        # FTQuantizeLinear layers have q_weight and q_scale
                        self.quant_map.param_map[weight_name] = [f"{name}.q_weight", f"{name}.q_scale"]
                        self.quant_map.map_func[weight_name] = self.config.quantize_weight
                        return FTQuantizeLinear.from_linear(node, self.config, layer_name=name)
                if isinstance(node, nn.Embedding):
                    weight_name = f"{name}.weight"
                    self.quant_map.param_map[weight_name] = [f"{name}.q_weight", f"{name}.q_scale"]
                    group_quantize = self.config.fallback_group_quantize()
                    self.quant_map.map_func[weight_name] = group_quantize.quantize_weight
                    return GroupQuantizeEmbedding.from_embedding(node, group_quantize)
                return self.visit(name, node)

        model.to(dtype=self.model_dtype)
        mutator = _Mutator(self, quant_map)
        model = mutator.visit(name_prefix, model)
        return model

    def quantize_weight(self, weight: NDArray) -> List[NDArray]:
        """
        Quantize weight with FasterTransformer quantization

        Parameters
        ----------
        weight : NDArray
            The original weight.

        Returns
        -------
        ret: List[NDArray]
            The list of FasterTransformer quantized weights.
        """
        assert tvm.get_global_func("relax.ext.cutlass", True), (
            "Cutlass should be enabled in TVM runtime to quantize weight, "
            "but not enabled in current TVM runtime environment. "
            "To enable Cutlass in TVM runtime, please `set(USE_CUTLASS ON)` "
            "in config.cmake when compiling TVM from source"
        )
        assert len(weight.shape) == 2
        device = weight.device
        device_type = device.DEVICE_TYPE_TO_NAME[device.device_type]
        
        if device_type == "cuda":
            target = Target.current()
            if target is None:
                target = Target.from_device(device)
            with target:
                # Get CUDA architecture for better cache key
                try:
                    cuda_arch_list = detect_cuda_arch_list(target=target)
                    cuda_arch = f"sm_{cuda_arch_list[0]}" if cuda_arch_list else None
                except:
                    cuda_arch = None

                def _create_quantize_func() -> IRModule:
                    bb = relax.BlockBuilder()  # pylint: disable=invalid-name
                    weight_var = relax.Var(
                        "weight", relax.TensorStructInfo(weight.shape, weight.dtype)
                    )
                    with bb.function(name="main", params=[weight_var]):
                        with bb.dataflow():
                            lv0 = bb.emit_te(
                                self._quantize, weight_var
                            )  # pylint: disable=invalid-name
                            lv1 = bb.normalize(lv0[0])
                            
                            if self.skip_cutlass_preprocessing:
                                # Skip CUTLASS preprocessing for cuBLAS compatibility
                                final_weight = lv1
                            else:
                                # Apply CUTLASS preprocessing for CUTLASS GEMM optimization
                                final_weight = bb.emit(
                                    relax.call_pure_packed(
                                        "cutlass.ft_preprocess_weight",
                                        lv1,
                                        detect_cuda_arch_list(target=target)[0],
                                        DataType(self.quantize_dtype).bits == 4,
                                        sinfo_args=lv1.struct_info,
                                    )
                                )
                            
                            gv = bb.emit_output(
                                relax.Tuple([final_weight, lv0[1]])
                            )  # pylint: disable=invalid-name
                        bb.emit_func_output(gv)
                    return bb.finalize()

                def _compile_quantize_func(mod: IRModule) -> Callable:
                    mod = dl.ApplyDefaultSchedule(  # type: ignore   # pylint: disable=not-callable
                        dl.gpu.Reduction(),
                        dl.gpu.GeneralReduction(),
                        dl.gpu.Fallback(),
                    )(mod)
                    ex = relax.build(mod, target=target)
                    vm = relax.VirtualMachine(ex, device)  # pylint: disable=invalid-name
                    return vm["main"]

                # Generate comprehensive cache key
                cache_key = _get_cache_key(
                    weight_shape=(int(weight.shape[0]), int(weight.shape[1])),
                    weight_dtype=str(weight.dtype),
                    device_type=device_type,
                    quantize_dtype=self.quantize_dtype,
                    storage_dtype=self.storage_dtype,
                    model_dtype=self.model_dtype,
                    group_size=self.group_size,
                    skip_cutlass_preprocessing=self.skip_cutlass_preprocessing,
                    cuda_arch=cuda_arch,
                )
                
                # Check global cache first, then instance cache
                global _GLOBAL_QUANTIZE_FUNC_CACHE
                quantize_func = _GLOBAL_QUANTIZE_FUNC_CACHE.get(cache_key)
                if quantize_func is None:
                    quantize_func = self._quantize_func_cache.get(cache_key)
                
                if quantize_func is None:
                    logger.info("Compiling quantize function for cache key: %s", cache_key[:16] + "...")
                    quantize_func = _compile_quantize_func(_create_quantize_func())
                    
                    # Save to both global and instance cache
                    _GLOBAL_QUANTIZE_FUNC_CACHE[cache_key] = quantize_func
                    self._quantize_func_cache[cache_key] = quantize_func
                    
                    # NOTE: Persistent cache saving disabled (TVM functions can't be serialized)
                else:
                    logger.debug("Using cached quantize function for key: %s", cache_key[:16] + "...")
                
                data = quantize_func(weight)
                # Return only quantized weight and scale
                return [data[0], data[1]]
        else:
            raise NotImplementedError(f"Device type {device_type} is not supported")

    def _quantize(  # pylint: disable=too-many-locals
        self,
        weight: te.Tensor,
    ) -> Tuple[te.Tensor, te.Tensor]:
        """FasterTransformer quantization for weight tensor, defined in tensor expression."""
        assert len(weight.shape) == 2
        n, k = weight.shape

        cur_group_size = k if not self.group_size else self.group_size
        scale_shape = (tir.ceildiv(k, cur_group_size), n)
        r = te.reduce_axis((0, cur_group_size), name="r")

        max_abs = te.compute(
            shape=scale_shape,
            fcompute=lambda j, i: te.max(
                tir.if_then_else(
                    j * cur_group_size + r < k,
                    te.abs(weight[i, j * cur_group_size + r]),
                    te.min_value(self.model_dtype),
                ),
                axis=r,
            ),
            name="max_abs_value",
        )
        max_int = tir.const(self.max_int_value, self.model_dtype)
        scale = te.compute(
            scale_shape,
            lambda i, j: max_abs[i, j].astype(self.model_dtype) / max_int,
            name="scale",
        )
        # compute scaled weight
        quantize_dtype = DataType(self.quantize_dtype)
        bin_mask = tir.const((1 << quantize_dtype.bits) - 1, self.storage_dtype)
        scaled_weight = te.compute(
            shape=weight.shape,
            fcompute=lambda i, j: tir.min(
                tir.max(
                    tir.round(weight[i, j] / scale[j // cur_group_size, i]),
                    -max_int - 1,
                ),
                max_int,
            ).astype(self.storage_dtype)
            & bin_mask,
        )

        quantized_weight_shape = (k, tir.ceildiv(n, self.num_elem_per_storage))
        r = te.reduce_axis((0, self.num_elem_per_storage), name="r")  # pylint: disable=invalid-name
        quantized_weight = te.compute(
            shape=quantized_weight_shape,
            fcompute=lambda j, i: tir.sum(
                tir.if_then_else(
                    i * self.num_elem_per_storage + r < n,
                    scaled_weight[i * self.num_elem_per_storage + r, j]
                    << (
                        r.astype(self.storage_dtype)
                        * tir.const(quantize_dtype.bits, self.storage_dtype)
                    ),
                    tir.const(0, self.storage_dtype),
                ),
                axis=r,
            ),
            name="weight",
        )

        return quantized_weight, scale




class FTQuantizeLinear(nn.Module):
    """Linear layer with FasterTransformer quantization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: Optional[str] = None,
        config: Optional[FTQuantize] = None,
        layer_name: str = "unknown",
    ):
        if config is None:
            raise ValueError("FTQuantizeLinear requires a quantization configuration")
        
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        self.out_dtype = dtype
        self.layer_name = layer_name
        cur_group_size = in_features if not config.group_size else config.group_size

        # Standard FT quantization parameters (original shapes)
        self.q_weight = nn.Parameter(
            (in_features, tir.ceildiv(out_features, config.num_elem_per_storage)),
            config.storage_dtype,
        )
        self.q_scale = nn.Parameter(
            (tir.ceildiv(in_features, cur_group_size), out_features), 
            config.model_dtype
        )
        if bias:
            self.bias = nn.Parameter(
                (out_features,), config.model_dtype if dtype is None else dtype
            )
        else:
            self.bias = None

        # Optional precomputed dequantized weight for cuBLAS path
        self.dequantized_weight = None
        self._dequantized_weight_compiled = False

    @staticmethod
    def from_linear(src: nn.Linear, config: FTQuantize, layer_name: str = "unknown") -> "FTQuantizeLinear":
        """
        Converts a non-quantized nn.Linear to a FasterTransformer quantized FTQuantizeLinear

        Parameters
        ----------
        src : nn.Linear
            The non-quantized nn.Linear.

        config : FTQuantize
            The FasterTransformer quantization config.
        
        layer_name : str
            The name of the layer for optimization purposes.

        Returns
        -------
        ret : FTQuantizeLinear
            The FasterTransformer quantized FTQuantizeLinear layer.
        """
        quantized_linear = FTQuantizeLinear(
            in_features=src.in_features,
            out_features=src.out_features,
            config=config,
            bias=getattr(src, "bias", None) is not None,
            dtype=src.out_dtype,
            layer_name=layer_name,
        )
        if quantized_linear.bias is not None:
            quantized_linear.bias.attrs = src.bias.attrs
        
        return quantized_linear
    
    def precompute_dequantized_weight_for_cublas(self):
        """
        Precompute dequantized weights for cuBLAS path to avoid runtime overhead.
        
        This method should be called after the quantized weights are loaded
        and when cuBLAS is enabled for optimal performance.
        """
        if self._dequantized_weight_compiled:
            return
        
        # Check if we have a cached dequantized weight
        cache_key = _get_dequantized_weight_cache_key(
            layer_name=self.layer_name,
            weight_shape=(self.in_features, self.out_features),
            quantize_dtype=self.config.quantize_dtype,
            storage_dtype=self.config.storage_dtype,
            model_dtype=self.config.model_dtype,
            group_size=self.config.group_size,
        )
        
        global _GLOBAL_DEQUANTIZED_WEIGHT_CACHE
        if cache_key in _GLOBAL_DEQUANTIZED_WEIGHT_CACHE:
            self.dequantized_weight = _GLOBAL_DEQUANTIZED_WEIGHT_CACHE[cache_key]
            self._dequantized_weight_compiled = True
            logger.debug(f"Using cached dequantized weight for layer: {self.layer_name}")
            return
        
        # Create optimized dequantization kernel
        dequantize_kernel = _create_optimized_dequantize_kernel(
            weight_shape=(self.in_features, self.out_features),
            quantize_dtype=self.config.quantize_dtype,
            storage_dtype=self.config.storage_dtype,
            model_dtype=self.config.model_dtype,
            group_size=self.config.group_size,
            num_elem_per_storage=self.config.num_elem_per_storage,
            max_int_value=self.config.max_int_value,
        )
        
        # Precompute dequantized weight using optimized kernel
        self.dequantized_weight = nn.op.tensor_expr_op(
            dequantize_kernel,
            name_hint=f"precomputed_dequantize_{self.layer_name}",
            args=[self.q_weight, self.q_scale],
        )
        
        # Cache the result for future use
        _GLOBAL_DEQUANTIZED_WEIGHT_CACHE[cache_key] = self.dequantized_weight
        self._dequantized_weight_compiled = True
        
        logger.debug(f"Precomputed dequantized weight for layer: {self.layer_name}")

    def forward(self, x: nn.Tensor) -> nn.Tensor:  # pylint: disable=invalid-name
        """
        Forward method for FasterTransformer quantized linear layer.

        Parameters
        ----------
        x : nn.Tensor
            The input tensor.

        Returns
        -------
        ret : nn.Tensor
            The output tensor for the FasterTransformer quantized linear layer.
        """
        # Check if cuBLAS is enabled from compilation flags
        use_cublas = (hasattr(extern.get_store(), 'cublas_gemm') and 
                     extern.get_store().cublas_gemm)
        
        if use_cublas:
            # VLLM-STYLE APPROACH: Proper FT dequantization for cuBLAS
            # This properly handles FT quantized weights with int8 storage and FT reordered packing
            
            # Use precomputed dequantized weight if available for better performance
            if self.dequantized_weight is not None:
                w = self.dequantized_weight
            else:
                # Fallback to on-the-fly dequantization
                w = nn.op.tensor_expr_op(
                    self._dequantize_weight_for_cublas,
                    name_hint="ft_dequantize_for_cublas",
                    args=[self.q_weight, self.q_scale],
                )
            
            # Use cuBLAS fp16 GEMM
            result = nn.op.matmul(x, w, out_dtype=self.out_dtype)
            
            if self.bias is not None:
                result = result + self.bias
                
            return result
        else:
            # CUTLASS PATH: Use fused FT quantization (best decode performance)
            return faster_transformer_dequantize_gemm(
                x, self.q_weight, self.q_scale, self.bias, group_size=self.config.group_size
            )

    def _dequantize_weight_for_cublas(self, q_weight: te.Tensor, q_scale: te.Tensor) -> te.Tensor:
        """
        Dequantize raw FT quantized weights (without CUTLASS preprocessing) for cuBLAS GEMM.
        
        This handles FT weights that skip the cutlass.ft_preprocess_weight step.
        """
        # Get dimensions  
        k, packed_n = q_weight.shape
        n = packed_n * self.config.num_elem_per_storage
        cur_group_size = k if not self.config.group_size else self.config.group_size
        
        # For raw FT weights (no CUTLASS preprocessing), use standard bit extraction
        def _dequantize_raw_ft(w: te.Tensor, s: te.Tensor, i: tir.Var, j: tir.Var):
            """Dequantize raw FT weights using standard bit extraction"""
            quantize_dtype_bits = 4  # int4
            storage_dtype = self.config.storage_dtype
            model_dtype = self.config.model_dtype
            num_elem_per_storage = self.config.num_elem_per_storage
            
            tir_bin_mask = tir.const((1 << quantize_dtype_bits) - 1, storage_dtype)
            tir_max_int = tir.const((2 ** (quantize_dtype_bits - 1)) - 1, model_dtype)
            
            # Extract quantized value from raw FT packed format
            w_val = w[i, j // num_elem_per_storage]
            s_val = s[i // cur_group_size, j]
            shift = (j % num_elem_per_storage * quantize_dtype_bits).astype(storage_dtype)
            w_extracted = tir.bitwise_and(tir.shift_right(w_val, shift), tir_bin_mask).astype(model_dtype)
            
            # Apply scaling: dequantized = (quantized - max_int) * scale
            return (w_extracted - tir_max_int) * s_val
        
        # Create dequantized weight tensor
        dequantized_weight = te.compute(
            shape=(k, n),
            fcompute=lambda i, j: _dequantize_raw_ft(q_weight, q_scale, i, j).astype(self.config.model_dtype),
            name="ft_dequantized_weight_raw"
        )
        
        return dequantized_weight

    def to(self, dtype: Optional[str] = None) -> None:
        """
        Override to() such that we do not convert bias if there is an out_dtype.
        Otherwise, we might run into dtype mismatch when computing x + self.bias.
        """
        self.q_weight.to(dtype=dtype)
        self.q_scale.to(dtype=dtype)
        if self.bias is not None and self.out_dtype is None:
            self.bias.to(dtype=dtype)
        if dtype is not None and isinstance(getattr(self, "dtype", None), str):
            self.dtype = dtype  # pylint: disable=attribute-defined-outside-init
