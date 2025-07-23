"""Potential externel modules managed by MLC compilation stack.

An externl module could contain one or multiple handcrafted kernels, as long as it is provided as
an object file (`.o`), a C++ source file (`.cc`), or a CUDA source file (`.cu`). It can be
integrated into the system pretty smoothly.

As examples, `flashinfer.py` contains such an example that instructs MLC to compile
"$tvm_home/3rdparty/flashinfer/src/tvm_wrapper.cu" with a specific set of compilation flags and then
link into the generated artifact of MLC LLM. TVM PR #16247
(https://github.com/apache/tvm/pull/16247/) provides more details of using TVM's
`nn.SourceModule` to integrate C++ and CUDA files, and `nn.ObjectModule` to integrate object files.

To conveniently use those externel modules, MLC LLM compilation pipeline manages an extra global
singleton `Store: ExternalModuleStore` to store the configured modules. It is supposed to be enabled
before any compilation happens, and configured during a model's `forward` method is invoked.
"""

import dataclasses
from typing import Optional, Dict

from tvm.target import Target


@dataclasses.dataclass
class ExternModuleStore:
    """Global store of external modules enabled during compilation."""

    configured: bool = False
    target: Optional[Target] = None
    flashinfer: bool = False
    faster_transformer: bool = False
    cutlass_group_gemm: bool = False
    cutlass_gemm: bool = False
    cublas_gemm: bool = False


STORE: ExternModuleStore = ExternModuleStore()
"""Singleton of `ExternModuleStore`."""

_EXTERNAL_FLAGS: Optional[Dict[str, bool]] = None


def set_external_flags(flags: Dict[str, bool]) -> None:
    """Set external module flags for inference configuration.
    
    Parameters
    ----------
    flags : Dict[str, bool]
        External module flags to set.
    """
    global _EXTERNAL_FLAGS
    _EXTERNAL_FLAGS = flags


def enable(target: Target, flashinfer: bool, faster_transformer: bool, cutlass: bool, cublas_gemm: bool = False) -> None:
    """Enable external modules. It should be called before any compilation happens."""
    global STORE  # pylint: disable=global-statement
    cutlass = (
        cutlass
        and target.kind.name == "cuda"
        and target.attrs.get("arch", "") in ["sm_90a", "sm_100a"]
    )
    faster_transformer = False
    STORE = ExternModuleStore(
        configured=False,
        target=target,
        flashinfer=flashinfer,
        faster_transformer=faster_transformer,
        cutlass_group_gemm=cutlass,
        cutlass_gemm=cutlass,
        cublas_gemm=cublas_gemm,
    )


def get_store() -> ExternModuleStore:
    """Get the global store of external modules."""
    return STORE


def configure(external_flags: Optional[Dict[str, bool]] = None) -> None:
    """Configure external modules with extra parameters. It should be called during a model's
    `forward` method is invoked.

    Parameters
    ----------
    external_flags : Optional[Dict[str, bool]]
        External module flags to configure. If None, will use globally set flags.
    """
    global _EXTERNAL_FLAGS
    store = get_store()
    if store.configured:
        return
    store.configured = True
    
    # Use provided flags, or fall back to globally set flags
    flags_to_use = external_flags or _EXTERNAL_FLAGS
    
    # If external flags are available, use them to configure the store
    if flags_to_use is not None:
        store.flashinfer = flags_to_use.get("flashinfer", store.flashinfer)
        store.faster_transformer = flags_to_use.get("faster_transformer", store.faster_transformer)
        store.cutlass_group_gemm = flags_to_use.get("cutlass", store.cutlass_group_gemm)
        store.cutlass_gemm = flags_to_use.get("cutlass", store.cutlass_gemm)
        store.cublas_gemm = flags_to_use.get("cublas_gemm", store.cublas_gemm)
    
    if store.flashinfer or store.faster_transformer:
        assert store.target.kind.name == "cuda"
