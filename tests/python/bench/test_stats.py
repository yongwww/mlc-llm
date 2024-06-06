"""Test stats collection."""
import asyncio
import time

from mlc_llm.bench import (
    MetricsProcessor,
    OpenAIRequestSender,
    load_replay_log,
    replay,
)

import mlc_llm
import json
import time
from dataclasses import dataclass
from typing import Literal, Dict, Optional, Any, List
import random
import sys


def test_collect_stats():
    """Test collecting stats."""
    # log_path = "/opt/scratch/yongwu/mlc-llm/Data/datadog-sample.csv"
    log_path = "/opt/scratch/yongwu/mlc-llm/tests/python/bench/sample.jsonl"
    # log_path = "/opt/scratch/yongwu/mlc-llm/tests/python/bench/output_llama3_70b_0531.jsonl"
    replay_log = load_replay_log(log_path)

    async def _get_stats():
        # Consider adding an extra arg like enable_stats,
        # then we can merge OpenAIRequestSender and OpenaiStatsCollector
        # collector = OpenaiStatsCollector("localhost", 8008)
        async with OpenAIRequestSender("localhost", 8008) as sender:
            start_time = time.monotonic()
            await replay(replay_log, sender)
            metrics = sender.get_metrics()
            get_metrics_summary(metrics, start_time, time.monotonic())

    asyncio.run(_get_stats())


def test_collect_metrics():
    """Test metrics collector"""
    metrics_collector = MetricsCollector()
    new_metrics = {
        "inter_token_latency": 0.01,
        "decode_tokens_per_s": 0.02,
        "ttft": 0.03,
        "end_to_end_latency": 0.04,
        "prompt_tokens": 10,
        "completion_tokens": 20,
    }
    metrics_collector.add_metrics(new_metrics)
    metric_summary = metrics_collector.get_metrics_summary(0, 10)
    assert "overall_output_throughput" in metric_summary


def test_prompts(log_path="/opt/scratch/yongwu/mlc-llm/tests/python/bench/sample_wo_prompt.jsonl"):
    # log_path = "/opt/scratch/yongwu/mlc-llm/Data/datadog-sample.csv"
    # log_path = "/opt/scratch/yongwu/mlc-llm/tests/python/bench/sample.jsonl"
    # log_path = "/opt/scratch/yongwu/mlc-llm/tests/python/bench/sample_wo_prompt.jsonl"
    # log_path = "/opt/scratch/yongwu/mlc-llm/tests/python/bench/output_llama3_70b_0531.jsonl"
    replay_log = load_replay_log(log_path)

    async def _get_stats():
        async with OpenAIRequestSender("localhost", 8008) as sender:
            start_time = time.monotonic()
            await replay(replay_log, sender)
            req_records = sender.get_request_records()
        metric_processor = MetricsProcessor(req_records)
        metric_processor.filter_metrics()
        metric_processor.generate_metrics_summary(start_time, time.monotonic())

    asyncio.run(_get_stats())



def test_server_metrics():
    engine = mlc_llm.MLCEngine(
        "HF://mlc-ai/Llama-3-8B-Instruct-q0f16-MLC",
        mode="server",
    )
    config = Config(max_counter=1000, top_p=1.0, repeat=1)

    setting = f"temp={config.temperature}, top_p={config.top_p}, counting={config.max_counter}, gen-mode={config.gen_mode}"

    print(setting)

    seed = random.randint(0, 1 << 30)
    _ = engine.chat.completions.create(
        messages=config.messages,
        temperature=config.temperature,
        top_p=config.top_p,
        n=config.n,
        response_format=config.response_format,
        max_tokens=30,
        seed=seed,
    )

    seed = random.randint(0, 1 << 30)
    response = engine.chat.completions.create(
        messages=config.messages,
        temperature=config.temperature,
        top_p=config.top_p,
        n=config.n,
        response_format=config.response_format,
        max_tokens=30,
        seed=seed,
    ) 



def test_openai_async(log_path="/opt/scratch/yongwu/mlc-llm/tests/python/bench/sample_wo_prompt.jsonl"):

    replay_log = load_replay_log(log_path)
    async def _get_stats():
        start_time = time.monotonic()
        async with OpenAIRequestSender("localhost", 8008) as sender:
            await replay(replay_log, sender)
            req_records = sender.get_request_records()
        metric_processor = MetricsProcessor(req_records)
        metric_processor.filter_metrics()
        end_time = time.monotonic()
        metric_processor.generate_metrics_summary(start_time, end_time)

    asyncio.run(_get_stats())


def test_aiohttp(log_path="/opt/scratch/yongwu/mlc-llm/tests/python/bench/llmperf.jsonl"):
    # log_path = "/opt/scratch/yongwu/mlc-llm/tests/python/bench/llmperf.jsonl"
    replay_log = load_replay_log(log_path)

    async def _get_stats():
        start_time = time.monotonic()
        async with OpenAIRequestSender("localhost", 8008) as sender:
            await replay(replay_log, sender.aiohttp_send_request)
            req_records = sender.get_request_records()
        end_time = time.monotonic()
        metric_processor = MetricsProcessor(req_records)
        metric_processor.filter_metrics()
        metric_processor.generate_metrics_summary(start_time, end_time)

    asyncio.run(_get_stats())


def test_httpx(log_path="/opt/scratch/yongwu/mlc-llm/tests/python/bench/llmperf.jsonl"):
    # log_path = "/opt/scratch/yongwu/mlc-llm/tests/python/bench/llmperf.jsonl"
    replay_log = load_replay_log(log_path)

    async def _get_stats():
        start_time = time.monotonic()
        async with OpenAIRequestSender("localhost", 8008) as sender:
            await replay(replay_log, sender.httpx_send_request)
            req_records = sender.get_request_records()
        end_time = time.monotonic()
        metric_processor = MetricsProcessor(req_records)
        metric_processor.filter_metrics()
        metric_processor.generate_metrics_summary(start_time, end_time)

    asyncio.run(_get_stats())


if __name__ == "__main__":
    log_path = "/opt/scratch/yongwu/mlc-llm/tests/python/bench/martian.jsonl"
    # log_path = "/opt/scratch/yongwu/mlc-llm/tests/python/bench/one_request.jsonl"
    # test_prompts()
    # test_async_openai()
    # test_aiohttp()
    test_openai_async(log_path)
    # test_aiohttp(log_path)
    # test_httpx(log_path)
    # test_server_metrics()
    # test_collect_stats()
    # test_collect_metrics()
