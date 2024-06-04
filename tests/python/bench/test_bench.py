"""Test MLC-LLM Bench."""
import asyncio
import time
import os

from mlc_llm.bench import (
    MetricsProcessor,
    OpenAIRequestSender,
    load_replay_log,
    replay,
)


def test_bench():
    cur_file_path = os.path.dirname(os.path.abspath(__file__))
    log_path = cur_file_path + "/bench_sample.jsonl"
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


if __name__ == "__main__":
    test_bench()
