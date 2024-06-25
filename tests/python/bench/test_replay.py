import asyncio
from mlc_llm.bench import OpenAIRequestSender, load_replay_log, MetricsProcessor, replay
import time


def test_replay(log_path):
    replay_log = load_replay_log(log_path)
    async def _get_stats():
        start_time = time.monotonic()
        async with OpenAIRequestSender("localhost", 8008, include_server_metrics=True) as sender:
            await replay(replay_log, sender)
            req_records = sender.get_request_records()
        # print(f"req_records: {req_records}")
        metric_processor = MetricsProcessor(req_records)
        metric_processor.filter_metrics()
        end_time = time.monotonic()
        metric_processor.generate_metrics_summary(start_time, end_time)

    asyncio.run(_get_stats())


if __name__ == "__main__":
    log_path = "/opt/scratch/yongwu/mlc-llm/tests/python/bench/martian.jsonl"
    log_path = "/opt/scratch/yongwu/mlc-llm/tests/python/bench/replay_log/output_nh_pro_llama3_8b_0531_tp2_fp16.jsonl"
    log_path = "/opt/scratch/yongwu/mlc-llm/tests/python/bench/replay_log/output_llama3_70b_0531_tp4_fp8.jsonl"
    # log_path = "/opt/scratch/yongwu/mlc-llm/tests/python/bench/replay_log/debug.jsonl"
    # log_path = "/opt/scratch/yongwu/mlc-llm/tests/python/bench/replay_log/output_llama3_70b_0531_tp4_fp8.jsonl"
    # log_path = "/opt/scratch/yongwu/mlc-llm/tests/python/bench/replay_log/debug.jsonl"
    # log_path = "/opt/scratch/yongwu/mlc-llm/tests/python/bench/replay_log/one_request.jsonl"
    test_replay(log_path)