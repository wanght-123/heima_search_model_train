"""训练命令和部署指标的最小测试。"""

from job_finetune_course.model_deployment import RequestResult, summarize
from job_finetune_course.model_training import build_command


def test_training_command_is_easy_to_inspect(tmp_path) -> None:
    config = tmp_path / "sft.yaml"
    config.write_text("stage: sft\n", encoding="utf-8")
    assert build_command("train", config) == ["llamafactory-cli", "train", str(config)]


def test_service_summary() -> None:
    results = [
        RequestResult(True, 10.0, 200, 20),
        RequestResult(True, 30.0, 200, 20),
        RequestResult(False, 40.0, 500, 0, "server error"),
    ]
    summary = summarize(results, wall_seconds=2.0, concurrency=2)
    assert summary["success_rate"] == 2 / 3
    assert summary["requests_per_second"] == 1.0
    assert summary["latency_ms"]["p50"] == 10.0
