"""模型 API 启动与并发压测工具。"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .model_evaluation import load_evaluation_rows
from .model_training import run_llamafactory


@dataclass(frozen=True)
class RequestResult:
    """一次 HTTP 请求的测量结果。"""

    ok: bool
    latency_ms: float
    status_code: int | None
    output_tokens: int
    error: str | None = None


def start_api(
    config_path: Path, *, host: str, port: int, model_name: str, dry_run: bool = False
) -> None:
    """通过 LLaMA-Factory 启动 OpenAI 兼容 API。"""

    run_llamafactory(
        "api",
        config_path,
        dry_run=dry_run,
        extra_env={
            "API_HOST": host,
            "API_PORT": str(port),
            "API_MODEL_NAME": model_name,
        },
    )


def send_request(
    url: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    timeout: float,
    api_key: str | None,
) -> RequestResult:
    """发送一次 OpenAI 格式的聊天请求。"""

    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "do_sample": False,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            status = response.status
        choices = payload.get("choices", [])
        if status != 200 or not choices:
            raise ValueError("返回中没有 choices")
        tokens = int(payload.get("usage", {}).get("completion_tokens", 0))
        return RequestResult(
            True, (time.perf_counter() - started) * 1000, status, tokens
        )
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        return RequestResult(
            False, (time.perf_counter() - started) * 1000, exc.code, 0, detail
        )
    except Exception as exc:
        return RequestResult(
            False, (time.perf_counter() - started) * 1000, None, 0, str(exc)
        )


def percentile(values: list[float], quantile: float) -> float | None:
    """使用 nearest-rank 方式计算分位数。"""

    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def summarize(
    results: list[RequestResult], wall_seconds: float, concurrency: int
) -> dict[str, Any]:
    """汇总成功率、吞吐、延迟和错误类型。"""

    successes = [result for result in results if result.ok]
    latencies = [result.latency_ms for result in successes]
    output_tokens = sum(result.output_tokens for result in successes)
    errors = Counter(
        result.error or f"HTTP {result.status_code}"
        for result in results
        if not result.ok
    )
    return {
        "requests": len(results),
        "concurrency": concurrency,
        "success_rate": len(successes) / len(results) if results else 0.0,
        "requests_per_second": len(successes) / wall_seconds if wall_seconds else 0.0,
        "output_tokens_per_second": output_tokens / wall_seconds
        if wall_seconds
        else 0.0,
        "latency_ms": {
            "mean": sum(latencies) / len(latencies) if latencies else None,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "max": max(latencies) if latencies else None,
        },
        "errors": dict(errors.most_common(10)),
    }


def benchmark(
    dataset_path: Path,
    *,
    url: str,
    model: str,
    request_count: int,
    concurrency: int,
    max_tokens: int,
    timeout: float,
    api_key: str | None,
) -> tuple[dict[str, Any], list[RequestResult]]:
    """从 diagnostic 集取提示词，并发请求 API。"""

    prompts = [
        row["prompt_messages"]
        for row in load_evaluation_rows(dataset_path, limit=request_count)
    ]
    started = time.perf_counter()
    results: list[RequestResult] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                send_request,
                url,
                model,
                prompts[index % len(prompts)],
                max_tokens=max_tokens,
                timeout=timeout,
                api_key=api_key,
            )
            for index in range(request_count)
        ]
        for future in as_completed(futures):
            results.append(future.result())
    return summarize(results, time.perf_counter() - started, concurrency), results


def save_benchmark_report(
    path: Path, summary: dict[str, Any], results: list[RequestResult]
) -> None:
    """保存压测汇总和逐请求明细。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"summary": summary, "results": [asdict(result) for result in results]},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
