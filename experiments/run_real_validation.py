"""Run a small real-model comparison after explicit user invocation."""

from collections.abc import Callable, Mapping
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from statistics import fmean
from typing import Any

from zhiheng_router.adapters import (
    DeepSeekAdapter,
    DeepSeekPricing,
    ModelAdapter,
    ModelRequest,
    ModelResponse,
)
from zhiheng_router.estimates import estimate_candidates
from zhiheng_router.evaluators import LLMJudgeEvaluator, QualityEvaluator, QualityResult
from zhiheng_router.predictor import RoutingArtifact, load_artifact
from zhiheng_router.router import route
from zhiheng_router.runner import BatchRunner, BenchmarkRunner, RunRecord
from zhiheng_router.schemas import (
    CandidateEstimate,
    ModelCandidate,
    RoutingDecision,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "real_validation"
ARTIFACT_PATH = ROOT / "outputs" / "mvp" / "router.joblib"
JUDGE_MODEL_ID = "deepseek-v4-pro"
QUALITY_THRESHOLD = 0.8
QUERIES = [
    ("real-query-001", "Write a Python function that returns the median of a non-empty list."),
    ("real-query-002", "修复下面函数在空列表时抛出的异常，并说明修改理由：\n```python\ndef average(values):\n    return sum(values) / len(values)\n```"),
    ("real-query-003", "Implement an O(n) JavaScript function that finds the longest substring without repeated characters, and include two tests."),
    ("real-query-004", "请用 Python 实现一个带指数退避和最大重试次数的异步请求函数；取消任务时必须立即传播 CancelledError。"),
    ("real-query-005", "Design a dependency resolver that returns a valid installation order and reports one concrete cycle when no order exists. Provide pseudocode and complexity analysis."),
    ("real-query-006", "Solve 3x + 7 = 25 and briefly verify the result."),
    ("real-query-007", "袋中有 4 个红球和 6 个蓝球，不放回连续抽取 2 个球。求两球颜色相同的概率，并写出计算过程。"),
    ("real-query-008", "A fair six-sided die is rolled until a 6 appears. What is the expected number of rolls? Explain the derivation."),
    ("real-query-009", "求函数 f(x)=x^3-6x^2+9x+1 在区间 [0,4] 上的最大值和最小值，并说明端点为何需要检查。"),
    ("real-query-010", "Prove by induction that 1 + 2 + ... + n = n(n+1)/2 for every positive integer n."),
    ("real-query-011", "A monitoring service becomes slower after traffic doubles, while CPU usage stays low. Give three plausible causes and the first measurement you would check for each."),
    ("real-query-012", "甲、乙、丙三人分别在周一、周二、周三值班，每人一天。已知甲不在周一，乙在甲之前。请给出唯一排班并解释。"),
    ("real-query-013", "Plan a zero-downtime database migration when old and new application versions must run concurrently for one hour. List the steps in dependency order."),
    ("real-query-014", "一次上线后错误率上升，同时请求量、CPU 和内存都未明显变化。请区分相关性与因果性，并提出两个可验证的解释。"),
    ("real-query-015", "Three jobs take 2, 4, and 6 hours. Two identical workers are available, and each job cannot be split. Find a schedule with minimum completion time and justify why it is optimal."),
    ("real-query-016", "In two sentences, explain what a model router does."),
    ("real-query-017", "请向没有网络背景的同学解释 HTTPS 为什么能够同时提供加密和身份验证，控制在 120 字以内。"),
    ("real-query-018", "Summarize this passage in one sentence: Retrieval-augmented generation looks up external documents at inference time, while fine-tuning changes model parameters using training examples."),
    ("real-query-019", "Compare retrieval-augmented generation with fine-tuning for a support assistant whose product documentation changes every week. Give one advantage and one limitation of each."),
    ("real-query-020", "某分布式系统在网络分区期间仍允许两侧写入。请用一个库存扣减场景解释它可能牺牲的一致性，以及分区恢复后需要处理什么问题。"),
]

# Prices are explicit experiment inputs rather than hidden adapter defaults.
MODEL_CONFIG = {
    "deepseek-v4-flash": {
        "predictor_proxy_model_id": "dev-model-e",
        "pricing_usd_per_million_tokens": {
            "cache_hit_input": 0.0028,
            "cache_miss_input": 0.14,
            "output": 0.28,
        },
        "routing_profile": {
            "estimated_cost": 0.14,
            "estimated_latency_ms": 1_000.0,
        },
    },
    "deepseek-v4-pro": {
        "predictor_proxy_model_id": "dev-model-a",
        "pricing_usd_per_million_tokens": {
            "cache_hit_input": 0.003625,
            "cache_miss_input": 0.435,
            "output": 0.87,
        },
        "routing_profile": {
            "estimated_cost": 0.435,
            "estimated_latency_ms": 1_500.0,
        },
    },
}


class ProgressLogger:
    """Emit synchronous, flush-on-write progress for the current strategy."""

    def __init__(self, strategy: str, total: int) -> None:
        self.strategy = strategy
        self.total = total
        self.current = 0

    def begin_query(self) -> None:
        self.current += 1
        print(
            f"[{self.strategy}] query {self.current}/{self.total} started",
            flush=True,
        )

    def model_completed(self, model_id: str) -> None:
        print(
            f"[{self.strategy}] query {self.current}/{self.total} "
            f"model call completed: model={model_id}",
            flush=True,
        )

    def judged(self, quality_score: float) -> None:
        print(
            f"[{self.strategy}] query {self.current}/{self.total} "
            f"judged: quality={quality_score:.4f}",
            flush=True,
        )

    def finish_query(self) -> None:
        print(
            f"[{self.strategy}] query {self.current}/{self.total} finished",
            flush=True,
        )


class RecordingAdapter:
    """Capture successful responses without changing BenchmarkRunner records."""

    def __init__(self, delegate: ModelAdapter, progress: ProgressLogger) -> None:
        self._delegate = delegate
        self._progress = progress
        self.responses: dict[str, ModelResponse] = {}

    @property
    def model_id(self) -> str:
        return self._delegate.model_id

    def invoke(self, request: ModelRequest) -> ModelResponse:
        response = self._delegate.invoke(request)
        self.responses[request.request_id] = response
        self._progress.model_completed(response.model_id)
        return response


class RecordingQualityEvaluator:
    """Retain Judge metadata that the minimal RunRecord intentionally omits."""

    def __init__(self, delegate: QualityEvaluator,
                 progress: ProgressLogger) -> None:
        self._delegate = delegate
        self._progress = progress
        self.results: dict[tuple[str, str], QualityResult] = {}

    def evaluate(self, query_text: str, response_text: str,
                 metadata: Mapping[str, Any] | None = None) -> QualityResult:
        result = self._delegate.evaluate(query_text, response_text)
        self.results[(query_text, response_text)] = result
        self._progress.judged(result.quality_score)
        return result


class ProgressBenchmarkRunner(BenchmarkRunner):
    """Add per-query progress around the unchanged BenchmarkRunner flow."""

    def __init__(self, selector: Callable[[str], RoutingDecision],
                 adapters: Mapping[str, ModelAdapter],
                 quality_evaluator: QualityEvaluator,
                 progress: ProgressLogger) -> None:
        super().__init__(selector, adapters, quality_evaluator)
        self._progress = progress

    def run_one(self, query_id: str, query_text: str) -> RunRecord:
        self._progress.begin_query()
        try:
            return super().run_one(query_id, query_text)
        finally:
            self._progress.finish_query()


def _candidates() -> tuple[ModelCandidate, ...]:
    return tuple(ModelCandidate(model_id) for model_id in MODEL_CONFIG)


def _mapped_estimates(query: str, artifact: RoutingArtifact
                      ) -> tuple[CandidateEstimate, ...]:
    proxy_ids = [values["predictor_proxy_model_id"] for values in MODEL_CONFIG.values()]
    proxy_rows = estimate_candidates(
        query, proxy_ids, artifact.predictor, artifact.statistics)
    proxy_by_id = {row.model_id: row for row in proxy_rows}
    return tuple(
        CandidateEstimate(
            model_id=model_id,
            pass_probability=proxy_by_id[values["predictor_proxy_model_id"]].pass_probability,
            **values["routing_profile"],
        )
        for model_id, values in MODEL_CONFIG.items()
    )


def _fixed_decision(model_id: str, estimates: tuple[CandidateEstimate, ...],
                    reason: str) -> RoutingDecision:
    selected = next(row for row in estimates if row.model_id == model_id)
    return RoutingDecision(
        selected_model=selected.model_id,
        pass_probability=selected.pass_probability,
        estimated_cost=selected.estimated_cost,
        estimated_latency_ms=selected.estimated_latency_ms,
        fallback_used=False,
        decision_reason=reason,
        qualified_models=(),
        candidate_estimates=estimates,
    )


def _selectors(artifact: RoutingArtifact
               ) -> dict[str, Callable[[str], RoutingDecision]]:
    candidates = _candidates()

    def router_selector(query: str) -> RoutingDecision:
        return route(candidates, _mapped_estimates(query, artifact), artifact.policy)

    def strongest_selector(query: str) -> RoutingDecision:
        return _fixed_decision(
            "deepseek-v4-pro",
            _mapped_estimates(query, artifact),
            "BASELINE_ALWAYS_DEEPSEEK_V4_PRO",
        )

    def lowest_cost_selector(query: str) -> RoutingDecision:
        return _fixed_decision(
            "deepseek-v4-flash",
            _mapped_estimates(query, artifact),
            "BASELINE_ALWAYS_DEEPSEEK_V4_FLASH",
        )

    return {
        "router": router_selector,
        "strongest": strongest_selector,
        "lowest_cost": lowest_cost_selector,
    }


def _adapters(api_key: str,
              progress: ProgressLogger) -> dict[str, RecordingAdapter]:
    adapters: dict[str, RecordingAdapter] = {}
    for model_id, values in MODEL_CONFIG.items():
        prices = values["pricing_usd_per_million_tokens"]
        delegate = DeepSeekAdapter(
            model_id,
            api_key=api_key,
            timeout=60.0,
            pricing=DeepSeekPricing(
                cache_hit_input_per_million=prices["cache_hit_input"],
                cache_miss_input_per_million=prices["cache_miss_input"],
                output_per_million=prices["output"],
            ),
        )
        adapters[model_id] = RecordingAdapter(delegate, progress)
    return adapters


def _serialize_record(strategy: str, record: RunRecord,
                      adapters: dict[str, RecordingAdapter],
                      evaluator: RecordingQualityEvaluator) -> dict[str, object]:
    response = adapters[record.selected_model].responses.get(record.request_id)
    quality = (
        evaluator.results.get((record.query_text, record.response_text))
        if record.response_text is not None else None
    )
    return {
        "strategy": strategy,
        "query_id": record.query_id,
        "request_id": record.request_id,
        "query_text": record.query_text,
        "selected_model": record.selected_model,
        "fallback_used": record.fallback_used,
        "status": record.status,
        "response": record.response_text,
        "latency_ms": record.model_latency_ms,
        "cost": record.cost,
        "cost_currency": "USD" if response is not None else None,
        "input_tokens": response.input_units if response is not None else None,
        "output_tokens": response.output_units if response is not None else None,
        "finish_reason": response.finish_reason if response is not None else None,
        "quality_score": record.quality_score,
        "quality_pass": record.quality_pass,
        "judge_reason": quality.metadata["reason"] if quality is not None else None,
        "measurement_kind": record.measurement_kind,
        "error_kind": record.error_kind,
        "error_message": record.error_message,
    }


def _average(rows: list[dict[str, object]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row[field] is not None]
    return fmean(values) if values else None


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    successful = [row for row in rows if row["status"] == "success"]
    judged = [row for row in successful if row["quality_score"] is not None]
    return {
        "total_count": len(rows),
        "success_count": len(successful),
        "failure_count": len(rows) - len(successful),
        "success_rate": len(successful) / len(rows) if rows else None,
        "average_cost_usd": _average(successful, "cost"),
        "total_cost_usd": sum(float(row["cost"]) for row in successful),
        "average_latency_ms": _average(successful, "latency_ms"),
        "quality_pass_rate": (
            fmean(float(row["quality_pass"]) for row in judged) if judged else None
        ),
        "average_quality": _average(judged, "quality_score"),
        "total_input_tokens": sum(int(row["input_tokens"]) for row in successful),
        "total_output_tokens": sum(int(row["output_tokens"]) for row in successful),
    }


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def main() -> None:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if api_key is None or not api_key.strip():
        raise SystemExit("DEEPSEEK_API_KEY is not set")
    configured_judge_key = os.getenv("LLM_JUDGE_API_KEY")
    has_separate_judge_key = (
        configured_judge_key is not None and bool(configured_judge_key.strip()))
    judge_api_key = configured_judge_key if has_separate_judge_key else api_key
    judge_key_source = (
        "LLM_JUDGE_API_KEY environment variable"
        if has_separate_judge_key
        else "DEEPSEEK_API_KEY environment variable"
    )
    if not ARTIFACT_PATH.is_file():
        raise SystemExit(f"trusted Router artifact not found: {ARTIFACT_PATH}")
    artifact = load_artifact(ARTIFACT_PATH)
    proxy_ids = {
        values["predictor_proxy_model_id"] for values in MODEL_CONFIG.values()
    }
    if not proxy_ids <= artifact.predictor.models.keys():
        missing = sorted(proxy_ids - artifact.predictor.models.keys())
        raise SystemExit(f"Router artifact is missing predictor proxy models: {missing}")

    all_rows: list[dict[str, object]] = []
    results: dict[str, dict[str, object]] = {}
    selectors = _selectors(artifact)
    judge = LLMJudgeEvaluator(
        JUDGE_MODEL_ID,
        api_key=judge_api_key,
        threshold=QUALITY_THRESHOLD,
        timeout=60.0,
    )
    for strategy, selector in selectors.items():
        progress = ProgressLogger(strategy, len(QUERIES))
        evaluator = RecordingQualityEvaluator(judge, progress)
        adapters = _adapters(api_key, progress)
        print(f"[{strategy}] starting: {len(QUERIES)} queries", flush=True)
        records = BatchRunner(ProgressBenchmarkRunner(
            selector, adapters, evaluator, progress)).run_many(QUERIES)
        rows = [
            _serialize_record(strategy, record, adapters, evaluator)
            for record in records
        ]
        all_rows.extend(rows)
        results[strategy] = _summarize(rows)
        print(f"[{strategy}] completed: {len(records)} queries", flush=True)

    created_at = datetime.now(timezone.utc).isoformat()
    config = {
        "experiment": "minimal_real_deepseek_validation",
        "timestamp": created_at,
        "measurement_kind": "measured",
        "query_count": len(QUERIES),
        "queries": [{"query_id": query_id, "query_text": text}
                    for query_id, text in QUERIES],
        "strategies": list(selectors),
        "models": MODEL_CONFIG,
        "router_policy": asdict(artifact.policy),
        "predictor_artifact": str(ARTIFACT_PATH.relative_to(ROOT)),
        "predictor_quality_score_threshold": artifact.predictor.quality_score_threshold,
        "predictor_proxy_mapping": {
            model_id: values["predictor_proxy_model_id"]
            for model_id, values in MODEL_CONFIG.items()
        },
        "selection_estimates_kind": "query_aware_predictor_with_explicit_proxy_mapping",
        "selection_notice": (
            "Pass probabilities come from the existing synthetic dev-model predictor through "
            "an explicit role proxy mapping. They are not learned DeepSeek performance "
            "estimates and must not be reported as validated DeepSeek quality predictions."
        ),
        "quality_judge": {
            "model_id": JUDGE_MODEL_ID,
            "quality_threshold": QUALITY_THRESHOLD,
            "measurement_kind": "model_judge",
            "api_key_source": judge_key_source,
        },
        "api_base_url": "https://api.deepseek.com",
        "api_key_source": "DEEPSEEK_API_KEY environment variable",
    }

    records_text = "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
        for row in all_rows
    )
    results_text = _json_text(results)
    config_text = _json_text(config)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "records.jsonl").write_text(records_text, encoding="utf-8")
    (OUTPUT_DIR / "results.json").write_text(results_text, encoding="utf-8")
    (OUTPUT_DIR / "experiment_config.json").write_text(config_text, encoding="utf-8")

    print(_json_text(results), end="")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
