"""Sequential BatchRunner behavior built on the single-query runner."""

from unittest.mock import Mock

from zhiheng_router.adapters import MockModelAdapter
from zhiheng_router.evaluators import MockQualityEvaluator
from zhiheng_router.runner import BatchRunner, BenchmarkRunner
from zhiheng_router.schemas import CandidateEstimate, RoutingDecision


def decision(model_id: str) -> RoutingDecision:
    estimate = CandidateEstimate(model_id, 0.9, 0.2, 12.0)
    return RoutingDecision(
        selected_model=model_id,
        pass_probability=estimate.pass_probability,
        estimated_cost=estimate.estimated_cost,
        estimated_latency_ms=estimate.estimated_latency_ms,
        fallback_used=False,
        decision_reason="TEST_SELECTION",
        qualified_models=(model_id,),
        candidate_estimates=(estimate,),
    )


def make_runner() -> BenchmarkRunner:
    adapter = MockModelAdapter(
        "dev-model-a",
        response_text="fixed response",
        input_units=10,
        output_units=3,
        cost=0.2,
        model_latency_ms=12.0,
    )
    evaluator = MockQualityEvaluator(default_score=0.9)
    return BenchmarkRunner(lambda query: decision("dev-model-a"),
                           {"dev-model-a": adapter}, evaluator)


def test_multiple_queries_execute_in_input_order() -> None:
    batch = BatchRunner(make_runner())
    queries = [("query-2", "second"), ("query-1", "first"), ("query-3", "third")]

    records = batch.run_many(queries)

    assert [(record.query_id, record.query_text) for record in records] == queries
    assert all(record.status == "success" for record in records)


def test_each_query_calls_run_one_once() -> None:
    runner = make_runner()
    original = runner.run_one
    runner.run_one = Mock(wraps=original)
    queries = [("query-1", "first"), ("query-2", "second")]

    BatchRunner(runner).run_many(queries)

    assert runner.run_one.call_count == len(queries)
    assert runner.run_one.call_args_list == [
        (("query-1", "first"),),
        (("query-2", "second"),),
    ]


def test_result_order_matches_mixed_model_selection_order() -> None:
    adapters = {
        "dev-model-a": MockModelAdapter("dev-model-a", response_text="A"),
        "dev-model-b": MockModelAdapter("dev-model-b", response_text="B"),
    }
    runner = BenchmarkRunner(
        lambda query: decision("dev-model-a" if query == "use A" else "dev-model-b"),
        adapters,
    )

    records = BatchRunner(runner).run_many([
        ("query-b", "use B"), ("query-a", "use A"), ("query-b2", "use B again")])

    assert [record.selected_model for record in records] == [
        "dev-model-b", "dev-model-a", "dev-model-b"]
    assert [record.response_text for record in records] == ["B", "A", "B"]


def test_failed_queries_keep_records_and_do_not_stop_the_batch() -> None:
    adapters = {
        "dev-model-a": MockModelAdapter("dev-model-a", response_text="success"),
        "dev-model-b": MockModelAdapter(
            "dev-model-b", response_text="unused", error_kind="timeout"),
    }

    def selector(query: str) -> RoutingDecision:
        selected = {"ok": "dev-model-a", "timeout": "dev-model-b"}.get(
            query, "dev-model-unknown")
        return decision(selected)

    records = BatchRunner(BenchmarkRunner(selector, adapters)).run_many([
        ("query-1", "ok"),
        ("query-2", "timeout"),
        ("query-3", "unknown"),
        ("query-4", "ok"),
    ])

    assert [record.status for record in records] == [
        "success", "adapter_error", "unknown_model", "success"]
    assert records[1].error_kind == "timeout"
    assert records[2].selected_model == "dev-model-unknown"
    assert records[3].query_id == "query-4"


def test_duplicate_input_is_reproducible() -> None:
    batch = BatchRunner(make_runner())
    duplicate = ("query-1", "same query")

    first = batch.run_many([duplicate, duplicate])
    second = batch.run_many([duplicate, duplicate])

    assert first == second
    assert first[0] == first[1]
