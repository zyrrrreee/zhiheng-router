"""Analyze real-validation records without model or network access."""

from collections import Counter
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RECORDS_PATH = ROOT / "outputs" / "real_validation" / "records.jsonl"
OUTPUT_PATH = ROOT / "outputs" / "real_validation" / "analysis_report.json"
REQUIRED_FIELDS = {
    "strategy",
    "query_id",
    "query_text",
    "selected_model",
    "status",
    "cost",
    "latency_ms",
    "input_tokens",
    "output_tokens",
}


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"records file not found: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"line {line_number} must contain a JSON object")
        missing = REQUIRED_FIELDS - record.keys()
        if missing:
            raise ValueError(f"line {line_number} is missing fields: {sorted(missing)}")
        records.append(record)
    if not records:
        raise ValueError("records file is empty")
    return records


def _finite_values(records: list[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = record[field]
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field} must contain only numbers or null")
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"{field} must contain finite nonnegative values")
        values.append(number)
    return values


def _average(records: list[dict[str, Any]], field: str) -> float | None:
    values = _finite_values(records, field)
    return fmean(values) if values else None


def _model_statistics(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for model_id in sorted({record["selected_model"] for record in records}):
        selected = [record for record in records if record["selected_model"] == model_id]
        successful = [record for record in selected if record["status"] == "success"]
        result[model_id] = {
            "query_count": len(selected),
            "success_count": len(successful),
            "average_cost": _average(successful, "cost"),
            "average_latency_ms": _average(successful, "latency_ms"),
            "average_input_tokens": _average(successful, "input_tokens"),
            "average_output_tokens": _average(successful, "output_tokens"),
        }
    return result


def _strategy_statistics(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for strategy in sorted({record["strategy"] for record in records}):
        rows = [record for record in records if record["strategy"] == strategy]
        successful = [record for record in rows if record["status"] == "success"]
        model_counts = Counter(record["selected_model"] for record in rows)
        result[strategy] = {
            "query_count": len(rows),
            "success_count": len(successful),
            "failure_count": len(rows) - len(successful),
            "selected_model_distribution": dict(sorted(model_counts.items())),
            "average_cost": _average(successful, "cost"),
            "average_latency_ms": _average(successful, "latency_ms"),
            "average_input_tokens": _average(successful, "input_tokens"),
            "average_output_tokens": _average(successful, "output_tokens"),
            "selected_model_statistics": _model_statistics(rows),
        }
    return result


def _query_summary(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "query_id": record["query_id"],
        "query_text": record["query_text"],
        "selected_model": record["selected_model"],
        "cost": record["cost"],
        "latency_ms": record["latency_ms"],
        "output_tokens": record["output_tokens"],
    }


def _top_five(records: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    eligible = [record for record in records
                if record["status"] == "success" and record[field] is not None]
    ordered = sorted(
        eligible,
        key=lambda record: (-float(record[field]), record["query_id"]),
    )
    return [_query_summary(record) for record in ordered[:5]]


def _router_analysis(records: list[dict[str, Any]]) -> dict[str, Any]:
    router = [record for record in records if record["strategy"] == "router"]
    if not router:
        raise ValueError("records do not contain the router strategy")
    queries_by_model = {
        model_id: [
            {"query_id": record["query_id"], "query_text": record["query_text"]}
            for record in router if record["selected_model"] == model_id
        ]
        for model_id in ("deepseek-v4-flash", "deepseek-v4-pro")
    }
    return {
        "deepseek-v4-flash_selection_count": len(queries_by_model["deepseek-v4-flash"]),
        "deepseek-v4-pro_selection_count": len(queries_by_model["deepseek-v4-pro"]),
        "queries_by_selected_model": queries_by_model,
        "top_5_by_cost": _top_five(router, "cost"),
        "top_5_by_latency": _top_five(router, "latency_ms"),
    }


def _router_vs_strongest(strategy_stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if "router" not in strategy_stats or "strongest" not in strategy_stats:
        raise ValueError("records must contain router and strongest strategies")
    router = strategy_stats["router"]
    strongest = strategy_stats["strongest"]
    fields = (
        "average_cost",
        "average_latency_ms",
        "average_input_tokens",
        "average_output_tokens",
    )
    return {
        field: {
            "router": router[field],
            "strongest": strongest[field],
            "router_minus_strongest": (
                router[field] - strongest[field]
                if router[field] is not None and strongest[field] is not None else None
            ),
        }
        for field in fields
    }


def _format_number(value: float | None, precision: int) -> str:
    return "null" if value is None else f"{value:.{precision}f}"


def main() -> None:
    records = _load_records(RECORDS_PATH)
    strategy_stats = _strategy_statistics(records)
    report = {
        "input_file": str(RECORDS_PATH.relative_to(ROOT)),
        "record_count": len(records),
        "averages_scope": "successful records with non-null measurements",
        "cost_unit": "USD",
        "latency_unit": "ms",
        "strategy_statistics": strategy_stats,
        "router_analysis": _router_analysis(records),
        "router_vs_strongest": _router_vs_strongest(strategy_stats),
    }
    OUTPUT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    for strategy, stats in strategy_stats.items():
        print(
            f"{strategy}: queries={stats['query_count']}, "
            f"success={stats['success_count']}, failures={stats['failure_count']}, "
            f"models={stats['selected_model_distribution']}, "
            f"avg_cost={_format_number(stats['average_cost'], 10)}, "
            f"avg_latency_ms={_format_number(stats['average_latency_ms'], 3)}, "
            f"avg_input_tokens={_format_number(stats['average_input_tokens'], 2)}, "
            f"avg_output_tokens={_format_number(stats['average_output_tokens'], 2)}"
        )
    print(f"Report: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
