"""Run fixed-config train/validation/test evaluation on synthetic development data."""

import argparse
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import platform

from threadpoolctl import threadpool_info, threadpool_limits

from zhiheng_router.baselines import Baselines
from zhiheng_router.data import group_split, load_development_dataset, read_config, split_summary
from zhiheng_router.estimates import estimate_candidates, summarize_models
from zhiheng_router.evaluation import evaluate_policy, evaluate_predictor
from zhiheng_router.predictor import QualityPredictor, RoutingArtifact, save_artifact
from zhiheng_router.router import route
from zhiheng_router.schemas import FeatureConfig, ModelCandidate, RoutingPolicy, TrainingConfig

ROOT = Path(__file__).resolve().parents[1]


def run_experiment(config: dict, root: Path = ROOT) -> dict:
    candidates = tuple(ModelCandidate(**row) for row in config["models"])
    active_ids = [c.model_id for c in candidates if c.enabled]
    records, manifest = load_development_dataset(root / config["data_path"], root / config["manifest_path"],
                                                  [c.model_id for c in candidates])
    splits = group_split(records, **config["split"])
    train = splits["train"]
    training = TrainingConfig(**config["training"])
    policy = RoutingPolicy(**config["policy"])
    evaluation = config["evaluation"]
    for key in ("probability_bins", "numeric_threads"):
        if type(evaluation[key]) is not int or evaluation[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(evaluation["timing_warmup_queries"]) is not int or evaluation["timing_warmup_queries"] < 0:
        raise ValueError("timing_warmup_queries must be a nonnegative integer")
    environment = {
        "python": platform.python_version(), "platform": platform.platform(),
        "machine": platform.machine(), "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "dependencies": {name: importlib.metadata.version(name) for name in
                         ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")},
    }
    # Fixed before seeing held-out results; validation is diagnostic, not automatic tuning.
    with threadpool_limits(limits=evaluation["numeric_threads"]):
        environment["numeric_libraries"] = threadpool_info()
        statistics = summarize_models(train, active_ids, training.quality_score_threshold)
        baselines = Baselines(candidates, statistics, policy, config["rules"])
        predictor = QualityPredictor(FeatureConfig(**config["features"]), training).fit(train, active_ids)

        def select(query: str):
            return route(candidates, estimate_candidates(query, active_ids, predictor, statistics), policy)

        selectors = {"query_aware_router": select, "always_strongest": baselines.strongest,
                     "always_lowest_cost": baselines.lowest_cost, "rule_based": baselines.rule_based,
                     "global_historical_router": baselines.global_historical}
        report = {
            "source_kind": "synthetic", "measurement_kind": "simulated",
            "notice": "SYNTHETIC DEVELOPMENT DATA. Probabilities are predictions, not actual quality guarantees.",
            "config": config, "dataset_manifest": manifest, "environment": environment,
            "split": split_summary(splits), "training_counts": predictor.training_counts,
            "training_statistics": {m: asdict(s) for m, s in statistics.items()},
            "selection_protocol": "Fixed, predeclared hyperparameters and policy. No test tuning. No calibration model. No Oracle.",
            "baseline_definition": {
                "always_strongest": {"criterion": "training mean quality", "model_id": baselines.strongest_model},
                "always_lowest_cost": {"criterion": "training mean cost", "model_id": baselines.lowest_cost_model},
                "rule_based": {"criterion": "predeclared Query-only task hints", "rules": config["rules"]},
                "global_historical_router": {"criterion": "training pass rates + same RoutingPolicy, no Query features", "decision": asdict(baselines.global_decision)},
            },
            "predictor": {}, "policies": {},
            "limitations": [
                "All quality/cost/model-latency outcomes are simulated, not actual model measurements.",
                "Per-model cost and latency are fixed training means, not Query-dependent or live estimates.",
                "Full-coverage development data does not solve counterfactual selection bias in production logs.",
                "LR probabilities are uncalibrated; normal routing does not guarantee actual quality.",
                "Fixed/rule baselines have no probability gate: fallback metrics are null (not applicable).",
                "Runtime measurements describe this local serial router only, not Kunpeng or model-serving throughput.",
            ],
        }
        for split_name in ("validation", "test"):
            held = splits[split_name]
            report["predictor"][split_name] = evaluate_predictor(predictor, held, evaluation["probability_bins"])
            warmup = sorted({r.query for r in held})[:evaluation["timing_warmup_queries"]]
            for query in warmup:
                select(query)  # No fitting or statistic update.
            report["policies"][split_name] = {
                name: evaluate_policy(held, selector, training.quality_score_threshold,
                                      measure_runtime=name == "query_aware_router",
                                      probability_bins=evaluation["probability_bins"])
                for name, selector in selectors.items()
            }
    output = root / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    artifact = RoutingArtifact(predictor, statistics, candidates, policy, manifest, config, environment)
    save_artifact(artifact, output / "router.joblib")
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/mvp.json")
    args = parser.parse_args()
    config = read_config(args.config)
    print(json.dumps({"event": "experiment_start", "source_kind": "synthetic",
                      "quality_score_threshold": config["training"]["quality_score_threshold"],
                      "router_probability_threshold": config["policy"]["router_probability_threshold"],
                      "cost_tolerance_abs": config["policy"]["cost_tolerance_abs"]}))
    report = run_experiment(config)
    print("SYNTHETIC DEVELOPMENT DATA | simulated validation/test outcomes")
    print(json.dumps({"training_counts": report["training_counts"],
                      "test": {name: values["metrics"] for name, values in report["policies"]["test"].items()}},
                     ensure_ascii=False, indent=2, allow_nan=False))
    print(f"Report: {ROOT / report['config']['output_dir'] / 'report.json'}")


if __name__ == "__main__":
    main()
