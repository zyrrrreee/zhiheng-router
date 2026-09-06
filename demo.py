"""Interactive routing demo; loads a trained artifact and never invokes a real LLM."""

import argparse
from pathlib import Path

from threadpoolctl import threadpool_limits

from zhiheng_router.estimates import estimate_candidates
from zhiheng_router.predictor import load_artifact
from zhiheng_router.router import route

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=ROOT / "outputs/mvp/router.joblib")
    parser.add_argument("--query", help="Omit for interactive input")
    args = parser.parse_args()
    if not args.artifact.is_file():
        parser.error("trained artifact not found; run python experiments/run.py first")
    artifact = load_artifact(args.artifact)
    if artifact.dataset_manifest["source_kind"] != "synthetic":
        parser.error("this demo is scoped to synthetic development data")
    try:
        query = args.query if args.query is not None else input("Query: ")
        ids = [c.model_id for c in artifact.candidates if c.enabled]
        with threadpool_limits(limits=artifact.config["evaluation"]["numeric_threads"]):
            estimates = estimate_candidates(query, ids, artifact.predictor, artifact.statistics)
            decision = route(artifact.candidates, estimates, artifact.policy)
    except (ValueError, EOFError) as exc:
        parser.error(str(exc))
    print("Dataset: SYNTHETIC DEVELOPMENT DATA")
    print(f"Query: {query}")
    print(f"quality_score_threshold: {artifact.predictor.quality_score_threshold}")
    print(f"router_probability_threshold: {artifact.policy.router_probability_threshold}")
    print(f"cost_tolerance_abs: {artifact.policy.cost_tolerance_abs}")
    print(f"Cost unit: {artifact.dataset_manifest['cost_unit']}; latency unit: ms")
    print("model_id       pass_probability  estimated_cost  estimated_latency_ms")
    for e in decision.candidate_estimates:
        print(f"{e.model_id:14} {e.pass_probability:16.6f} {e.estimated_cost:15.8f} {e.estimated_latency_ms:21.3f}")
    disabled = [c.model_id for c in artifact.candidates if not c.enabled]
    if disabled:
        print(f"Statically disabled candidates: {', '.join(disabled)}")
    print(f"Qualified Models: {', '.join(decision.qualified_models) or '(none)'}")
    print(f"Selected Model: {decision.selected_model}")
    print(f"Fallback Used: {str(decision.fallback_used).lower()}")
    print(f"Decision Reason: {decision.decision_reason}")
    if decision.fallback_used:
        print("预测质量门槛未满足，启用 fallback。")
    print("Cost / latency are per-model training means and do not vary with this Query.")
    print("This is a routing decision on synthetic development data.")
    print("Predicted pass probability is not an actual quality guarantee.")


if __name__ == "__main__":
    main()
