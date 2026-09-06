"""Generate the Router v0.2 Pilot Surface Set; no outcomes or Router runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from zhiheng_router.benchmark_v02.pilot import write_pilot_artifacts


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/router_v0_2_benchmark.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else args.root / args.config
    manifest = write_pilot_artifacts(config_path, args.root)
    print(json.dumps({
        "stage": manifest["stage"],
        "query_count": manifest["query_count"],
        "task_query_counts": manifest["task_query_counts"],
        "ready_for_human_review": manifest["automatic_audit_ready_for_human_review"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
