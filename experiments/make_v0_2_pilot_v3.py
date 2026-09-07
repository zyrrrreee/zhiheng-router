"""Generate Router v0.2 Pilot v3 calibration artifacts without outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from zhiheng_router.benchmark_v02.pilot_v3 import write_pilot_v3_artifacts


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/router_v0_2_pilot_v3.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else args.root / args.config
    manifest = write_pilot_v3_artifacts(config_path, args.root)
    print(json.dumps({
        "stage": manifest["stage"],
        "pilot_version": manifest["pilot_version"],
        "generator_version": manifest["generator_version"],
        "query_count": manifest["query_count"],
        "task_query_counts": manifest["task_query_counts"],
        "ready_for_final_spot_review": (
            manifest["automatic_audit_ready_for_final_spot_review"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
