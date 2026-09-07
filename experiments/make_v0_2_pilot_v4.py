from __future__ import annotations

import argparse
import json
from pathlib import Path

from zhiheng_router.benchmark_v02.pilot_v4 import write_pilot_v4_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Router v0.2 Pilot v4 artifacts")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config", type=Path, default=Path("configs/router_v0_2_pilot_v4.json"))
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else args.root / args.config
    manifest = write_pilot_v4_artifacts(config_path, args.root)
    print(json.dumps({
        "stage": manifest["stage"],
        "pilot_version": manifest["pilot_version"],
        "generator_version": manifest["generator_version"],
        "query_count": manifest["query_count"],
        "task_query_counts": manifest["task_query_counts"],
        "ready_for_final_human_spot_review": manifest[
            "automatic_audit_ready_for_final_human_spot_review"],
    }, indent=2))


if __name__ == "__main__":
    main()
