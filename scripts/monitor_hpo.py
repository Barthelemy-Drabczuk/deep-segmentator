#!/usr/bin/env python3
"""Monitor an ongoing Optuna HPO study."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sulcal_seg.hpo.study_manager import StudyManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor Optuna HPO study")
    parser.add_argument("--storage", required=True, help="Path to SQLite database")
    parser.add_argument("--study-name", required=True, help="Optuna study name")
    parser.add_argument("--interval", default=60, type=int, help="Refresh interval in seconds")
    parser.add_argument("--once", action="store_true", help="Print once and exit")
    args = parser.parse_args()

    manager = StudyManager(storage_path=args.storage, study_name=args.study_name)

    while True:
        try:
            summary = manager.summary()
            print(f"\r[{time.strftime('%H:%M:%S')}] "
                  f"Trials: {summary['n_complete']}/{summary['n_trials']} | "
                  f"Best: {summary['best_value']:.4f}", end="", flush=True)
        except Exception as e:
            print(f"\rError reading study: {e}", end="", flush=True)

        if args.once:
            print()
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
