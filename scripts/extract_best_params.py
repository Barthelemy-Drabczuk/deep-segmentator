#!/usr/bin/env python3
"""Extract best Optuna trial parameters and write to YAML."""
import argparse
import sys
from pathlib import Path

# Allow running as: python scripts/extract_best_params.py
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sulcal_seg.hpo.study_manager import StudyManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Export best Optuna trial to YAML")
    parser.add_argument("--storage", required=True, help="Path to SQLite database")
    parser.add_argument("--study-name", required=True, help="Optuna study name")
    parser.add_argument("--output", default="outputs/hpo/best_params.yaml", help="Output YAML path")
    args = parser.parse_args()

    manager = StudyManager(storage_path=args.storage, study_name=args.study_name)
    summary = manager.summary()
    print(f"Study: {summary['study_name']}")
    print(f"Completed trials: {summary['n_complete']} / {summary['n_trials']}")
    print(f"Best value: {summary['best_value']:.4f}")

    output_path = manager.export_best_params(args.output)
    print(f"Best params saved to: {output_path}")


if __name__ == "__main__":
    main()
