#!/usr/bin/env python3
"""Create required output and data directory structure."""
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).parent.parent

    dirs = [
        "data/ukbiobank",
        "data/abcd",
        "data/abide",
        "data/senior",
        "outputs/checkpoints",
        "outputs/logs",
        "outputs/hpo",
        "outputs/validation",
        "outputs/ablations",
        "slurm_logs",
    ]

    print("Creating directory structure...")
    for d in dirs:
        path = root / d
        path.mkdir(parents=True, exist_ok=True)
        print(f"  {path}")

    print("\nSetup complete.")
    print("Next steps:")
    print("  1. Edit configs/datasets/ to point to your data directories")
    print("  2. Run: pixi run preprocess")
    print("  3. Run: pixi run train-full")


if __name__ == "__main__":
    main()
