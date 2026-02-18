#!/usr/bin/env python3
"""Preprocess all datasets (intensity normalisation + validation)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess all datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ukbiobank", "abcd", "abide", "senior"],
        choices=["ukbiobank", "abcd", "abide", "senior"],
        help="Datasets to preprocess"
    )
    parser.add_argument("--normalisation", default="zscore", choices=["zscore", "minmax", "percentile"])
    parser.add_argument("--validate", action="store_true", default=True, help="Run data validation")
    args = parser.parse_args()

    from sulcal_seg.data.preprocessing.normalizer import IntensityNormalizer
    from sulcal_seg.data.preprocessing.validator import DataValidator

    normalizer = IntensityNormalizer(method=args.normalisation)
    validator = DataValidator()

    print(f"Preprocessing {len(args.datasets)} datasets with {args.normalisation} normalisation...")

    # TODO: Load each dataset via DatasetManager, iterate subjects,
    #   apply normalizer.normalize() and validator.check_all()
    print("NOTE: Actual preprocessing requires data on disk.")
    print("Configured normalizer:", normalizer)
    print("Configured validator:", validator)


if __name__ == "__main__":
    main()
