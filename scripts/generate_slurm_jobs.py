#!/usr/bin/env python3
"""Generate SLURM job scripts for HPO."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sulcal_seg.hpo.cluster_manager import ClusterManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SLURM HPO job scripts")
    parser.add_argument("--study-name", required=True, help="Optuna study name")
    parser.add_argument("--storage", required=True, help="SQLite database path")
    parser.add_argument("--mode", choices=["single", "multi"], default="single")
    parser.add_argument("--n-nodes", default=4, type=int, help="Number of array tasks (multi mode)")
    parser.add_argument("--n-trials", default=20, type=int, help="Trials per node")
    parser.add_argument("--output-dir", default="slurm_jobs", help="Output directory for scripts")
    args = parser.parse_args()

    manager = ClusterManager(
        study_name=args.study_name,
        storage_path=args.storage,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "single":
        script_path = out_dir / f"{args.study_name}_single.slurm"
        manager.generate_single_node_script(
            n_trials=args.n_trials * args.n_nodes,
            output_path=str(script_path),
        )
        print(f"Generated: {script_path}")
    else:
        script_path = out_dir / f"{args.study_name}_array.slurm"
        manager.generate_multi_node_script(
            n_nodes=args.n_nodes,
            trials_per_node=args.n_trials,
            output_path=str(script_path),
        )
        print(f"Generated: {script_path}")
    print(f"Submit with: sbatch {script_path}")


if __name__ == "__main__":
    main()
