#!/usr/bin/env python3
"""Monitor training progress via checkpoint metrics."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor training via checkpoints")
    parser.add_argument("--checkpoint-dir", required=True, help="Checkpoint directory")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.exists():
        print(f"Checkpoint directory does not exist: {ckpt_dir}")
        sys.exit(1)

    checkpoints = sorted(ckpt_dir.glob("*.pt"))
    if not checkpoints:
        print("No checkpoints found.")
        return

    print(f"Found {len(checkpoints)} checkpoint(s) in {ckpt_dir}:")
    for ckpt in checkpoints:
        print(f"  {ckpt.name} ({ckpt.stat().st_size / 1e6:.1f} MB)")

    best = ckpt_dir / "best.pt"
    if best.exists():
        print(f"\nBest checkpoint: {best} ({best.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
