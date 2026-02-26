"""Phase 1, Week 3 — Evaluation script for trained MONAI nnU-Net.

Loads a saved checkpoint, runs inference on the test set (500 subjects),
computes all 13 required metrics, generates sample ARG files, and writes
a summary report.

Usage:
    python scripts/evaluate_nnunet.py \\
        --checkpoint outputs/checkpoints/best.pt \\
        --config     configs/training.yaml \\
        --output-dir outputs/evaluation/

    # Generate ARG files for first 50 subjects:
    python scripts/evaluate_nnunet.py \\
        --checkpoint outputs/checkpoints/best.pt \\
        --config     configs/training.yaml \\
        --output-dir outputs/evaluation/ \\
        --n-arg 50

Phase 1 gate:
    PASS : Test Dice ≥ 95 %,  training stable,  GPU inference works
    FAIL : Any gate criterion not met → debug and retest
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import click
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# ── 13 required metrics (names) ─────────────────────────────────────────────
_METRIC_NAMES: List[str] = [
    "dice_mean",               # 1.  Mean Dice across all classes
    "dice_per_class",          # 2.  Per-sulcus Dice (list)
    "hausdorff_distance",      # 3.  95th-pct Hausdorff (mm)
    "sensitivity",             # 4.  Macro sensitivity
    "specificity",             # 5.  Macro specificity
    "volumetric_similarity",   # 6.  Volumetric similarity
    "surface_distance_mean",   # 7.  Mean surface distance (mm)
    "connectivity",            # 8.  Connected-component analysis
    "symmetry_check",          # 9.  L/R sulcal symmetry score
    "anatomical_plausibility", # 10. Heuristic label consistency
    "inference_time_gpu_s",    # 11. Seconds per subject (GPU)
    "peak_gpu_memory_mb",      # 12. Peak GPU memory usage (MB)
    "per_sulcus_accuracy",     # 13. Per-sulcus voxel accuracy
]


@click.command()
@click.option(
    "--checkpoint",
    required=True,
    type=click.Path(exists=True),
    help="Path to saved model checkpoint (.pt).",
)
@click.option(
    "--config",
    "config_path",
    default="configs/training.yaml",
    show_default=True,
    help="Training YAML config.",
)
@click.option(
    "--output-dir",
    default="outputs/evaluation",
    show_default=True,
    help="Directory to write results and ARG files.",
)
@click.option(
    "--n-subjects",
    default=500,
    show_default=True,
    help="Number of test subjects to evaluate.",
)
@click.option(
    "--n-arg",
    default=50,
    show_default=True,
    help="Number of subjects for which to generate ARG files.",
)
@click.option("--cpu", is_flag=True, default=False, help="Force CPU inference.")
def main(
    checkpoint: str,
    config_path: str,
    output_dir: str,
    n_subjects: int,
    n_arg: int,
    cpu: bool,
) -> None:
    """Evaluate MONAI nnU-Net on the test set and write metrics + ARG files."""
    import torch

    from sulcal_seg.models.monai_nnunet import MONAInnUNetModel
    from sulcal_seg.validation.metrics import compute_all_metrics

    device = torch.device("cpu") if cpu else (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arg_dir = out_dir / "arg_files"
    arg_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("MONAI nnU-Net — Phase 1 Evaluation")
    print("=" * 60)
    print(f"  Checkpoint   : {checkpoint}")
    print(f"  Device       : {device}")
    print(f"  Test subjects: {n_subjects}")
    print(f"  ARG files    : {n_arg}")
    print()

    # ── Load config ────────────────────────────────────────────────────
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    data_cfg: dict = cfg.get("data", {})
    model_cfg: dict = cfg.get("model", {})

    # ── Load model ─────────────────────────────────────────────────────
    print("Loading model …")
    model = MONAInnUNetModel(model_cfg)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state.get("model_state_dict", state))
    model.to(device)
    model.eval()
    print("  Model loaded OK")

    # ── Load test data ─────────────────────────────────────────────────
    print("Loading test data …")
    # TODO: Build test DataLoader from existing UKBiobankLoader
    # from sulcal_seg.data.loaders.ukbiobank_loader import UKBiobankLoader
    # test_loader = ...
    raise NotImplementedError(
        "Connect the UK Biobank test DataLoader here.\n"
        "  Expected batch: {'image': Tensor(B,1,256,256,256),\n"
        "                   'label': Tensor(B,1,256,256,256),\n"
        "                   'subject_id': List[str]}\n"
        "See src/sulcal_seg/data/loaders/ukbiobank_loader.py\n\n"
        "Once connected, replace this raise with the evaluation loop below:\n"
        "\n"
        "  all_metrics = []\n"
        "  for i, batch in enumerate(test_loader):\n"
        "      if i >= n_subjects: break\n"
        "      t0 = time.perf_counter()\n"
        "      with torch.no_grad():\n"
        "          probs = model.get_probabilities(batch['image'].to(device))\n"
        "      elapsed = time.perf_counter() - t0\n"
        "      pred = probs.argmax(dim=1, keepdim=True).cpu().numpy()\n"
        "      label = batch['label'].numpy()\n"
        "      m = compute_all_metrics(pred, label)\n"
        "      m['inference_time_gpu_s'] = elapsed\n"
        "      all_metrics.append(m)\n"
        "      if i < n_arg:\n"
        "          _generate_arg_file(probs, batch, arg_dir)\n"
        "\n"
        "  _write_summary(all_metrics, out_dir)"
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _generate_arg_file(probs, batch: dict, arg_dir: Path) -> None:
    """
    Write a Morphologist-compatible ARG file for one subject.

    TODO: Call segmentation_to_morphologist_format() from
          src/sulcal_seg/inference/morphologist_output.py once
          the full implementation is in place.
    """
    from sulcal_seg.inference.morphologist_output import (
        segmentation_to_morphologist_format,
    )
    import numpy as np

    subject_id = batch.get("subject_id", ["unknown"])[0]
    seg_np = probs[0].cpu().numpy()  # (52, D, H, W)
    segmentation_to_morphologist_format(
        segmentation=seg_np,
        output_dir=arg_dir,
        subject_id=subject_id,
    )


def _write_summary(all_metrics: List[Dict], out_dir: Path) -> None:
    """Aggregate metrics and write results.json + summary.txt."""
    import numpy as np

    n = len(all_metrics)
    dice_scores = [m.get("macro_dice", 0.0) for m in all_metrics]
    mean_dice = float(np.mean(dice_scores))
    inf_times = [m.get("inference_time_gpu_s", 0.0) for m in all_metrics]
    mean_time = float(np.mean(inf_times))

    summary = {
        "n_subjects": n,
        "mean_dice": mean_dice,
        "std_dice": float(np.std(dice_scores)),
        "min_dice": float(np.min(dice_scores)),
        "max_dice": float(np.max(dice_scores)),
        "mean_inference_time_s": mean_time,
        "gate_pass": mean_dice >= 0.95,
        "per_subject": all_metrics,
    }

    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    txt_path = out_dir / "summary.txt"
    gate = "PASS ✓" if summary["gate_pass"] else "FAIL ✗"
    with open(txt_path, "w") as f:
        f.write("PHASE 1 EVALUATION SUMMARY\n")
        f.write("=" * 40 + "\n")
        f.write(f"Subjects evaluated : {n}\n")
        f.write(f"Mean Dice          : {mean_dice * 100:.2f} %\n")
        f.write(f"Std  Dice          : {float(np.std(dice_scores)) * 100:.2f} %\n")
        f.write(f"Mean inference time: {mean_time:.1f} s/subject\n")
        f.write(f"Gate (Dice ≥ 95%)  : {gate}\n")

    print()
    print("=" * 40)
    print(f"  Mean Dice          : {mean_dice * 100:.2f} %")
    print(f"  Mean inference time: {mean_time:.1f} s/subject")
    print(f"  Gate result        : {gate}")
    print(f"  Results written to : {results_path}")
    print("=" * 40)


if __name__ == "__main__":
    main()
