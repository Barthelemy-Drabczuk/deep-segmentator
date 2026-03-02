"""Run nnU-Net inference on test subjects and optionally convert to Morphologist format.

Usage:
    # Stage 1 — smoke test (2 subjects, prediction only)
    python scripts/infer_nnunet.py \\
        --work-dir /neurospin/dico/bdrabczuk/deep_segmentor_nnunet_training/hcp \\
        --output-dir ./outputs/inference \\
        --n-subjects 2

    # Stage 2 — full run + Morphologist .arg files
    pixi run --environment inference python scripts/infer_nnunet.py \\
        --work-dir /neurospin/dico/bdrabczuk/deep_segmentor_nnunet_training/hcp \\
        --output-dir ./outputs/inference \\
        --morphologist

Requirements:
    Prediction : default pixi environment (torch, nnunetv2)
    --morphologist : inference pixi environment (soma.aims, pydantic < 2)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

_TRAINER = "nnUNetTrainer_250epochs"
_PLANS   = "nnUNetResEncUNetPlans"
_CONFIG  = "3d_fullres"
_DATASET = "1"


@click.command()
@click.option(
    "--work-dir",
    required=True,
    type=click.Path(exists=True),
    help="nnU-Net work directory (contains nnunet_raw/, nnunet_results/).",
)
@click.option(
    "--output-dir",
    default="outputs/inference",
    show_default=True,
    help="Directory for predictions and Morphologist output.",
)
@click.option(
    "--n-subjects",
    default=None,
    type=int,
    help="Limit to first N test subjects (omit for all).",
)
@click.option(
    "--morphologist",
    is_flag=True,
    default=False,
    help="Convert predictions to Morphologist .arg format (requires inference env).",
)
@click.option(
    "--checkpoint",
    default="best",
    type=click.Choice(["best", "final"]),
    show_default=True,
    help="Which nnU-Net checkpoint to use.",
)
@click.option(
    "--fold",
    default=0,
    type=int,
    show_default=True,
    help="Fold to use for prediction.",
)
def main(
    work_dir: str,
    output_dir: str,
    n_subjects: int | None,
    morphologist: bool,
    checkpoint: str,
    fold: int,
) -> None:
    """nnU-Net inference → (optionally) Morphologist .arg conversion."""
    _work_dir  = Path(work_dir).resolve()
    _out_dir   = Path(output_dir).resolve()
    pred_dir   = _out_dir / "predictions"
    arg_dir    = _out_dir / "arg_files"
    images_ts  = _work_dir / "nnunet_raw" / "Dataset001_hcp" / "imagesTs"

    pred_dir.mkdir(parents=True, exist_ok=True)
    if morphologist:
        arg_dir.mkdir(parents=True, exist_ok=True)

    # ── Validate inputs ─────────────────────────────────────────────────
    if not images_ts.exists():
        raise FileNotFoundError(
            f"imagesTs not found: {images_ts}\n"
            "Run prepare_nnunet_dataset.py first."
        )

    all_images = sorted(images_ts.glob("*_0000.nii.gz"))
    if not all_images:
        raise FileNotFoundError(f"No *_0000.nii.gz files in {images_ts}")

    if n_subjects is not None:
        selected = all_images[:n_subjects]
        print(f"Staged mode: using {len(selected)} of {len(all_images)} test subjects")
    else:
        selected = all_images
        print(f"Full mode: {len(selected)} test subjects")

    # ── Stage inputs (copy subset to temp dir if needed) ────────────────
    if n_subjects is not None:
        input_dir = pred_dir / "_input_tmp"
        input_dir.mkdir(exist_ok=True)
        for img in selected:
            dst = input_dir / img.name
            if not dst.exists():
                shutil.copy2(img, dst)
    else:
        input_dir = images_ts

    # ── Set nnU-Net env vars ─────────────────────────────────────────────
    env = {
        **os.environ,
        "nnUNet_raw":          str(_work_dir / "nnunet_raw"),
        "nnUNet_preprocessed": str(_work_dir / "nnunet_preprocessed"),
        "nnUNet_results":      str(_work_dir / "nnunet_results"),
    }

    # ── Run nnUNetv2_predict ─────────────────────────────────────────────
    cmd = [
        "nnUNetv2_predict",
        "-i",   str(input_dir),
        "-o",   str(pred_dir),
        "-d",   _DATASET,
        "-c",   _CONFIG,
        "-f",   str(fold),
        "-p",   _PLANS,
        "-chk", f"checkpoint_{checkpoint}.pth",
        "--verbose",
    ]
    print("\n" + "=" * 60)
    print("Step 1/2  Running nnUNetv2_predict …")
    print("  Command:", " ".join(cmd))
    print("=" * 60)

    t0 = time.perf_counter()
    result = subprocess.run(cmd, env=env)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        sys.exit(f"nnUNetv2_predict failed with exit code {result.returncode}")

    predictions = sorted(pred_dir.glob("*.nii.gz"))
    # exclude any files inside _input_tmp/ subdirectory
    predictions = [p for p in predictions if p.parent == pred_dir]
    n_pred = len(predictions)
    print(f"\n  {n_pred} predictions written in {elapsed:.1f} s "
          f"({elapsed / max(n_pred, 1):.1f} s/subject)")

    # ── Morphologist conversion ──────────────────────────────────────────
    if not morphologist:
        print("\nStep 2/2  Skipped (--morphologist not set).")
        print(f"  To convert, re-run with --morphologist in the inference env:")
        print(f"    pixi run --environment inference python {Path(__file__).name} "
              f"--work-dir {work_dir} --output-dir {output_dir} --morphologist")
    else:
        print("\n" + "=" * 60)
        print("Step 2/2  Converting to Morphologist format …")
        print("=" * 60)

        try:
            from sulcal_seg.inference.morphologist_output import (
                segmentation_to_morphologist_format,
            )
            import nibabel as nib
            import numpy as np
        except ImportError as e:
            sys.exit(
                f"Import error: {e}\n"
                "Run with the inference pixi environment:\n"
                "  pixi run --environment inference python scripts/infer_nnunet.py "
                "--work-dir ... --morphologist"
            )

        t1 = time.perf_counter()
        n_converted = 0
        for seg_file in predictions:
            subject_id = seg_file.name.replace(".nii.gz", "")
            mri_path = images_ts / f"{subject_id}_0000.nii.gz"
            if not mri_path.exists():
                # fall back: look in the temp input dir
                mri_path_tmp = input_dir / f"{subject_id}_0000.nii.gz"
                mri_path = mri_path_tmp if mri_path_tmp.exists() else None

            vol = nib.load(str(seg_file)).get_fdata().astype(np.int32)
            try:
                paths = segmentation_to_morphologist_format(
                    segmentation=vol,
                    output_dir=arg_dir,
                    mri_path=mri_path,
                    subject_id=subject_id,
                )
                n_converted += 1
                print(f"  [{n_converted}/{n_pred}] {subject_id}  "
                      f"→ {paths['arg'].name}")
            except Exception as exc:
                print(f"  [WARN] {subject_id}: {exc}", file=sys.stderr)

        elapsed2 = time.perf_counter() - t1
        print(f"\n  {n_converted}/{n_pred} subjects converted in {elapsed2:.1f} s")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Summary")
    print(f"  Predictions : {pred_dir}")
    if morphologist:
        print(f"  ARG files   : {arg_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
