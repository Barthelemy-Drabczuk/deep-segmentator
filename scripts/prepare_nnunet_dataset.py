"""Prepare HCP (or other dataset) data for nnUNetV2Runner.

Reads data via the registered AbstractDataLoader (no files written to data_root).
All outputs go to --work-dir, which should be OUTSIDE the git repository.

Outputs (all under --work-dir):
  images/   {subject_id}_0000.nii.gz   symlink → skull-stripped T1 (already on disk)
  labels/   {subject_id}.nii.gz        combined L+R grey/white label  (created)
  datalist.json                         MONAI datalist with 5-fold CV assignments

Usage:
    python scripts/prepare_nnunet_dataset.py \\
        --data-root /neurospin/dico/data/bv_databases/human/not_labeled/hcp/hcp \\
        --work-dir  ~/nnunet_work/hcp

    # Custom number of folds:
    python scripts/prepare_nnunet_dataset.py \\
        --data-root /path/to/hcp/hcp \\
        --work-dir  ~/nnunet_work/hcp \\
        --n-folds 5
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click

# Ensure src/ is importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@click.command()
@click.option(
    "--data-root",
    required=True,
    help="Root directory of the HCP per-subject data (passed to HCPLoader as root_dir).",
)
@click.option(
    "--work-dir",
    default=str(Path.home() / "nnunet_work" / "hcp"),
    show_default=True,
    help="Output directory for prepared data. Must be OUTSIDE the git repository.",
)
@click.option(
    "--n-folds",
    default=5,
    show_default=True,
    type=int,
    help="Number of cross-validation folds to assign (round-robin over sorted subject IDs).",
)
def main(data_root: str, work_dir: str, n_folds: int) -> None:
    """Prepare HCP data for nnUNetV2Runner (run once before training)."""
    import nibabel as nib
    import numpy as np

    from sulcal_seg.data.loaders.hcp_loader import HCPLoader

    work = Path(work_dir).expanduser().resolve()
    images_dir = work / "images"
    labels_dir = work / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("HCP → nnU-Net dataset preparation")
    print("=" * 60)
    print(f"  Data root : {data_root}")
    print(f"  Work dir  : {work}")
    print(f"  CV folds  : {n_folds}")
    print()

    # ── Collect all subjects across train / val / test splits ────────
    all_sids: list[tuple[str, str]] = []   # (subject_id, subset)
    for subset in ("train", "val", "test"):
        loader = HCPLoader(root_dir=data_root, subset=subset)
        sids = loader.get_subject_ids()
        print(f"  {subset:5s}: {len(sids)} subjects")
        for sid in sids:
            all_sids.append((sid, subset))

    print(f"\n  Total: {len(all_sids)} subjects\n")

    # ── Materialise images (symlinks) and labels (new NIfTIs) ────────
    n_created = 0
    n_skipped = 0
    skipped_sids: set[str] = set()  # subjects with no readable T1 — excluded from datalist
    for sid, subset in all_sids:
        loader = HCPLoader(root_dir=data_root, subset=subset)

        # Image symlink → skull-stripped T1 (already on disk, not copied)
        src_img = (
            Path(data_root)
            / sid / "t1mri" / "BL"
            / "default_analysis" / "segmentation"
            / f"skull_stripped_{sid}.nii.gz"
        )
        dst_img = images_dir / f"{sid}_0000.nii.gz"
        if not dst_img.exists():
            if src_img.exists():
                dst_img.symlink_to(src_img)
            else:
                # Fall back to raw T1 if skull-stripped is missing
                raw = Path(data_root) / sid / "t1mri" / "BL" / f"{sid}.nii.gz"
                if not raw.exists():
                    print(f"  WARNING: no T1 found for {sid} (checked skull-stripped and raw) — skipping")
                    skipped_sids.add(sid)
                    continue
                dst_img.symlink_to(raw)

        # Combined L+R grey/white label (created in memory by HCPLoader, saved here)
        dst_lbl = labels_dir / f"{sid}.nii.gz"
        if not dst_lbl.exists():
            label = loader.load_morphologist_label(sid)   # (H,W,D) int32
            ref_img = nib.load(str(src_img) if src_img.exists() else str(dst_img.resolve()))
            nib.save(nib.Nifti1Image(label.astype(np.int16), ref_img.affine), str(dst_lbl))
            n_created += 1
        else:
            n_skipped += 1

    if skipped_sids:
        print(f"  WARNING: {len(skipped_sids)} subject(s) skipped (no T1 image found)")
    print(f"  Labels created : {n_created}")
    print(f"  Labels skipped (already exist): {n_skipped}")

    # ── Build MONAI datalist with fold assignments ────────────────────
    all_sids.sort(key=lambda x: x[0])   # deterministic order by subject ID

    training_entries: list[dict] = []
    test_entries: list[dict] = []

    for i, (sid, subset) in enumerate(all_sids):
        if sid in skipped_sids:
            continue
        img_path = str(images_dir / f"{sid}_0000.nii.gz")
        lbl_path = str(labels_dir / f"{sid}.nii.gz")

        if subset == "test":
            test_entries.append({"image": img_path})
        else:
            training_entries.append({
                "image": img_path,
                "label": lbl_path,
                "fold":  i % n_folds,
            })

    datalist = {"training": training_entries, "test": test_entries}
    datalist_path = work / "datalist.json"
    datalist_path.write_text(json.dumps(datalist, indent=2))

    print(f"\n  Datalist written: {datalist_path}")
    print(f"    training entries : {len(training_entries)}")
    print(f"    test entries     : {len(test_entries)}")
    print()
    print("Done. Pass --work-dir to train_monai_nnunet.py:")
    print(f"  python scripts/train_monai_nnunet.py --config configs/training.yaml")


if __name__ == "__main__":
    main()
