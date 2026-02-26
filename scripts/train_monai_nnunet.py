"""Phase 1 — Train nnU-Net v2 via MONAI nnUNetV2Runner.

Usage (on GPU node):
    # Step 1 — prepare data once (CPU only, run before training):
    python scripts/prepare_nnunet_dataset.py \\
        --data-root /neurospin/dico/data/bv_databases/human/not_labeled/hcp/hcp \\
        --work-dir  ~/nnunet_work/hcp

    # Step 2 — train:
    python scripts/train_monai_nnunet.py --config configs/training.yaml

    # Override work dir or fold:
    python scripts/train_monai_nnunet.py \\
        --config configs/training.yaml \\
        --work-dir ~/nnunet_work/hcp \\
        --fold 0

Requirements:
    pip install "monai>=1.2" "nnunetv2>=2.2"

Expected outcome (Phase 1 gate):
    Validation Dice ≥ 95 %
    Best checkpoint saved under work_dir/nnunet_results/
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import yaml

# Ensure src/ is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

_DEFAULT_WORK_DIR = str(Path.home() / "nnunet_work" / "hcp")


@click.command()
@click.option(
    "--config",
    "config_path",
    default="configs/training.yaml",
    show_default=True,
    help="Path to training YAML config.",
)
@click.option(
    "--data-root",
    default=None,
    help="Override data.root_dir (HCP per-subject directory).",
)
@click.option(
    "--work-dir",
    default=None,
    help=f"Override data.work_dir (nnU-Net staging dir, outside git repo). "
         f"Default: {_DEFAULT_WORK_DIR}",
)
@click.option(
    "--fold",
    default=None,
    type=int,
    help="Override training.fold (0–4 for 5-fold CV). Default: 0.",
)
def main(
    config_path: str,
    data_root: str | None,
    work_dir: str | None,
    fold: int | None,
) -> None:
    """Train nnU-Net v2 on HCP sulcal segmentation data via MONAI nnUNetV2Runner."""
    # ── Load config ────────────────────────────────────────────────────
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    training_cfg: dict = cfg.get("training", {})
    data_cfg: dict = cfg.get("data", {})

    # Apply CLI overrides
    if data_root is not None:
        data_cfg["root_dir"] = data_root
    if work_dir is not None:
        data_cfg["work_dir"] = work_dir
    if fold is not None:
        training_cfg["fold"] = fold

    # Resolve work_dir (expand ~ and make absolute)
    _work_dir = Path(
        data_cfg.get("work_dir", _DEFAULT_WORK_DIR)
    ).expanduser().resolve()

    # ── Deferred imports (so --help works without GPU / MONAI) ─────────
    import torch
    from monai.apps.nnunet import nnUNetV2Runner

    print("=" * 60)
    print("nnU-Net v2 — Phase 1 Training  (via MONAI nnUNetV2Runner)")
    print("=" * 60)
    print(f"  Config       : {config_path}")
    print(f"  Work dir     : {_work_dir}")
    print(f"  GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU          : {torch.cuda.get_device_name(0)}")
    print()

    # ── Validate datalist exists ────────────────────────────────────────
    datalist_path = _work_dir / "datalist.json"
    if not datalist_path.exists():
        raise FileNotFoundError(
            f"datalist.json not found at {datalist_path}.\n"
            "Run the preparation script first:\n"
            "  python scripts/prepare_nnunet_dataset.py "
            f"--data-root <HCP_ROOT> --work-dir {_work_dir}"
        )

    # ── Build nnUNetV2Runner input config ──────────────────────────────
    input_config = {
        "modality":            {"0": "MRI"},      # single T1 channel
        "datalist":            str(datalist_path),
        "dataroot":            str(_work_dir),
        "nnunet_raw":          str(_work_dir / "nnunet_raw"),
        "nnunet_preprocessed": str(_work_dir / "nnunet_preprocessed"),
        "nnunet_results":      str(_work_dir / "nnunet_results"),
        "dataset_name_or_id":  1,
    }

    trainer_class = training_cfg.get("nnunet_trainer", "nnUNetTrainer_250epochs")
    _fold         = training_cfg.get("fold", 0)
    config_name   = training_cfg.get("nnunet_config", "3d_fullres")

    print(f"  Trainer      : {trainer_class}")
    print(f"  Config       : {config_name}")
    print(f"  Fold         : {_fold}")
    print()

    runner = nnUNetV2Runner(
        input_config=input_config,
        trainer_class_name=trainer_class,
        work_dir=str(_work_dir / "runner_work"),
    )

    # ── Convert dataset (idempotent — skipped if raw dir already exists) ──
    raw_dir = Path(input_config["nnunet_raw"])
    if not raw_dir.exists():
        print("Step 1/3  Converting dataset to nnU-Net format …")
        runner.convert_dataset()
    else:
        print("Step 1/3  nnU-Net raw dir already exists — skipping conversion.")

    # ── Plan & preprocess (idempotent) ─────────────────────────────────
    preprocessed_dir = Path(input_config["nnunet_preprocessed"])
    if not preprocessed_dir.exists():
        print("Step 2/3  Planning and preprocessing …")
        runner.plan_and_process()
    else:
        print("Step 2/3  Preprocessed dir already exists — skipping.")

    # ── Train single model (fold N, 3d_fullres) ────────────────────────
    print(f"Step 3/3  Training {config_name}, fold {_fold} …")
    runner.train_single_model(config=config_name, fold=_fold)

    results_dir = Path(input_config["nnunet_results"])
    print(f"\nTraining complete. Results saved to: {results_dir}")
    print("To train remaining folds (1–4), rerun with --fold 1, --fold 2, etc.")


if __name__ == "__main__":
    main()
