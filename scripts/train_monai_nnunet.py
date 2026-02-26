"""Phase 1 — Train nnU-Net v2 via MONAI nnUNetV2Runner.

Usage (on GPU node):
    # Step 1 — prepare data once (CPU only, run before training):
    python scripts/prepare_nnunet_dataset.py \\
        --data-root /neurospin/dico/data/bv_databases/human/not_labeled/hcp/hcp \\
        --work-dir  ~/nnunet_work/hcp

    # Step 2 — train (full dataset):
    python scripts/train_monai_nnunet.py --config configs/training.yaml

    # Subset / staged training (quick sanity check on 50 subjects):
    python scripts/train_monai_nnunet.py \\
        --config configs/training.yaml \\
        --max-subjects 50

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

import json
import shutil
import sys
from pathlib import Path

import click
import yaml

# Ensure src/ is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

_DEFAULT_WORK_DIR = str(Path.home() / "nnunet_work" / "hcp")


def _dataset_subject_count(raw_dir: Path) -> int | None:
    """Return number of training images in the existing Dataset001_* folder, or None."""
    folders = list(raw_dir.glob("Dataset001_*/imagesTr"))
    if not folders:
        return None
    return len(list(folders[0].glob("*.nii.gz")))


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
@click.option(
    "--max-subjects",
    default=None,
    type=int,
    help="Limit training to the first N subjects (staged / subset training). "
         "Override data.max_subjects from config. If the cached dataset has a "
         "different subject count it is automatically rebuilt.",
)
def main(
    config_path: str,
    data_root: str | None,
    work_dir: str | None,
    fold: int | None,
    max_subjects: int | None,
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
    if max_subjects is not None:
        data_cfg["max_subjects"] = max_subjects

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

    # ── Apply subject limit (staged / subset training) ─────────────────
    _max_subjects = data_cfg.get("max_subjects")
    if _max_subjects is not None:
        _max_subjects = int(_max_subjects)
        raw_dl = json.loads(datalist_path.read_text())
        raw_dl["training"] = raw_dl["training"][:_max_subjects]
        filtered_path = _work_dir / f"datalist_{_max_subjects}s.json"
        filtered_path.write_text(json.dumps(raw_dl, indent=2))
        effective_datalist = filtered_path
        n_train = _max_subjects
        print(f"  Subjects     : {_max_subjects} (subset mode)")
    else:
        effective_datalist = datalist_path
        raw_dl = json.loads(datalist_path.read_text())
        n_train = len(raw_dl.get("training", []))
        print(f"  Subjects     : {n_train} (full dataset)")

    # ── Training config ─────────────────────────────────────────────────
    trainer_class = training_cfg.get("nnunet_trainer", "nnUNetTrainer_250epochs")
    _fold = training_cfg.get("fold", 0)
    config_name = training_cfg.get("nnunet_config", "3d_fullres")
    planner_name = training_cfg.get("nnunet_planner", "ExperimentPlanner")
    plans_name = training_cfg.get("nnunet_plans", None)

    print(f"  Trainer      : {trainer_class}")
    print(f"  Config       : {config_name}")
    print(f"  Fold         : {_fold}")
    print(f"  Planner      : {planner_name}")
    if plans_name:
        print(f"  Plans        : {plans_name}")
    print()

    # ── Build nnUNetV2Runner input config ──────────────────────────────
    # NOTE: nnUNetV2Runner.__init__ creates nnunet_raw/, nnunet_preprocessed/,
    # and nnunet_results/ as empty directories on instantiation. We must NOT
    # rely on parent-dir existence for skip logic — use Dataset001_* glob instead.
    input_config = {
        "modality":            {"0": "MRI"},      # single T1 channel
        "datalist":            str(effective_datalist),
        "dataroot":            str(_work_dir),
        "nnunet_raw":          str(_work_dir / "nnunet_raw"),
        "nnunet_preprocessed": str(_work_dir / "nnunet_preprocessed"),
        "nnunet_results":      str(_work_dir / "nnunet_results"),
        "dataset_name_or_id":  1,
    }

    raw_dir = _work_dir / "nnunet_raw"
    preprocessed_dir = _work_dir / "nnunet_preprocessed"
    results_dir = _work_dir / "nnunet_results"

    runner = nnUNetV2Runner(
        input_config=input_config,
        trainer_class_name=trainer_class,
        work_dir=str(_work_dir / "runner_work"),
    )

    # ── Convert dataset (with subject-count cache invalidation) ────────
    existing_count = _dataset_subject_count(raw_dir)

    if existing_count != n_train:
        if existing_count is not None:
            print(
                f"Step 1/3  Subject count changed "
                f"({existing_count} → {n_train}). Reconverting …"
            )
            for d in raw_dir.glob("Dataset001_*"):
                shutil.rmtree(d)
            for d in preprocessed_dir.glob("Dataset001_*"):
                shutil.rmtree(d)
        else:
            print("Step 1/3  Converting dataset to nnU-Net format …")
        runner.convert_dataset()
    else:
        print(
            f"Step 1/3  nnU-Net raw dir exists "
            f"with {existing_count} subjects — skipping conversion."
        )

    # ── Plan & preprocess ───────────────────────────────────────────────
    if not any(preprocessed_dir.glob("Dataset001_*")):
        print("Step 2/3  Planning and preprocessing …")
        runner.plan_and_process(pl=planner_name)
    else:
        print("Step 2/3  Preprocessed dir already exists — skipping.")

    # ── Train single model (fold N, 3d_fullres) ────────────────────────
    print(f"Step 3/3  Training {config_name}, fold {_fold} …")
    train_kwargs: dict = {}
    if plans_name:
        train_kwargs["p"] = plans_name
    runner.train_single_model(config=config_name, fold=_fold, **train_kwargs)
    print(f"\nTraining complete. Results saved to: {results_dir}")
    print("To train remaining folds (1–4), rerun with --fold 1, --fold 2, etc.")


if __name__ == "__main__":
    main()
