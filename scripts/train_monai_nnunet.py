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


def _validate_dataset_json(raw_dir: Path) -> bool:
    """Return True iff Dataset001_*/dataset.json has string-valued channel_names.

    A dict-valued channel_names (e.g. {"0": {"0": "MRI"}}) was written when
    input_config["modality"] was incorrectly set to a dict.  Returning False
    triggers automatic re-conversion with the correct modality string.
    """
    candidates = list(raw_dir.glob("Dataset001_*/dataset.json"))
    if not candidates:
        return False
    ds = json.loads(candidates[0].read_text())
    return all(isinstance(v, str) for v in ds.get("channel_names", {}).values())


def _validate_plans_file(preprocessed_dir: Path, plans_name: str) -> bool:
    """Return True iff the expected <plans_name>.json exists in nnunet_preprocessed.

    plan_and_process() has overwrite_plans_name="nnUNetPlans" as a hardcoded
    default, which silently overrides ResEncUNetPlanner's own identifier.
    Passing overwrite_plans_name=plans_name fixes this, but any run done before
    that fix will have the wrong file. Returning False triggers re-preprocessing.
    """
    return bool(list(preprocessed_dir.glob(f"Dataset001_*/{plans_name}.json")))


def _validate_datalist(datalist_path: Path, work_dir: Path) -> None:
    """Raise clearly if the first training image in the datalist is inaccessible.

    MONAI's convert_dataset() uses absolute paths from the datalist as-is.
    If the datalist was prepared with a different --work-dir, paths will be
    wrong and convert_dataset() will silently swallow the resulting error.
    """
    dl = json.loads(datalist_path.read_text())
    entries = dl.get("training", [])
    if not entries:
        raise ValueError(f"No training entries found in {datalist_path}")
    first_img = entries[0].get("image", "")
    if not Path(first_img).exists():
        raise FileNotFoundError(
            f"Datalist image not accessible: {first_img}\n"
            f"The datalist at {datalist_path} was likely prepared with a "
            f"different --work-dir than the current one ({work_dir}).\n"
            "Fix: re-run the preparation script with matching --work-dir:\n"
            f"  python scripts/prepare_nnunet_dataset.py "
            f"--data-root <HCP_ROOT> --work-dir {work_dir}"
        )


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
        "modality":            "MRI",             # single T1 channel
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

    # ── Validate datalist paths before touching the runner ─────────────
    # MONAI's convert_dataset() uses absolute paths from the datalist as-is
    # and swallows any exception. A path mismatch causes a silent failure.
    _validate_datalist(effective_datalist, _work_dir)

    runner = nnUNetV2Runner(
        input_config=input_config,
        trainer_class_name=trainer_class,
        work_dir=str(_work_dir / "runner_work"),
    )

    # ── Convert dataset (with subject-count / format cache invalidation) ─
    existing_count = _dataset_subject_count(raw_dir)
    dataset_json_ok = _validate_dataset_json(raw_dir)

    if existing_count != n_train or not dataset_json_ok:
        if existing_count is not None:
            if existing_count != n_train:
                reason = f"subject count changed ({existing_count} → {n_train})"
            else:
                reason = "dataset.json has wrong channel_names format"
            print(f"Step 1/3  {reason.capitalize()} — reconverting …")
            for d in raw_dir.glob("Dataset001_*"):
                shutil.rmtree(d)
            for d in preprocessed_dir.glob("Dataset001_*"):
                shutil.rmtree(d)
        else:
            print("Step 1/3  Converting dataset to nnU-Net format …")
        runner.convert_dataset()
        # convert_dataset() catches BaseException internally and only logs a
        # WARNING — check that dataset.json was actually written.
        if not list(raw_dir.glob("Dataset001_*/dataset.json")):
            raise RuntimeError(
                "convert_dataset() failed silently (see WARNING above).\n"
                "Most common cause: image paths in the datalist are inaccessible.\n"
                "Re-run the preparation script with matching --work-dir:\n"
                f"  python scripts/prepare_nnunet_dataset.py "
                f"--data-root <HCP_ROOT> --work-dir {_work_dir}"
            )
    else:
        print(
            f"Step 1/3  nnU-Net raw dir exists "
            f"with {existing_count} subjects — skipping conversion."
        )

    # ── Plan & preprocess ───────────────────────────────────────────────
    # plan_and_process() has overwrite_plans_name="nnUNetPlans" hardcoded by
    # default, which overrides ResEncUNetPlanner's own identifier.  We must
    # pass overwrite_plans_name=plans_name explicitly to get the right file.
    _plans_ok = not plans_name or _validate_plans_file(preprocessed_dir, plans_name)
    if not any(preprocessed_dir.glob("Dataset001_*")) or not _plans_ok:
        if not _plans_ok and any(preprocessed_dir.glob("Dataset001_*")):
            print("Step 2/3  Plans file missing or wrong name — re-preprocessing …")
            for d in preprocessed_dir.glob("Dataset001_*"):
                shutil.rmtree(d)
        else:
            print("Step 2/3  Planning and preprocessing …")
        pp_kwargs: dict = {"pl": planner_name}
        if plans_name:
            pp_kwargs["overwrite_plans_name"] = plans_name
        runner.plan_and_process(**pp_kwargs)
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
