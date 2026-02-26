# Sulcal Segmentation — deep_segmentator

Lifespan-robust cortical sulcal segmentation using nnU-Net v2 — self-configuring
3-D U-Net trained on HCP, target Dice ≥ 95 %.

**Input**: skull-stripped T1 MRI (0.7 mm isotropic, `*.nii.gz`)
**Output**: 52-class sulcal label map (Morphologist-compatible)

---

## Installation

Requires [pixi](https://pixi.sh) (recommended) or a manual conda/pip environment.

```bash
git clone <repo-url> deep_segmentator
cd deep_segmentator
pixi install
```

This installs Python 3.11, PyTorch (CUDA 12), MONAI ≥ 1.2, nnunetv2 ≥ 2.2, and the
`sulcal_seg` package in editable mode.

---

## Data

The pipeline is trained on the **Human Connectome Project (HCP)** dataset
(1 114 subjects, 0.7 mm T1, Morphologist Lgrey\_white / Rgrey\_white segmentations).

Expected layout on disk:

```
<data-root>/                          # e.g. /neurospin/dico/.../hcp/hcp/
├── 100206/
│   └── t1mri/BL/
│       ├── 100206.nii.gz                              # raw T1 (fallback)
│       └── default_analysis/segmentation/
│           ├── skull_stripped_100206.nii.gz            # input image
│           ├── Lgrey_white_100206.nii.gz               # L GM/WM label
│           └── Rgrey_white_100206.nii.gz               # R GM/WM label
├── 100307/ ...
└── ...
```

---

## Training

### Step 1 — Prepare dataset (run once, CPU only)

Reads from `<data-root>` (read-only), writes everything to `<work-dir>` (outside the git repo).

```bash
python scripts/prepare_nnunet_dataset.py \
    --data-root /neurospin/dico/data/bv_databases/human/not_labeled/hcp/hcp \
    --work-dir  ~/nnunet_work/hcp
```

Outputs under `~/nnunet_work/hcp/`:

| Path | Description |
|------|-------------|
| `images/{sid}_0000.nii.gz` | Symlink to skull-stripped T1 |
| `labels/{sid}.nii.gz` | Combined L+R grey/white label (0=bg, 1=L, 2=R) |
| `datalist.json` | MONAI datalist with 5-fold CV assignments |

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--data-root` | *(required)* | HCP per-subject root directory |
| `--work-dir` | `~/nnunet_work/hcp` | Output directory (must be outside the git repo) |
| `--n-folds` | `5` | Number of cross-validation folds |

### Step 2 — Train (GPU required)

```bash
python scripts/train_monai_nnunet.py \
    --config   configs/training.yaml \
    --work-dir ~/nnunet_work/hcp \
    --fold     0
```

This calls `nnUNetV2Runner` which:
1. Converts the dataset to nnU-Net raw format (idempotent)
2. Runs fingerprint extraction and preprocessing (idempotent)
3. Trains one model (`3d_fullres`, fold 0 by default)

Best checkpoint is saved to `~/nnunet_work/hcp/nnunet_results/`.

**Training options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `configs/training.yaml` | Training YAML configuration |
| `--work-dir` | *(from config)* | Override `data.work_dir` |
| `--fold` | *(from config)* | Override `training.fold` (0–4) |
| `--data-root` | *(from config)* | Override `data.root_dir` |

**Full 5-fold cross-validation:**

```bash
for fold in 0 1 2 3 4; do
    python scripts/train_monai_nnunet.py \
        --config configs/training.yaml \
        --work-dir ~/nnunet_work/hcp \
        --fold $fold
done
```

### Configuration reference (`configs/training.yaml`)

```yaml
data:
  root_dir: /path/to/hcp/hcp        # HCP per-subject root (read-only)
  dataset_name: hcp
  work_dir: ~/nnunet_work/hcp       # staging dir — must be OUTSIDE the git repo

training:
  nnunet_trainer: nnUNetTrainer_250epochs  # or nnUNetTrainer (1000 ep), nnUNetTrainer_100epochs
  nnunet_config:  3d_fullres               # 3-D full-resolution U-Net
  fold: 0                                  # CV fold (0–4)
  mixed_precision: true
  seed: 42

evaluation:
  checkpoint: ""                     # path to best checkpoint after training
  output_dir: outputs/evaluation
  gate_dice_threshold: 0.95          # pass criterion
  n_subjects: 500
  n_arg: 50
```

---

## Inference

After training, use the **nnU-Net v2 CLI** directly. Set the three environment variables
that tell nnU-Net where your data live:

```bash
export nnUNet_raw=~/nnunet_work/hcp/nnunet_raw
export nnUNet_preprocessed=~/nnunet_work/hcp/nnunet_preprocessed
export nnUNet_results=~/nnunet_work/hcp/nnunet_results
```

**Single-fold prediction:**

```bash
nnUNetv2_predict \
    -i /path/to/input_images/ \    # directory with *_0000.nii.gz files
    -o /path/to/predictions/ \
    -d 1 \                          # dataset ID (1 = HCP)
    -c 3d_fullres \
    -f 0                            # fold index (or 'all' for ensemble)
```

**Ensemble prediction (all 5 folds):**

```bash
nnUNetv2_predict \
    -i /path/to/input_images/ \
    -o /path/to/predictions/ \
    -d 1 \
    -c 3d_fullres \
    -f all
```

Input images must follow the nnU-Net naming convention:
`{case_identifier}_0000.nii.gz` (channel index `_0000` for single-modality T1).

---

## Evaluation (gate: Dice ≥ 95 %)

The quality criterion is **mean Dice ≥ 95 %** on the held-out test set.

A partial evaluation script is provided at [scripts/evaluate_nnunet.py](scripts/evaluate_nnunet.py).
It loads a checkpoint, runs inference, and computes 13 metrics including Dice, Hausdorff distance,
sensitivity/specificity, L/R sulcal symmetry, and GPU inference time.

```bash
python scripts/evaluate_nnunet.py \
    --checkpoint ~/nnunet_work/hcp/nnunet_results/<model_path>/checkpoint_best.pth \
    --config     configs/training.yaml \
    --output-dir outputs/evaluation/ \
    --n-subjects 500 \
    --n-arg      50
```

> **Note**: the DataLoader connection in `evaluate_nnunet.py` is currently a stub
> (`NotImplementedError`). Until completed, use `nnUNetv2_predict` followed by
> nnU-Net's built-in evaluation tool:
>
> ```bash
> nnUNetv2_evaluate_folder \
>     -ref /path/to/ground_truth/ \
>     -pred /path/to/predictions/ \
>     -d 1 -c 3d_fullres -f 0
> ```

---

## Pixi task shortcuts

```bash
pixi run prepare-nnunet   # prepare HCP dataset (edit task in pixi.toml for your paths)
pixi run train-nnunet     # train fold 0 with default config
pixi run test             # run all unit tests with coverage
pixi run test-unit        # unit tests only
pixi run lint             # ruff + black check
pixi run format           # auto-format with black + ruff --fix
```

---

## Project structure

```
deep_segmentator/
├── configs/
│   ├── training.yaml              # nnU-Net training config (data, training, evaluation)
│   └── default.yaml               # baseline defaults
├── scripts/
│   ├── prepare_nnunet_dataset.py  # Step 1: materialise labels + datalist.json
│   ├── train_monai_nnunet.py      # Step 2: train via nnUNetV2Runner
│   ├── evaluate_nnunet.py         # Step 3: evaluate + ARG file generation (partial)
│   ├── preprocess_all.py          # standalone preprocessing helper
│   ├── download_datasets.py       # dataset download utility
│   ├── monitor_training.py        # live training monitor
│   ├── generate_slurm_jobs.py     # SLURM job generator
│   └── setup.py                   # environment setup helper
├── src/sulcal_seg/
│   ├── config/                    # Pydantic v2 config classes (DataConfig, TrainingConfig)
│   ├── data/
│   │   └── loaders/               # HCPLoader, UKBiobankLoader, ABCDLoader, …
│   ├── models/
│   │   └── monai_nnunet.py        # MONAInnUNetModel
│   ├── inference/
│   │   └── morphologist_output.py # segmentation → Morphologist ARG format
│   ├── training/
│   │   └── monai_trainer.py       # MonaiTrainer (stub)
│   ├── validation/
│   │   └── metrics.py             # Dice, Hausdorff, symmetry, …
│   └── utils/                     # logging, visualisation, checkpoints
├── tests/
│   ├── unit/                      # pytest unit tests
│   └── fixtures/                  # synthetic NIfTI data generators
├── pixi.toml                      # environment + task definitions
└── pyproject.toml                 # package metadata (hatchling)
```

---

## Development

```bash
# Run tests
pixi run test

# Lint and format
pixi run lint
pixi run format

# Type check
pixi run python -m mypy src/
```

All intermediate files (checkpoints, preprocessed data, predictions) are written outside
the git repository by default. The `.gitignore` blocks `work_dir/`, `outputs/`, `logs/`,
`*.pt`, `*.pth`, `*.nii.gz`, and other large artefacts.
