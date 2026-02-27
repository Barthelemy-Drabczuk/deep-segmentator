"""Convert segmentation arrays to Morphologist-compatible output format."""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np


def _surface_area(verts: np.ndarray, faces: np.ndarray) -> float:
    """Return total surface area (mm²) of a triangulated mesh."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    return float(0.5 * np.linalg.norm(cross, axis=1).sum())


def segmentation_to_morphologist_format(
    segmentation: np.ndarray,
    output_dir: Union[str, Path],
    mri_path: Optional[Union[str, Path]] = None,
    subject_id: str = "unknown",
) -> Dict[str, Path]:
    """
    Convert a segmentation array to Morphologist-compatible output files.

    Writes three artefacts to ``output_dir``:

    * ``<subject_id>.arg``
        BrainVISA RoiArg graph (native soma.aims format) with one vertex
        per foreground label, each carrying an ``AimsTimeSurface`` mesh
        extracted via marching cubes.
    * ``<subject_id>_metadata.json``
        Algorithm parameters and per-label statistics (volume, centroid,
        surface area).
    * ``<subject_id>_segmentation.nii.gz``
        Argmax label volume in NIfTI format.

    Args:
        segmentation: Probability or logit array of shape
            (num_classes, D, H, W) or (D, H, W) for hard labels.
        output_dir: Directory where output files will be written.
        mri_path: Optional path to the source T1 MRI (used for the
            voxel-to-world affine and stored in metadata).
        subject_id: Identifier used as the filename stem.

    Returns:
        Dict mapping ``'arg'``, ``'metadata'``, ``'segmentation'`` to the
        corresponding output ``Path`` objects.
    """
    import nibabel as nib
    from skimage.measure import marching_cubes
    from soma import aims

    # ── Parse input → hard label volume ────────────────────────────────
    if segmentation.ndim == 4:      # (C, D, H, W) probabilities / logits
        label_vol = np.argmax(segmentation, axis=0).astype(np.int32)
    elif segmentation.ndim == 3:    # (D, H, W) hard labels
        label_vol = segmentation.astype(np.int32)
    else:
        raise ValueError(
            f"Expected 3-D or 4-D array, got shape {segmentation.shape}"
        )

    # ── Affine from reference MRI ───────────────────────────────────────
    if mri_path is not None:
        affine = nib.load(str(mri_path)).affine      # (4, 4)
    else:
        affine = np.eye(4)

    voxel_size = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))  # (3,) mm/vox

    # ── Write NIfTI segmentation ────────────────────────────────────────
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seg_path = output_dir / f"{subject_id}_segmentation.nii.gz"
    nib.save(
        nib.Nifti1Image(label_vol.astype(np.int16), affine), str(seg_path)
    )

    # ── Extract per-label meshes (marching cubes) ───────────────────────
    LABEL_NAMES = {1: "left_grey_white", 2: "right_grey_white"}

    meshes: dict = {}
    for label_id, label_name in LABEL_NAMES.items():
        mask = (label_vol == label_id).astype(np.float32)
        if mask.sum() == 0:
            continue
        verts, faces, _normals, _ = marching_cubes(mask, level=0.5)
        # voxel → world-space coordinates
        verts_world = (affine[:3, :3] @ verts.T + affine[:3, 3:]).T
        meshes[label_id] = (verts_world, faces, label_name)

    # ── Build BrainVISA RoiArg graph ────────────────────────────────────
    graph = aims.Graph("RoiArg")
    graph["datagraph_VERSION"] = "3.0"
    graph["type"] = "RoiArg"
    graph["subject"] = subject_id
    graph["mri_path"] = str(mri_path) if mri_path else ""
    graph["voxel_size"] = aims.Point3df(*voxel_size.tolist())
    graph["boundingbox_min"] = aims.Point3df(0., 0., 0.)
    graph["boundingbox_max"] = aims.Point3df(*reversed(label_vol.shape))

    for label_id, (verts_w, faces, label_name) in meshes.items():
        v = graph.addVertex("roi")
        v["label"] = str(label_id)
        v["name"] = label_name

        surf = aims.AimsTimeSurface(3)
        surf.vertex().assign([aims.Point3df(*p) for p in verts_w])
        surf.polygon().assign(
            [aims.AimsVector_U32_3(*f.tolist())
             for f in faces.astype(np.uint32)]
        )
        surf.updateNormals()
        aims.GraphManip.storeAims(graph, v, "roi", surf)

    arg_path = output_dir / f"{subject_id}.arg"
    aims.write(graph, str(arg_path))

    # ── Compute per-label statistics and write metadata JSON ────────────
    label_stats: dict = {}
    for label_id, (verts_w, faces, label_name) in meshes.items():
        mask = label_vol == label_id
        vox_vol = float(mask.sum()) * float(np.prod(voxel_size))
        cx, cy, cz = [float(np.mean(c)) for c in np.where(mask)]
        centroid = (
            affine[:3, :3] @ np.array([cx, cy, cz]) + affine[:3, 3]
        ).tolist()
        label_stats[label_name] = {
            "label_id": label_id,
            "volume_mm3": round(vox_vol, 3),
            "centroid_mm": [round(c, 3) for c in centroid],
            "surface_area_mm2": round(
                _surface_area(verts_w, faces.astype(int)), 3
            ),
        }

    meta_path = output_dir / f"{subject_id}_metadata.json"
    meta_path.write_text(json.dumps({
        "subject_id": subject_id,
        "algorithm": "nnU-Net Phase 1 (deep_segmentator)",
        "generation_date": datetime.now().isoformat(),
        "mri_path": str(mri_path) if mri_path else None,
        "labels": label_stats,
    }, indent=2))

    return {"arg": arg_path, "metadata": meta_path, "segmentation": seg_path}
