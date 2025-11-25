#!/usr/bin/env python

"""
Verbose QA version of coordinates_in_regions_oneMM_DD.

Usage:
    python coordinates_in_regions_oneMM_DD.py \
        --ID S00775 \
        --input-dir  /mnt/newStor/paros/paros_WORK/hanwen/ad_decode_test/output/ \
        --output-dir /mnt/newStor/paros/paros_WORK/column_code_tester/ \
        [--contrast QSM] [--force]
"""

import argparse
from pathlib import Path
import numpy as np
from scipy.io import loadmat, savemat
import nibabel.freesurfer as fsio


# ------------------ LOW-LEVEL LOADERS ------------------ #

def _load_cp_dwi_mats(contrast_dir: Path, ID: str):
    """Load *_column_* DWI coordinate matrices and print shapes."""
    lh_mat = contrast_dir / f"{ID}_column_lh.mat"
    rh_mat = contrast_dir / f"{ID}_column_rh.mat"
    lh_dwi_mat = contrast_dir / f"{ID}_column_lh_dwi.mat"
    rh_dwi_mat = contrast_dir / f"{ID}_column_rh_dwi.mat"

    print("\n[LOAD] column dir:", contrast_dir)
    print("[LOAD] Expecting:")
    print(f"  {lh_mat}")
    print(f"  {rh_mat}")
    print(f"  {lh_dwi_mat}")
    print(f"  {rh_dwi_mat}")

    if not lh_dwi_mat.is_file():
        raise FileNotFoundError(f"Missing required: {lh_dwi_mat}")
    if not rh_dwi_mat.is_file():
        raise FileNotFoundError(f"Missing required: {rh_dwi_mat}")

    if lh_mat.is_file():
        _ = loadmat(lh_mat)
    if rh_mat.is_file():
        _ = loadmat(rh_mat)

    lh_dwi = loadmat(lh_dwi_mat)
    rh_dwi = loadmat(rh_dwi_mat)

    if "lh_cp_dwi" not in lh_dwi:
        raise KeyError(f"{lh_dwi_mat} missing 'lh_cp_dwi'")
    if "rh_cp_dwi" not in rh_dwi:
        raise KeyError(f"{rh_dwi_mat} missing 'rh_cp_dwi'")

    ori_lh_cp_dwi = np.asarray(lh_dwi["lh_cp_dwi"])
    ori_rh_cp_dwi = np.asarray(rh_dwi["rh_cp_dwi"])

    print(f"[LOAD] LH cp_dwi shape: {ori_lh_cp_dwi.shape}")
    print(f"[LOAD] RH cp_dwi shape: {ori_rh_cp_dwi.shape}")

    return ori_lh_cp_dwi, ori_rh_cp_dwi


def _load_aparc_annot(subject_dir: Path, hemi: str):
    """Load FreeSurfer annotation and print stats."""
    annot_fname = subject_dir / "label" / f"{hemi}.aparc.annot"
    print(f"\n[ANNOT] Loading {hemi}.aparc.annot from:\n        {annot_fname}")

    if not annot_fname.is_file():
        raise FileNotFoundError(f"Missing annotation: {annot_fname}")

    labels, ctab, names = fsio.read_annot(str(annot_fname))

    struct_names = [
        n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else str(n)
        for n in names
    ]

    print(f"[ANNOT] {hemi}: labels={labels.shape}, unique={len(np.unique(labels))}")
    print(f"[ANNOT] {hemi}: struct_names={len(struct_names)}, ctab rows={ctab.shape[0]}")
    print(f"[ANNOT] {hemi}: first names: {struct_names[:5]}")

    return labels, ctab, struct_names


def _region_indices_for_vertices(vertex_indices: np.ndarray) -> np.ndarray:
    """Generate 1-based depth indices."""
    if vertex_indices.size == 0:
        return np.array([], dtype=int)

    depths = np.arange(48, dtype=int)   # 0..47
    depth_indices = depths + 48         # 48..95

    repeated = np.tile(depth_indices[:, None], (1, vertex_indices.size))
    index_1b = repeated.flatten(order="F")

    return index_1b


# ------------------ CORE PER-HEMI PROCESSING ------------------ #

def _process_hemi(
    hemi: str,
    labels: np.ndarray,
    ctab: np.ndarray,
    struct_names: list[str],
    ori_cp_dwi: np.ndarray,
    out_dir: Path,
    force: bool = False,
):
    print(f"\n===== PROCESSING {hemi.upper()} HEMISPHERE =====")
    max_index = ori_cp_dwi.shape[1]

    skipped_regions = []
    n_written = 0
    n_regions_with_vertices = 0

    for i in range(2, 37):
        row_idx = i - 1
        region_name = struct_names[row_idx] if 0 <= row_idx < len(struct_names) else None

        # MATLAB skip
        if i == 5:
            msg = f"{hemi} i={i} region={region_name}: MATLAB skip (i=5)"
            print(f"    -> {msg}")
            skipped_regions.append(msg)
            continue

        if row_idx >= len(struct_names) or row_idx >= ctab.shape[0]:
            msg = f"{hemi} i={i}: row_idx={row_idx} out of bounds → skip"
            print(f"    -> {msg}")
            skipped_regions.append(msg)
            continue

        region_index = row_idx
        region_id_orig = int(ctab[row_idx, 4])

        vertex_indices = np.where(labels == region_index)[0]
        n_vertices = vertex_indices.size

        print(f"\n[{hemi}] REGION {region_name} (i={i})")
        print(f"    vertices = {n_vertices}")

        if n_vertices == 0:
            msg = f"{hemi} region '{region_name}': no vertices → skip"
            print(f"    -> {msg}")
            skipped_regions.append(msg)
            continue

        # -----------------------
        # FILES ALREADY EXIST? (skip unless --force)
        # -----------------------
        output_name = f"{hemi}_{region_name}"
        mat_path = out_dir / f"{output_name}.mat"
        csv_path = out_dir / f"{output_name}.csv"

        if not force and mat_path.is_file() and csv_path.is_file():
            msg = f"{hemi} region '{region_name}': output exists → skipping"
            print(f"    -> {msg}")
            skipped_regions.append(msg)
            continue

        if force:
            print(f"    -> FORCE recompute")

        # Compute depth indices
        index_1b = _region_indices_for_vertices(vertex_indices)
        index_1b = index_1b[index_1b <= max_index]

        if index_1b.size == 0:
            msg = f"{hemi} region '{region_name}': no valid depth idx → skip"
            print(f"    -> {msg}")
            skipped_regions.append(msg)
            continue

        idx_0b = index_1b - 1
        cp_dwi = ori_cp_dwi[:, idx_0b]

        savemat(mat_path, {f"{hemi}_cp_dwi": cp_dwi})
        np.savetxt(csv_path, cp_dwi, delimiter=",")

        print(f"    -> WROTE {mat_path.name}, {csv_path.name}")
        n_written += 1
        n_regions_with_vertices += 1

    print(f"\n>>> SUMMARY {hemi.upper()} <<<")
    print(f"    Files written: {n_written}")
    print(f"    Regions with vertices: {n_regions_with_vertices}")

    print("\n>>> SKIPPED REGIONS <<<")
    if skipped_regions:
        for msg in skipped_regions:
            print("    -", msg)
    else:
        print("    None.")


# ------------------ TOP-LEVEL ------------------ #

def coordinates_in_regions_oneMM_DD(ID: str, input_dir, output_dir, contrast: str, force: bool):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    contrast_dir = input_dir / ID / contrast
    subject_dir = input_dir / ID / ID
    out_dir = output_dir / ID / contrast / "label_coord_1mm"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[PATHS]")
    print("  ID:            ", ID)
    print("  contrast:      ", contrast)
    print("  contrast_dir:  ", contrast_dir)
    print("  subject_dir:   ", subject_dir)
    print("  out_dir:       ", out_dir)
    print("  FORCE mode:    ", force)

    ori_lh, ori_rh = _load_cp_dwi_mats(contrast_dir, ID)

    labels_lh, ctab_lh, names_lh = _load_aparc_annot(subject_dir, "lh")
    labels_rh, ctab_rh, names_rh = _load_aparc_annot(subject_dir, "rh")

    _process_hemi("lh", labels_lh, ctab_lh, names_lh, ori_lh, out_dir, force)
    _process_hemi("rh", labels_rh, ctab_rh, names_rh, ori_rh, out_dir, force)


# ------------------ CLI ------------------ #

def _cli():
    p = argparse.ArgumentParser()

    p.add_argument("--ID", required=True)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--contrast", default="QSM")
    p.add_argument(
        "--force",
        action="store_true",
        help="Recompute all regions even when outputs already exist."
    )

    args = p.parse_args()

    coordinates_in_regions_oneMM_DD(
        args.ID,
        args.input_dir,
        args.output_dir,
        contrast=args.contrast,
        force=args.force,
    )


if __name__ == "__main__":
    _cli()
