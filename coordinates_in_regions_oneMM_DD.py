#!/usr/bin/env python

"""
coordinates_in_regions_oneMM_DD.py

Verbose QA version of region-wise column coordinate extraction.

This script takes whole-hemisphere column coordinate matrices
(<ID>_column_lh_dwi.mat, <ID>_column_rh_dwi.mat) and parcels them
into region-wise coordinate files based on aparc annotations.

It is COMPLETELY contrast-agnostic.

Directory layout
----------------

Expected inputs (per subject):

    <output_dir>/<ID>/columns/
        <ID>_column_lh_dwi.mat   (var: lh_cp_dwi, shape (4, N_lh))
        <ID>_column_rh_dwi.mat   (var: rh_cp_dwi, shape (4, N_rh))

    <output_dir>/<ID>/<ID>/label/
        lh.aparc.annot
        rh.aparc.annot

Outputs (per subject, shared across all contrasts):

    <output_dir>/<ID>/columns/label_coord_1mm/
        lh_<region>.mat (var: lh_cp_dwi for that region)
        lh_<region>.csv
        rh_<region>.mat (var: rh_cp_dwi for that region)
        rh_<region>.csv

These region-wise coordinate files are then used by
get_columns_in_regions_oneMM_DD.py for ANY contrast.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat
import nibabel.freesurfer as fsio


DEPTH_SAMPLES = 21


# ------------------ LOW-LEVEL LOADERS ------------------ #

def _load_cp_dwi_mats(columns_dir: Path, ID: str):
    """Load *_column_*_dwi.mat coordinate matrices and print shapes."""
    lh_dwi_mat = columns_dir / f"{ID}_column_lh_dwi.mat"
    rh_dwi_mat = columns_dir / f"{ID}_column_rh_dwi.mat"

    print("\n[LOAD] Columns dir:", columns_dir)
    print("[LOAD] Expecting:")
    print(f"  {lh_dwi_mat}")
    print(f"  {rh_dwi_mat}")

    if not lh_dwi_mat.is_file():
        raise FileNotFoundError(f"Missing required: {lh_dwi_mat}")
    if not rh_dwi_mat.is_file():
        raise FileNotFoundError(f"Missing required: {rh_dwi_mat}")

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
    """
    Load FreeSurfer aparc annotation for one hemisphere and print basic stats.

    IMPORTANT: In nibabel, 'labels' are *indices* into ctab/names,
    not the original RGB-coded IDs (ctab[:,4]).
    """
    annot_fname = subject_dir / "label" / f"{hemi}.aparc.annot"
    print(f"\n[ANNOT] Loading {hemi}.aparc.annot from:")
    print("        ", annot_fname)

    if not annot_fname.is_file():
        raise FileNotFoundError(f"Missing annotation: {annot_fname}")

    labels, ctab, names = fsio.read_annot(str(annot_fname))

    struct_names = [
        n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else str(n)
        for n in names
    ]

    unique_labels = np.unique(labels)
    print(f"[ANNOT] {hemi}: labels shape = {labels.shape}, unique labels = {len(unique_labels)}")
    print(f"[ANNOT] {hemi}: struct_names = {len(struct_names)}, ctab rows = {ctab.shape[0]}")
    print(f"[ANNOT] {hemi}: first 10 unique label indices: {unique_labels[:10]}")

    return labels, ctab, struct_names


# ------------------ INDEXING UTIL ------------------ #

def _region_indices_for_vertices(vertex_indices_0b: np.ndarray, depth_samples: int = DEPTH_SAMPLES):
    """
    From 0-based vertex indices, compute 0-based depth-sample indices into
    the cp_dwi array.

    Each vertex has 'depth_samples' points; we assume the cp_dwi columns are
    laid out as:

        [v0_d0, v0_d1, ..., v0_d20,
         v1_d0, v1_d1, ..., v1_d20,
         v2_d0, ... ]

    For a vertex index v, its depth columns are:

        v * depth_samples + [0, 1, ..., depth_samples-1]
    """
    vertex_indices_0b = np.asarray(vertex_indices_0b, dtype=int)

    if vertex_indices_0b.size == 0:
        return np.zeros((0,), dtype=int)

    depth_idx = np.arange(depth_samples, dtype=int)  # [0..20]
    region_cols = vertex_indices_0b[:, None] * depth_samples + depth_idx[None, :]
    region_cols_flat = region_cols.ravel(order="C")

    return region_cols_flat


# ------------------ PER-REGION PROCESSING ------------------ #

def _process_hemi(
    hemi: str,
    labels: np.ndarray,
    ctab: np.ndarray,
    struct_names,
    ori_cp_dwi: np.ndarray,
    out_dir: Path,
):
    print(f"\n===== PROCESSING {hemi.upper()} HEMISPHERE =====")
    max_index = ori_cp_dwi.shape[1]

    n_regions_with_vertices = 0
    n_written = 0

    # MATLAB loop: for i = 2:36, skip i == 5
    max_i = min(36, len(struct_names), ctab.shape[0])

    for i in range(2, max_i + 1):  # inclusive
        if i == 5:
            print(f"[{hemi}] SKIP MATLAB i=5")
            continue

        row_idx = i - 1  # 0-based row in struct_names/ctab

        if row_idx >= len(struct_names) or row_idx >= ctab.shape[0]:
            print(f"[{hemi}] i={i}: row_idx={row_idx} out of bounds for struct_names/ctab, skipping.")
            continue

        region_name = struct_names[row_idx]
        region_id_orig = int(ctab[row_idx, 4])   # original FS ID (informational)
        region_index = row_idx                   # nibabel labels store this index

        # nibabel: labels contain the row index into names/ctab
        vertex_indices = np.where(labels == region_index)[0]
        n_vertices = vertex_indices.size

        print(f"\n[{hemi}] REGION i={i}, row_idx={row_idx}")
        print(f"    name          = '{region_name}'")
        print(f"    region_index  = {region_index} (matches 'labels')")
        print(f"    region_id_orig= {region_id_orig} (ctab[:,4])")
        print(f"    n_vertices    = {n_vertices}")

        if n_vertices == 0:
            print(f"    -> No vertices with label index {region_index}, skipping region.")
            continue

        n_regions_with_vertices += 1

        # Compute cp_dwi column indices for this region
        region_cols = _region_indices_for_vertices(vertex_indices, depth_samples=DEPTH_SAMPLES)
        region_cols = region_cols[region_cols < max_index]

        if region_cols.size == 0:
            print("    -> After bounds check, no columns remain; skipping.")
            continue

        cp_dwi = ori_cp_dwi[:, region_cols]

        # Save MAT and CSV
        mat_path = out_dir / f"{hemi}_{region_name}.mat"
        csv_path = out_dir / f"{hemi}_{region_name}.csv"

        savemat(mat_path, {f"{hemi}_cp_dwi": cp_dwi})
        np.savetxt(csv_path, cp_dwi, delimiter=",")

        print(f"    -> WROTE {mat_path.name}, {csv_path.name}")
        n_written += 1

    print(f"\n>>> SUMMARY {hemi.upper()} <<<")
    print(f"    Regions with vertices: {n_regions_with_vertices}")
    print(f"    Files written:         {n_written}")
    print(f"    Output dir:            {out_dir}")


# ------------------ TOP-LEVEL FUNCTION ------------------ #

def coordinates_in_regions_oneMM_DD(ID: str, output_dir):
    """
    Main driver for one subject.

    This is contrast-blind. It only:

        - Reads DWI-space column coords from:
              <output_dir>/<ID>/columns/<ID>_column_lh_dwi.mat
              <output_dir>/<ID>/columns/<ID>_column_rh_dwi.mat

        - Reads labels from:
              <output_dir>/<ID>/<ID>/label/lh.aparc.annot
              <output_dir>/<ID>/<ID>/label/rh.aparc.annot

        - Writes region coords to:
              <output_dir>/<ID>/columns/label_coord_1mm/
    """
    output_dir = Path(output_dir)

    columns_dir = output_dir / ID / "columns"
    subject_dir = output_dir / ID / ID
    out_dir = columns_dir / "label_coord_1mm"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[PATH] -------")
    print("[PATH] Subject ID:         ", ID)
    print("[PATH] Output root:        ", output_dir)
    print("[PATH] Columns dir:        ", columns_dir)
    print("[PATH] Subject dir (labels)", subject_dir)
    print("[PATH] Output dir:         ", out_dir)

    # Column coordinate matrices (DWI space)
    ori_lh_cp_dwi, ori_rh_cp_dwi = _load_cp_dwi_mats(columns_dir, ID)

    # Annotation files
    labels_lh, ctab_lh, names_lh = _load_aparc_annot(subject_dir, "lh")
    labels_rh, ctab_rh, names_rh = _load_aparc_annot(subject_dir, "rh")

    # Process hemispheres
    _process_hemi("lh", labels_lh, ctab_lh, names_lh, ori_lh_cp_dwi, out_dir)
    _process_hemi("rh", labels_rh, ctab_rh, names_rh, ori_rh_cp_dwi, out_dir)


# ------------------ CLI ENTRY ------------------ #

def _cli():
    parser = argparse.ArgumentParser(
        description="Verbose QA cortical column region extraction (contrast-agnostic)."
    )
    parser.add_argument("--ID", required=True, help="Subject ID (e.g. S00775)")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Root dir containing <ID>/columns and <ID>/<ID>/label.",
    )
    args = parser.parse_args()

    coordinates_in_regions_oneMM_DD(
        ID=args.ID,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    _cli()
