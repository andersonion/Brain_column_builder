#!/usr/bin/env python

"""
Python translation of MATLAB function get_thickness.m

Original MATLAB:

    function get_thickness(ID,input_dir,output_dir)

    - Checks for pair files:
        <output_dir>/<ID>/columns_1mm/<ID>_pair_lh.mat
        <output_dir>/<ID>/columns_1mm/<ID>_pair_rh.mat

    - Loads lh_pair, rh_pair
    - Computes lh_thickness = get_distance(lh_pair)
      and    rh_thickness = get_distance(rh_pair)
      (thickness per vertex / column)
    - Loads FreeSurfer aparc annotations (lh.aparc.annot, rh.aparc.annot)
    - For each region, extracts vertex indices and saves per-region thickness.

This Python version:

CLI usage example:

    python get_thickness.py \
        --ID S00775 \
        --output-dir /path/to/output_root

Expected layout:

    Pair files:
        <output_root>/<ID>/columns_1mm/<ID>_pair_lh.mat
        <output_root>/<ID>/columns_1mm/<ID>_pair_rh.mat

    FreeSurfer annot files:
        <output_root>/<ID>/<ID>/label/lh.aparc.annot
        <output_root>/<ID>/<ID>/label/rh.aparc.annot

Outputs:

    Per-region thickness (lh & rh):

        <output_root>/<ID>/thickness/
            <ID>_lh_<region>_thickness.mat   (var: thickness_lh_region)
            <ID>_lh_<region>_thickness.csv   (N_vertices_in_region x 1)

            <ID>_rh_<region>_thickness.mat   (var: thickness_rh_region)
            <ID>_rh_<region>_thickness.csv

Assumptions about lh_pair / rh_pair:

    - Stored in <ID>_pair_lh.mat as variable "lh_pair"
    - Stored in <ID>_pair_rh.mat as variable "rh_pair"
    - Shape is (N, 6) where columns are:
        [x1, y1, z1, x2, y2, z2]
      i.e. two 3D points per row (pial vs wm or vice versa)
    - Thickness = Euclidean distance between these two points.

If your pair arrays are shaped differently, the get_distance() helper will need
a small tweak, but it will print the actual shape to help you debug.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat
import nibabel as nib


# ------------------ distance helper ------------------ #

def get_distance_from_pair(pair_array: np.ndarray) -> np.ndarray:
    """
    Compute thickness from a "pair" array.

    Assumed default shape: (N, 6), where each row is:
        [x1, y1, z1, x2, y2, z2]

    Returns:
        thickness: shape (N,)  (Euclidean distance per row)
    """
    arr = np.asarray(pair_array)
    print(f"[get_distance] pair_array shape: {arr.shape}")

    if arr.ndim != 2:
        raise ValueError(
            f"Unsupported lh_pair/rh_pair array shape {arr.shape}. "
            f"Expected 2D (N, 6). Adjust get_distance_from_pair() as needed."
        )

    if arr.shape[1] == 6:
        p1 = arr[:, 0:3]
        p2 = arr[:, 3:6]
    elif arr.shape[0] == 6:
        # Maybe transposed (6, N); treat columns as rows
        arr = arr.T
        p1 = arr[:, 0:3]
        p2 = arr[:, 3:6]
    else:
        raise ValueError(
            f"Expected shape (N, 6) or (6, N), got {arr.shape}. "
            f"Please adapt get_distance_from_pair() to your data layout."
        )

    diff = p1 - p2
    thickness = np.linalg.norm(diff, axis=1)  # shape (N,)
    return thickness


# ------------------ annotation loader ------------------ #

def load_aparc_annotation(annot_path: Path):
    """
    Load a FreeSurfer aparc .annot file using nibabel.

    Returns:
        labels: (N_vertices,) int array
        names:  list of region names (decoded strings)
    """
    if not annot_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {annot_path}")

    labels, ctab, names = nib.freesurfer.io.read_annot(str(annot_path))
    # decode bytes → str
    names = [n.decode("utf-8") if isinstance(n, bytes) else n for n in names]

    print(f"[ANNOT] Loaded {annot_path.name}: "
          f"{len(labels)} vertices, {len(names)} regions (including 'unknown')")

    return labels, ctab, names


# ------------------ main logic ------------------ #

def get_thickness(ID: str, output_dir):
    """
    Python implementation of get_thickness.m

    Parameters
    ----------
    ID : str
        Subject ID, e.g. 'S00775'
    output_dir : str or Path
        Root output directory, containing:
            <output_dir>/<ID>/columns_1mm/<ID>_pair_lh.mat
            <output_dir>/<ID>/columns_1mm/<ID>_pair_rh.mat
            <output_dir>/<ID>/<ID>/label/lh.aparc.annot
            <output_dir>/<ID>/<ID>/label/rh.aparc.annot
    """
    output_dir = Path(output_dir)

    # ---------------- check / load pair files ---------------- #

    pair_dir = output_dir / ID / "columns_1mm"
    lh_pair_mat = pair_dir / f"{ID}_pair_lh.mat"
    rh_pair_mat = pair_dir / f"{ID}_pair_rh.mat"

    if not lh_pair_mat.is_file() or not rh_pair_mat.is_file():
        print(f"Subject {ID} doesnt have pair files in {pair_dir}")
        return

    print(f"[PAIR] Loading LH pair from: {lh_pair_mat}")
    lh_data = loadmat(lh_pair_mat)
    if "lh_pair" not in lh_data:
        raise KeyError(f"{lh_pair_mat} does not contain variable 'lh_pair'")
    lh_pair = lh_data["lh_pair"]

    print(f"[PAIR] Loading RH pair from: {rh_pair_mat}")
    rh_data = loadmat(rh_pair_mat)
    if "rh_pair" not in rh_data:
        raise KeyError(f"{rh_pair_mat} does not contain variable 'rh_pair'")
    rh_pair = rh_data["rh_pair"]

    # ---------------- compute thickness ---------------- #

    lh_thickness = get_distance_from_pair(lh_pair)  # shape (N_lh,)
    rh_thickness = get_distance_from_pair(rh_pair)  # shape (N_rh,)

    # ensure 2D column vectors for easier indexing/saving
    lh_thickness = lh_thickness.reshape(-1, 1)
    rh_thickness = rh_thickness.reshape(-1, 1)

    print(f"[THICK] LH thickness shape: {lh_thickness.shape}")
    print(f"[THICK] RH thickness shape: {rh_thickness.shape}")

    # ---------------- load annotations ---------------- #

    subj_fs_dir = output_dir / ID / ID / "label"

    lh_annot_path = subj_fs_dir / "lh.aparc.annot"
    rh_annot_path = subj_fs_dir / "rh.aparc.annot"

    labels_lh, ctab_lh, names_lh = load_aparc_annotation(lh_annot_path)
    labels_rh, ctab_rh, names_rh = load_aparc_annotation(rh_annot_path)

    if len(names_lh) != len(names_rh):
        print("[WARN] LH and RH aparc have different number of regions.")

    # ---------------- output directory ---------------- #

    thickness_dir = output_dir / ID / "thickness"
    thickness_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- region-wise loop ---------------- #
    #
    # MATLAB for i = 2:36, skip i == 5
    # 'i' is 1-based region index in struct_names/table.
    # In nibabel, region indices are 0-based into 'names' and 'labels'.
    #
    # MATLAB i = 2   → Python region_idx = 1
    # MATLAB i = 36  → Python region_idx = 35
    # MATLAB skip i=5 → skip region_idx=4.

    n_regions = min(len(names_lh), 36)  # safety
    for i in range(2, 37):  # MATLAB 2:36 inclusive
        region_idx = i - 1  # convert to 0-based

        if region_idx >= len(names_lh):
            continue

        if i == 5:
            # same skip as MATLAB
            print(f"[SKIP] MATLAB skipped i=5, skipping region_idx={region_idx}")
            continue

        # ---- LH hemisphere ---- #
        region_name_lh = names_lh[region_idx]
        # In MATLAB: vertx_num = find(label_lh == region_num)
        # For nibabel: labels_lh entries are region indices into 'names'
        vert_idx_lh = np.where(labels_lh == region_idx)[0]  # 0-based indices
        if vert_idx_lh.size == 0:
            print(f"[LH] Region '{region_name_lh}' has 0 vertices, skipping.")
        else:
            max_index_lh = lh_thickness.shape[0]
            vert_idx_lh = vert_idx_lh[vert_idx_lh < max_index_lh]

            thickness_lh_region = lh_thickness[vert_idx_lh, :]  # shape (#verts, 1)
            out_name_lh = f"{ID}_lh_{region_name_lh}_thickness"
            mat_path_lh = thickness_dir / f"{out_name_lh}.mat"
            csv_path_lh = thickness_dir / f"{out_name_lh}.csv"

            savemat(str(mat_path_lh), {"thickness_lh_region": thickness_lh_region})
            np.savetxt(csv_path_lh, thickness_lh_region, delimiter=",")

            print(f"[LH] {region_name_lh}: {thickness_lh_region.shape[0]} verts → "
                  f"{mat_path_lh.name}, {csv_path_lh.name}")

        # ---- RH hemisphere ---- #
        region_name_rh = names_rh[region_idx]
        vert_idx_rh = np.where(labels_rh == region_idx)[0]
        if vert_idx_rh.size == 0:
            print(f"[RH] Region '{region_name_rh}' has 0 vertices, skipping.")
        else:
            max_index_rh = rh_thickness.shape[0]
            vert_idx_rh = vert_idx_rh[vert_idx_rh < max_index_rh]

            thickness_rh_region = rh_thickness[vert_idx_rh, :]  # shape (#verts, 1)
            out_name_rh = f"{ID}_rh_{region_name_rh}_thickness"
            mat_path_rh = thickness_dir / f"{out_name_rh}.mat"
            csv_path_rh = thickness_dir / f"{out_name_rh}.csv"

            savemat(str(mat_path_rh), {"thickness_rh_region": thickness_rh_region})
            np.savetxt(csv_path_rh, thickness_rh_region, delimiter=",")

            print(f"[RH] {region_name_rh}: {thickness_rh_region.shape[0]} verts → "
                  f"{mat_path_rh.name}, {csv_path_rh.name}")

    print("\n[DONE] Thickness extraction complete.")


# ------------------ CLI ------------------ #

def _cli():
    parser = argparse.ArgumentParser(
        description="Compute cortical column thickness from pair files and parcel into regions."
    )
    parser.add_argument("--ID", required=True, help="Subject ID, e.g. S00775")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Root output dir containing <ID>/columns_1mm and <ID>/<ID>/label",
    )
    args = parser.parse_args()

    get_thickness(args.ID, args.output_dir)


if __name__ == "__main__":
    _cli()
