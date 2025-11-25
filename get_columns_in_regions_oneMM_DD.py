#!/usr/bin/env python

"""
get_columns_in_regions_oneMM_DD.py  (updated, no hardcoded QSM)

Changes:
- No 'QSM' hard-coded anywhere.
- Output directories now use <output_dir>/<ID>/<contrast>/...
- Per-column outputs:   <output_dir>/<ID>/<contrast>/<contrast>_cols_by_column/
- Region-mean outputs:  <output_dir>/<ID>/<contrast>/<contrast>_cols_region_mean/
- Label coords expected in: <output_dir>/<ID>/<contrast>/label_coord_1mm/
- Print shapes for debugging
- Save summary CSV
"""

import argparse
from pathlib import Path
import csv

import numpy as np
import nibabel as nib
from scipy.io import loadmat
from scipy.ndimage import map_coordinates


REGION_LIST = [
    "bankssts",
    "caudalanteriorcingulate",
    "caudalmiddlefrontal",
    "cuneus",
    "entorhinal",
    "fusiform",
    "inferiorparietal",
    "inferiortemporal",
    "isthmuscingulate",
    "lateraloccipital",
    "lateralorbitofrontal",
    "lingual",
    "medialorbitofrontal",
    "middletemporal",
    "parahippocampal",
    "paracentral",
    "parsopercularis",
    "parsorbitalis",
    "parstriangularis",
    "pericalcarine",
    "postcentral",
    "posteriorcingulate",
    "precentral",
    "precuneus",
    "rostralanteriorcingulate",
    "rostralmiddlefrontal",
    "superiorfrontal",
    "superiorparietal",
    "superiortemporal",
    "supramarginal",
    "frontalpole",
    "temporalpole",
    "transversetemporal",
    "insula",
]


# ---------------- IMAGE LOADING ---------------- #

def _load_contrast_image(input_dir: Path, ID: str, contrast: str) -> np.ndarray:
    image_path = input_dir / ID / f"{ID}_{contrast}_masked.nii.gz"
    print(f"[IMG] Looking for <{contrast}> image at: {image_path}")

    if not image_path.is_file():
        raise FileNotFoundError(f"Missing masked {contrast} image: {image_path}")

    img = nib.load(str(image_path))
    vol = img.get_fdata()

    if vol.ndim == 4:
        vol = vol[..., 0]

    print(f"[IMG] Loaded volume: shape {vol.shape}")
    return vol


# --------------- LOAD COORDS ------------------- #

def _load_region_cp_dwi(mat_path: Path, hemi: str) -> np.ndarray:
    if not mat_path.is_file():
        raise FileNotFoundError(f"Missing coordinate file: {mat_path}")

    data = loadmat(mat_path)
    var_name = f"{hemi}_cp_dwi"

    if var_name not in data:
        raise KeyError(f"{mat_path} missing variable {var_name}")

    cp = np.asarray(data[var_name])
    print(f"[CP] Loaded {var_name} from {mat_path.name}, shape={cp.shape}")
    return cp


# --------------- SAMPLING ------------------- #

def _sample_columns(vol: np.ndarray, cp_dwi: np.ndarray, points_num=21) -> np.ndarray:
    n_coords = cp_dwi.shape[1]

    if n_coords % points_num != 0:
        raise ValueError(f"Invalid cp_dwi size {n_coords} for points_num {points_num}")

    columns = n_coords // points_num
    print(f"[SAMPLE] columns={columns}, depths={points_num}")

    vals = np.zeros((columns, points_num), float)

    for d in range(points_num):
        idx = d + np.arange(columns) * points_num

        coords = cp_dwi[0:3, idx].astype(float)
        coords -= 1.0  # MATLAB → Python indexing

        sampled = map_coordinates(
            vol, coords, order=1, mode="nearest"
        )
        vals[:, d] = sampled

    return vals


# ---------------- MAIN ---------------- #

def get_columns_in_regions_oneMM_DD(ID, input_dir, output_dir, contrast):

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    print(f"\n[INFO] Running for subject {ID}")
    print(f"[INFO] Contrast: {contrast}")
    print(f"[INFO] I/O roots:")
    print(f"    input_dir:  {input_dir}")
    print(f"    output_dir: {output_dir}")

    # load MRI contrast
    vol = _load_contrast_image(input_dir, ID, contrast)

    # expected label coord folder from script 1
    label_coord_dir = output_dir / ID / contrast / "label_coord_1mm"
    if not label_coord_dir.is_dir():
        raise FileNotFoundError(f"Missing label_coord_1mm in: {label_coord_dir}")

    # new output layout
    contrast_dir = output_dir / ID / contrast

    per_column_dir = contrast_dir / f"{contrast}_cols_by_column"
    per_mean_dir = contrast_dir / f"{contrast}_cols_region_mean"

    per_column_dir.mkdir(parents=True, exist_ok=True)
    per_mean_dir.mkdir(parents=True, exist_ok=True)

    points_num = 21
    summary = []

    # loop over regions
    for region in REGION_LIST:
        print(f"\n[REGION] {region}")

        lh_mat = label_coord_dir / f"lh_{region}.mat"
        rh_mat = label_coord_dir / f"rh_{region}.mat"

        if not lh_mat.is_file() or not rh_mat.is_file():
            print(f"  -> missing lh/rh coord files, skipping region")
            continue

        # coord matrices
        lh_cp = _load_region_cp_dwi(lh_mat, "lh")
        rh_cp = _load_region_cp_dwi(rh_mat, "rh")

        # sample
        lh_vals = _sample_columns(vol, lh_cp, points_num)
        rh_vals = _sample_columns(vol, rh_cp, points_num)

        # --------- save per-column output ---------
        lh_path = per_column_dir / f"{ID}_lh_{region}_cols_{contrast}.csv"
        rh_path = per_column_dir / f"{ID}_rh_{region}_cols_{contrast}.csv"

        np.savetxt(lh_path, lh_vals, delimiter=",")
        np.savetxt(rh_path, rh_vals, delimiter=",")

        print(f"  -> LH per-column saved ({lh_vals.shape}) → {lh_path}")
        print(f"  -> RH per-column saved ({rh_vals.shape}) → {rh_path}")

        summary.append(["lh", region, lh_vals.shape[0], lh_vals.shape[1], lh_path.name])
        summary.append(["rh", region, rh_vals.shape[0], rh_vals.shape[1], rh_path.name])

        # --------- region mean depth profile ---------
        lh_mean = lh_vals.mean(axis=0).reshape(-1, 1)
        rh_mean = rh_vals.mean(axis=0).reshape(-1, 1)

        lh_mean_path = per_mean_dir / f"{ID}_lh_{region}_cols_{contrast}_mean.csv"
        rh_mean_path = per_mean_dir / f"{ID}_rh_{region}_cols_{contrast}_mean.csv"

        np.savetxt(lh_mean_path, lh_mean, delimiter=",")
        np.savetxt(rh_mean_path, rh_mean, delimiter=",")

        print(f"  -> LH depth-mean saved (21×1) → {lh_mean_path}")
        print(f"  -> RH depth-mean saved (21×1) → {rh_mean_path}")

    # --------- summary CSV ---------
    if summary:
        summary_path = per_column_dir / f"{ID}_cols_{contrast}_summary.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["hemi", "region", "n_columns", "n_depths", "csv_file"])
            writer.writerows(summary)
        print(f"\n[SUMMARY] wrote: {summary_path}")
    else:
        print("\n[SUMMARY] no regions processed.")


# ---------------- CLI ---------------- #

def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--ID", required=True)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--contrast", required=True)
    args = p.parse_args()

    get_columns_in_regions_oneMM_DD(
        args.ID, args.input_dir, args.output_dir, args.contrast
    )


if __name__ == "__main__":
    _cli()
