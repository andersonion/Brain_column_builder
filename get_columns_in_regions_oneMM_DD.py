#!/usr/bin/env python

"""
get_columns_in_regions_oneMM_DD.py

- No hardcoded 'QSM' anywhere.
- Supports arbitrary contrasts, via --contrast.
- Expects:
    <input_dir>/<ID>/<ID>_<contrast>_masked.nii.gz
    <output_dir>/<ID>/<contrast>/label_coord_1mm/*.mat

- Produces:
    Per-column values:
        <output_dir>/<ID>/<contrast>/<contrast>_cols_by_column/
            <ID>_lh_<region>_cols_<contrast>.csv
            <ID>_rh_<region>_cols_<contrast>.csv
            <ID>_cols_<contrast>_summary.csv

    Region-mean depth profiles (21×1):
        <output_dir>/<ID>/<contrast>/<contrast>_cols_region_mean/
            <ID>_lh_<region>_cols_<contrast}_mean.csv
            <ID>_rh_<region>_cols_<contrast}_mean.csv

    QA plots and line-profile CI CSV:
        <output_dir>/<ID>/<contrast>/plots_QA/
            <ID>_lh_<region>_profile_<contrast>.png
            <ID>_rh_<region>_profile_<contrast>.png
            <ID>_profiles_QA_<contrast>.csv
"""

import argparse
from pathlib import Path
import csv

import numpy as np
import nibabel as nib
from scipy.io import loadmat
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt


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
    """
    Sample volume at coordinates in cp_dwi.

    cp_dwi: shape (4, N_coords), first three rows are x,y,z (1-based).
    Returns: values shape (N_columns, points_num).
    """
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


# --------------- QA HELPERS ------------------- #

def _compute_mean_and_ci(values: np.ndarray, ci_level: float = 0.95):
    """
    Given values shape (n_columns, n_depths),
    return mean, ci_lower, ci_upper each shape (n_depths,).

    CI is computed as mean ± z * SEM with z from normal approximation.
    """
    n_cols = values.shape[0]
    if n_cols < 2:
        # no variance estimate; just return mean and identical bounds
        mean = values.mean(axis=0)
        return mean, mean.copy(), mean.copy(), n_cols

    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=1)
    sem = std / np.sqrt(n_cols)

    # For 95% CI, z ≈ 1.96. Could generalize but 0.95 is the common case.
    if ci_level == 0.95:
        z = 1.96
    else:
        from scipy.stats import norm
        z = norm.ppf(0.5 + ci_level / 2.0)

    ci_low = mean - z * sem
    ci_high = mean + z * sem
    return mean, ci_low, ci_high, n_cols


def _plot_profile_with_ci(
    depth_indices: np.ndarray,
    mean: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    out_path: Path,
    title: str,
    contrast: str,
):
    """
    Save a PNG of the depth profile with shaded CI.
    """
    fig, ax = plt.subplots()
    ax.plot(depth_indices, mean)
    ax.fill_between(depth_indices, ci_low, ci_high, alpha=0.3)

    ax.set_xlabel("Depth index")
    ax.set_ylabel(f"{contrast} intensity")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


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

    contrast_dir = output_dir / ID / contrast

    per_column_dir = contrast_dir / f"{contrast}_cols_by_column"
    per_mean_dir = contrast_dir / f"{contrast}_cols_region_mean"
    qa_dir = contrast_dir / "plots_QA"

    per_column_dir.mkdir(parents=True, exist_ok=True)
    per_mean_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    points_num = 21
    depth_indices = np.arange(points_num)

    summary = []       # for per-column summary CSV
    qa_rows = []       # for QA profiles CSV (all regions)

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

        # --------- region mean depth profile (21×1 CSV) ---------
        lh_mean_profile = lh_vals.mean(axis=0).reshape(-1, 1)
        rh_mean_profile = rh_vals.mean(axis=0).reshape(-1, 1)

        lh_mean_path = per_mean_dir / f"{ID}_lh_{region}_cols_{contrast}_mean.csv"
        rh_mean_path = per_mean_dir / f"{ID}_rh_{region}_cols_{contrast}_mean.csv"

        np.savetxt(lh_mean_path, lh_mean_profile, delimiter=",")
        np.savetxt(rh_mean_path, rh_mean_profile, delimiter=",")

        print(f"  -> LH depth-mean saved (21×1) → {lh_mean_path}")
        print(f"  -> RH depth-mean saved (21×1) → {rh_mean_path}")

        # --------- QA: mean + CI + plots ---------
        # LH
        lh_mean, lh_ci_low, lh_ci_high, lh_n = _compute_mean_and_ci(lh_vals, ci_level=0.95)
        lh_png_path = qa_dir / f"{ID}_lh_{region}_profile_{contrast}.png"
        _plot_profile_with_ci(
            depth_indices,
            lh_mean,
            lh_ci_low,
            lh_ci_high,
            lh_png_path,
            title=f"{ID} lh {region} ({contrast})",
            contrast=contrast,
        )
        print(f"  -> LH QA plot saved → {lh_png_path}")

        for d_idx, m, lo, hi in zip(depth_indices, lh_mean, lh_ci_low, lh_ci_high):
            qa_rows.append(["lh", region, int(d_idx), float(m), float(lo), float(hi), int(lh_n)])

        # RH
        rh_mean, rh_ci_low, rh_ci_high, rh_n = _compute_mean_and_ci(rh_vals, ci_level=0.95)
        rh_png_path = qa_dir / f"{ID}_rh_{region}_profile_{contrast}.png"
        _plot_profile_with_ci(
            depth_indices,
            rh_mean,
            rh_ci_low,
            rh_ci_high,
            rh_png_path,
            title=f"{ID} rh {region} ({contrast})",
            contrast=contrast,
        )
        print(f"  -> RH QA plot saved → {rh_png_path}")

        for d_idx, m, lo, hi in zip(depth_indices, rh_mean, rh_ci_low, rh_ci_high):
            qa_rows.append(["rh", region, int(d_idx), float(m), float(lo), float(hi), int(rh_n)])

    # --------- per-column summary CSV ---------
    if summary:
        summary_path = per_column_dir / f"{ID}_cols_{contrast}_summary.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["hemi", "region", "n_columns", "n_depths", "csv_file"])
            writer.writerows(summary)
        print(f"\n[SUMMARY] wrote per-column summary: {summary_path}")
    else:
        print("\n[SUMMARY] no regions processed for per-column summary.")

    # --------- QA profiles CSV (all regions) ---------
    if qa_rows:
        qa_csv_path = qa_dir / f"{ID}_profiles_QA_{contrast}.csv"
        with open(qa_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["hemi", "region", "depth_index", "mean", "ci_lower", "ci_upper", "n_columns"]
            )
            writer.writerows(qa_rows)
        print(f"[QA] wrote profile CI CSV: {qa_csv_path}")
    else:
        print("[QA] no QA rows generated; QA CSV not written.")


# ---------------- CLI ---------------- #

def _cli():
    p = argparse.ArgumentParser(
        description="Sample MRI contrast along cortical columns and generate QA outputs."
    )
    p.add_argument("--ID", required=True, help="Subject ID, e.g. S00775")
    p.add_argument("--input-dir", required=True, help="Root containing <ID>/<ID>_<contrast>_masked.nii.gz")
    p.add_argument("--output-dir", required=True, help="Root for <ID>/<contrast>/...")
    p.add_argument("--contrast", required=True, help="Contrast name, e.g. QSM, T1, FA, MD")
    args = p.parse_args()

    get_columns_in_regions_oneMM_DD(
        args.ID, args.input_dir, args.output_dir, args.contrast
    )


if __name__ == "__main__":
    _cli()
