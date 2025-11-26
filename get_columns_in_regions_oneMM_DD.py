#!/usr/bin/env python

"""
get_columns_in_regions_oneMM_DD.py

Sample MRI contrast along cortical columns and generate QA outputs.

Key behavior
------------

- Contrast images are contrast-specific and loaded from `input_dir` using
  flexible patterns:

    1) <input_dir>/<ID>/<ID>_<contrast>_masked.nii.gz
    2) <input_dir>/<ID>_<contrast>_masked.nii.gz
    3) <input_dir>/<ID>/<ID>_<contrast>.nii.gz
    4) <input_dir>/<ID>_<contrast>.nii.gz

- Column coordinates are contrast-agnostic and shared across all contrasts:

    <output_dir>/<ID>/columns/label_coord_1mm/
        lh_<region>.mat (var: lh_cp_dwi)
        rh_<region>.mat (var: rh_cp_dwi)

- Per-contrast outputs go under:

    <output_dir>/<ID>/<contrast>/...

Checksum-based skipping is per-region, per-contrast, via:

    <output_dir>/<ID>/<contrast>/plots_QA/<ID>_<region>_cols_<contrast>.sha256
"""

import argparse
from pathlib import Path
import csv
import hashlib

import numpy as np
import nibabel as nib
from scipy.io import loadmat
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt


PIPELINE_VERSION = "get_columns_v1.0_20251125"


def _region_checksum(ID: str, contrast: str, region: str) -> str:
    """Return a stable checksum string for one region's config."""
    h = hashlib.sha256()
    h.update(PIPELINE_VERSION.encode("utf-8"))
    h.update(ID.encode("utf-8"))
    h.update(contrast.encode("utf-8"))
    h.update(region.encode("utf-8"))
    return h.hexdigest()


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
    """
    Load contrast image with flexible path resolution.

    Acceptable patterns (in priority order):

        1) <input_dir>/<ID>/<ID>_<contrast>_masked.nii.gz
        2) <input_dir>/<ID>_<contrast>_masked.nii.gz
        3) <input_dir>/<ID>/<ID>_<contrast>.nii.gz
        4) <input_dir>/<ID>_<contrast>.nii.gz
    """
    candidates = [
        input_dir / ID / f"{ID}_{contrast}_masked.nii.gz",
        input_dir / f"{ID}_{contrast}_masked.nii.gz",
        input_dir / ID / f"{ID}_{contrast}.nii.gz",
        input_dir / f"{ID}_{contrast}.nii.gz",
    ]

    found = None
    for p in candidates:
        if p.is_file():
            found = p
            break

    if found is None:
        raise FileNotFoundError(
            f"Could not locate contrast image for ID={ID}, contrast={contrast}\n"
            f"Tried:\n" + "\n".join([f"  {str(c)}" for c in candidates])
        )

    print(f"[IMG] Using contrast image: {found}")

    img = nib.load(str(found))
    vol = img.get_fdata()

    if vol.ndim == 4:
        vol = vol[..., 0]

    print(f"[IMG] Loaded volume: shape {vol.shape}")
    return vol


# ---------------- COORDS LOADING ---------------- #

def _load_region_cp_dwi(mat_path: Path, hemi: str) -> np.ndarray:
    """
    Load cp_dwi for a single region/hemisphere.

    mat_path: lh_<region>.mat or rh_<region>.mat
    hemi: "lh" or "rh"
    """
    data = loadmat(mat_path)
    key_candidates = [
        f"{hemi}_cp_dwi",
        "cp_dwi",
    ]

    for key in key_candidates:
        if key in data:
            cp = np.asarray(data[key])
            print(f"[MAT] Loaded {key} from {mat_path.name} with shape {cp.shape}")
            return cp

    raise KeyError(
        f"None of {key_candidates} found in {mat_path.name}. Keys: {list(data.keys())}"
    )


# --------------- SAMPLING ALONG COLUMNS ----------------- #

def _sample_columns(
    vol: np.ndarray,
    cp_dwi: np.ndarray,
    points_num: int,
) -> np.ndarray:
    """
    Sample volume at coordinates in cp_dwi.

    cp_dwi: shape (4, N_coords) or (3, N_coords); first three rows are x,y,z (1-based).
    Returns: values shape (N_columns, points_num).
    """
    if cp_dwi.shape[0] == 4:
        coords = cp_dwi[0:3, :]  # drop homogeneous row
    elif cp_dwi.shape[0] == 3:
        coords = cp_dwi
    else:
        raise ValueError(f"cp_dwi must have 3 or 4 rows, got {cp_dwi.shape}")

    n_coords = coords.shape[1]

    if n_coords % points_num != 0:
        raise ValueError(f"Invalid cp_dwi size {n_coords} for points_num {points_num}")

    columns = n_coords // points_num
    print(f"[SAMPLE] columns={columns}, depths={points_num}")

    vals = np.zeros((columns, points_num), float)

    for d in range(points_num):
        idx = d + np.arange(columns) * points_num

        c = coords[:, idx].astype(float)
        c -= 1.0  # MATLAB 1-based → Python 0-based

        sampled = map_coordinates(
            vol, c, order=1, mode="nearest"
        )
        vals[:, d] = sampled

    return vals


# --------------- QA HELPERS ----------------- #

def _compute_mean_and_ci(
    vals: np.ndarray,
    ci_level: float = 0.95,
):
    """
    vals: (N_columns, N_depths)
    Returns:
        mean: (N_depths,)
        ci_low: (N_depths,)
        ci_high: (N_depths,)
        n_columns: int
    """
    if vals.size == 0:
        raise ValueError("Empty vals in _compute_mean_and_ci")

    n_cols = vals.shape[0]
    mean = vals.mean(axis=0)

    from scipy.stats import norm
    z = norm.ppf(0.5 + ci_level / 2.0)
    std = vals.std(axis=0, ddof=1)
    sem = std / np.sqrt(n_cols)

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
    """Save a PNG of the depth profile with shaded CI."""
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

def get_columns_in_regions_oneMM_DD(ID, input_dir, output_dir, contrast, force=False):

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    print(f"\n[INFO] Running for subject {ID}")
    print(f"[INFO] Contrast: {contrast}")
    print(f"[INFO] I/O roots:")
    print(f"    input_dir:  {input_dir}")
    print(f"    output_dir: {output_dir}")
    print(f"[INFO] Pipeline version: {PIPELINE_VERSION}")
    print(f"[INFO] FORCE mode: {force}")

    # load MRI contrast
    vol = _load_contrast_image(input_dir, ID, contrast)

    # SHARED label coord folder (contrast-agnostic)
    label_coord_dir = output_dir / ID / "columns" / "label_coord_1mm"
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

    for region in REGION_LIST:
        print(f"\n[REGION] {region}")

        lh_mat = label_coord_dir / f"lh_{region}.mat"
        rh_mat = label_coord_dir / f"rh_{region}.mat"

        if not lh_mat.is_file() or not rh_mat.is_file():
            print(f"  -> missing lh/rh coord files, skipping region")
            continue

        # expected per-region outputs
        lh_path = per_column_dir / f"{ID}_lh_{region}_cols_{contrast}.csv"
        rh_path = per_column_dir / f"{ID}_rh_{region}_cols_{contrast}.csv"
        lh_mean_path = per_mean_dir / f"{ID}_lh_{region}_cols_{contrast}_mean.csv"
        rh_mean_path = per_mean_dir / f"{ID}_rh_{region}_cols_{contrast}_mean.csv"
        lh_png_path = qa_dir / f"{ID}_lh_{region}_profile_{contrast}.png"
        rh_png_path = qa_dir / f"{ID}_rh_{region}_profile_{contrast}.png"
        checksum_path = qa_dir / f"{ID}_{region}_cols_{contrast}.sha256"

        outputs_exist = (
            lh_path.is_file()
            and rh_path.is_file()
            and lh_mean_path.is_file()
            and rh_mean_path.is_file()
            and lh_png_path.is_file()
            and rh_png_path.is_file()
        )

        checksum_ok = False
        if checksum_path.is_file():
            try:
                stored = checksum_path.read_text().strip()
                expected = _region_checksum(ID, contrast, region)
                if stored == expected:
                    checksum_ok = True
            except Exception as e:
                print(f"  [WARN] could not read checksum for region {region}: {e}")

        if (not force) and outputs_exist and checksum_ok:
            print("  -> outputs + checksum present; skipping recompute for this region")
            continue
        elif force:
            print("  -> FORCE recompute for this region")
        else:
            if outputs_exist and not checksum_ok:
                print("  -> outputs exist but checksum missing/mismatch; recomputing region")
            else:
                print("  -> outputs missing; computing region")

        # coord matrices
        lh_cp = _load_region_cp_dwi(lh_mat, "lh")
        rh_cp = _load_region_cp_dwi(rh_mat, "rh")

        # sample
        lh_vals = _sample_columns(vol, lh_cp, points_num)
        rh_vals = _sample_columns(vol, rh_cp, points_num)

        # --------- save per-column output ---------
        np.savetxt(lh_path, lh_vals, delimiter=",")
        np.savetxt(rh_path, rh_vals, delimiter=",")

        print(f"  -> LH per-column saved ({lh_vals.shape}) → {lh_path}")
        print(f"  -> RH per-column saved ({rh_vals.shape}) → {rh_path}")

        summary.append(["lh", region, lh_vals.shape[0], lh_vals.shape[1], lh_path.name])
        summary.append(["rh", region, rh_vals.shape[0], rh_vals.shape[1], rh_path.name])

        # --------- region mean depth profile (21×1 CSV) ---------
        lh_mean_profile = lh_vals.mean(axis=0).reshape(-1, 1)
        rh_mean_profile = rh_vals.mean(axis=0).reshape(-1, 1)

        np.savetxt(lh_mean_path, lh_mean_profile, delimiter=",")
        np.savetxt(rh_mean_path, rh_mean_profile, delimiter=",")

        print(f"  -> LH depth-mean saved (21×1) → {lh_mean_path}")
        print(f"  -> RH depth-mean saved (21×1) → {rh_mean_path}")

        # --------- QA: mean + CI + plots ---------
        # LH
        lh_mean, lh_ci_low, lh_ci_high, lh_n = _compute_mean_and_ci(lh_vals, ci_level=0.95)
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

        # --------- region checksum marker ---------
        try:
            checksum = _region_checksum(ID, contrast, region)
            checksum_path.write_text(checksum + "\n")
        except Exception as e:
            print(f"  [WARN] could not write checksum for region {region}: {e}")

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
    p.add_argument(
        "--input-dir",
        required=True,
        help="Root containing contrast images in one of the supported layouts.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Root for <ID>/columns and <ID>/<contrast>/ outputs.",
    )
    p.add_argument(
        "--contrast",
        required=True,
        help="Contrast name, e.g. QSM, T1, FA, MD",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Recompute regions even if outputs + checksum exist.",
    )
    args = p.parse_args()

    get_columns_in_regions_oneMM_DD(
        args.ID, args.input_dir, args.output_dir, args.contrast, force=args.force
    )


if __name__ == "__main__":
    _cli()
