#!/usr/bin/env python3
"""
Python port of MATLAB:
    view_columns_in_regions_oneMM_DD(ID, input_dir, output_dir)

Notes / assumptions:
- Expects QSM image at:
    /Volumes/newJetStor/newJetStor/paros/paros_WORK/hanwen/ad_decode_test/input/{ID}/{ID}_QSM_masked.nii.gz
  (You can override with --image-path)

- Expects .mat files at:
    {output_dir}/{ID}/QSM/label_coord_1mm/lh_<region>.mat
    {output_dir}/{ID}/QSM/label_coord_1mm/rh_<region>.mat

- Each .mat contains lh_cp_dwi / rh_cp_dwi shaped (3, N), where N is multiple of 21.
- Coordinates are treated as voxel coordinates in *nibabel array order* (x,y,z with 0-based indexing).
  If your MATLAB coordinates are 1-based or swapped axes, use the CLI options:
    --one-based
    --swap-xy
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.io import loadmat
from scipy.ndimage import map_coordinates

import matplotlib
matplotlib.use("Agg")  # headless
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


def _load_cp_mat(mat_path: Path, var_name: str) -> np.ndarray:
    """
    Load a 3xN coordinate matrix from a .mat file.
    """
    md = loadmat(mat_path)
    if var_name not in md:
        # helpfully list keys (excluding matlab internals)
        keys = [k for k in md.keys() if not k.startswith("__")]
        raise KeyError(f"Missing variable '{var_name}' in {mat_path}. Found: {keys}")
    arr = np.asarray(md[var_name])
    # Ensure shape (3, N)
    if arr.ndim != 2:
        raise ValueError(f"{var_name} in {mat_path} should be 2D, got shape {arr.shape}")
    if arr.shape[0] == 3:
        return arr
    if arr.shape[1] == 3:
        return arr.T
    raise ValueError(f"{var_name} in {mat_path} expected shape (3,N) or (N,3), got {arr.shape}")


def _interp3_sum_by_depth(vol: np.ndarray, cp: np.ndarray, points_num: int = 21,
                          one_based: bool = False, swap_xy: bool = False) -> np.ndarray:
    """
    Replicates the MATLAB pattern:

        columns_num = size(cp,2)/21
        for i=1:21
            index = i:21:(i + 21*(columns_num-1));
            interp_vol = interp3(vol, cp(1,index), cp(2,index), cp(3,index));
            values(i) = sum(interp_vol)

    Using scipy.ndimage.map_coordinates (order=1 == trilinear).

    cp is 3xN, with N multiple of points_num.
    """
    if cp.shape[0] != 3:
        raise ValueError(f"cp must be (3,N); got {cp.shape}")

    N = cp.shape[1]
    if N % points_num != 0:
        raise ValueError(f"N={N} is not a multiple of points_num={points_num}")

    # Copy + optional transforms
    coords = cp.astype(np.float64).copy()

    if one_based:
        coords -= 1.0  # MATLAB -> Python

    if swap_xy:
        coords[[0, 1], :] = coords[[1, 0], :]

    # MATLAB interp3 expects X,Y,Z in array index space depending on how vol is stored.
    # We assume coords are already aligned to nibabel array indexing (x,y,z) == (i,j,k).
    # scipy map_coordinates expects coords as (dim, npoints) in (axis0, axis1, axis2) order.
    # If vol is (X,Y,Z) in Python, axis0 is X, axis1 is Y, axis2 is Z -> this matches coords.
    columns_num = N // points_num
    values = np.zeros(points_num, dtype=np.float64)

    # For each depth i, take indices i, i+points_num, ...
    for i in range(points_num):
        idx = i + points_num * np.arange(columns_num)
        x = coords[0, idx]
        y = coords[1, idx]
        z = coords[2, idx]

        # map_coordinates wants coords stacked as (ndim, npoints) in axis order
        sample = map_coordinates(
            vol,
            np.vstack([x, y, z]),
            order=1,           # trilinear
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        values[i] = float(np.sum(sample))

    return values


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ID", required=True, help="Subject ID, e.g. S00775")
    ap.add_argument("--input-dir", default="", help="(kept for symmetry; not used unless building default image path)")
    ap.add_argument("--output-dir", required=True, help="Output base directory")
    ap.add_argument("--points-num", type=int, default=21, help="Depth samples per column (default 21)")
    ap.add_argument("--image-path", default="", help="Override QSM NIfTI path")
    ap.add_argument("--one-based", action="store_true",
                    help="If coords in .mat are 1-based (MATLAB style), subtract 1 before sampling")
    ap.add_argument("--swap-xy", action="store_true",
                    help="If coords are stored as (row,col,slice) but vol is (x,y,z), swap x<->y")
    ap.add_argument("--dpi", type=int, default=150, help="JPEG DPI")

    args = ap.parse_args()

    ID = args.ID
    output_dir = Path(args.output_dir)

    # Match your MATLAB hardcoded path unless overridden
    if args.image_path:
        image_path = Path(args.image_path)
    else:
        image_path = Path(
            f"/Volumes/newJetStor/newJetStor/paros/paros_WORK/hanwen/ad_decode_test/input/{ID}/{ID}_QSM_masked.nii.gz"
        )

    print(f"The subject is: {ID}")
    if not image_path.is_file():
        print(f"Subject {ID} doesnt have QSM image: {image_path}")
        return 0

    # Load NIfTI
    img = nib.load(str(image_path))
    vol = img.get_fdata(dtype=np.float32)  # float for interpolation

    # Output folders
    qsm_dir = output_dir / ID / "QSM"
    label_dir = qsm_dir / "label_coord_1mm"
    qsm_dir.mkdir(parents=True, exist_ok=True)

    for region_name in REGION_LIST:
        lh_mat = label_dir / f"lh_{region_name}.mat"
        rh_mat = label_dir / f"rh_{region_name}.mat"

        if not lh_mat.is_file() or not rh_mat.is_file():
            print(f"Subject {ID} doesnt have label files for region {region_name}")
            return 0

        # Load coords
        lh_cp = _load_cp_mat(lh_mat, "lh_cp_dwi")
        rh_cp = _load_cp_mat(rh_mat, "rh_cp_dwi")

        # Concatenate like MATLAB: [lh_cp_dwi(:,:), rh_cp_dwi(:,:)]
        rl_cp = np.concatenate([lh_cp, rh_cp], axis=1)
        points_size = rl_cp.shape[1]
        points_num = args.points_num

        # Collect values by depth
        lh_values_sum = _interp3_sum_by_depth(
            vol, lh_cp, points_num=points_num, one_based=args.one_based, swap_xy=args.swap_xy
        )
        rh_values_sum = _interp3_sum_by_depth(
            vol, rh_cp, points_num=points_num, one_based=args.one_based, swap_xy=args.swap_xy
        )

        # Normalize like MATLAB:
        # rl_values = (lh_values + rh_values) ./ (points_size / points_num);
        # lh_values = lh_values ./ (size(lh_cp_dwi,2) / points_num);
        # rh_values = rh_values ./ (size(rh_cp_dwi,2) / points_num);
        rl_values = (lh_values_sum + rh_values_sum) / (points_size / points_num)
        lh_values = lh_values_sum / (lh_cp.shape[1] / points_num)
        rh_values = rh_values_sum / (rh_cp.shape[1] / points_num)

        # Plot
        x = np.arange(points_num)  # 0..20
        fig = plt.figure(figsize=(6.4, 4.8))
        plt.plot(x, rl_values, linestyle="--", label="lr")
        plt.plot(x, lh_values, label="lh")
        plt.plot(x, rh_values, label="rh")
        plt.legend()
        plt.grid(True)

        plt.xticks([0, points_num - 1], ["WM/GM", "Pial"])
        plt.ylabel("QSM")
        title_region = region_name.replace("_", " ")
        plt.title(f"{ID} {title_region}")

        jpg_path = qsm_dir / f"{ID}_{region_name}_QSM.jpg"
        plt.savefig(jpg_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

        # Write CSVs (1 row, like writematrix of 1x21)
        np.savetxt(qsm_dir / f"{ID}_rl_{region_name}_QSM.csv", rl_values.reshape(1, -1), delimiter=",")
        np.savetxt(qsm_dir / f"{ID}_lh_{region_name}_QSM.csv", lh_values.reshape(1, -1), delimiter=",")
        np.savetxt(qsm_dir / f"{ID}_rh_{region_name}_QSM.csv", rh_values.reshape(1, -1), delimiter=",")

        print(f"[OK] {region_name}: saved plot + csvs")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
