#!/usr/bin/env python

import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.io import loadmat


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Quick sanity check: sample values from a DWI/AD volume at "
            "coordinates stored in *_column_*_dwi.mat."
        )
    )
    parser.add_argument(
        "--img",
        required=True,
        help="Path to NIfTI image (e.g. D0007_ad.nii.gz)",
    )
    parser.add_argument(
        "--mat",
        required=True,
        help="Path to MAT file (e.g. D0007_column_rh_dwi.mat or D0007_column_lh_dwi.mat)",
    )
    parser.add_argument(
        "--key",
        default="rh_cp_dwi",
        help="Variable name inside MAT file (lh_cp_dwi or rh_cp_dwi). Default: rh_cp_dwi",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=20,
        help="Number of coordinates to print. Default: 20",
    )

    args = parser.parse_args()

    img_path = Path(args.img)
    mat_path = Path(args.mat)

    print(f"[INFO] Loading image: {img_path}")
    nii = nib.load(str(img_path))
    vol = nii.get_fdata()
    shape = vol.shape
    print(f"[INFO] Volume shape: {shape}")

    print(f"[INFO] Loading MAT: {mat_path} (key={args.key})")
    data = loadmat(str(mat_path))
    if args.key not in data:
        raise KeyError(
            f"Key '{args.key}' not found in {mat_path}. "
            f"Available keys: {', '.join(k for k in data.keys() if not k.startswith('__'))}"
        )

    cp = data[args.key]  # expected shape (4, N) or (3, N)
    print(f"[INFO] cp shape: {cp.shape}")

    if cp.shape[0] < 3:
        raise ValueError(f"Expected cp to have at least 3 rows (x,y,z), got {cp.shape}")

    # Take first 3 rows as CRS voxel coords
    coords = cp[:3, :]  # (3, N)
    coords_int = np.round(coords).astype(int)

    print("\n[INFO] Coordinate ranges (rounded):")
    for dim_name, dim_vals, dim_len in zip(
        ["i", "j", "k"], coords_int, shape
    ):
        print(
            f"  {dim_name}: min={dim_vals.min()} max={dim_vals.max()} "
            f"(valid range 0..{dim_len-1})"
        )

    print("\n[INFO] Sampling first {} coordinates:".format(args.n))
    n = min(args.n, coords_int.shape[1])
    for idx in range(n):
        i, j, k = coords_int[:, idx]

        in_bounds = (
            0 <= i < shape[0]
            and 0 <= j < shape[1]
            and 0 <= k < shape[2]
        )

        if in_bounds:
            val = vol[i, j, k]
            print(f"{idx:3d}: (i,j,k)=({i:4d}, {j:4d}, {k:4d})  value={val:.6g}")
        else:
            print(
                f"{idx:3d}: (i,j,k)=({i:4d}, {j:4d}, {k:4d})  -> OUTSIDE VOLUME BOUNDS"
            )


if __name__ == "__main__":
    main()
