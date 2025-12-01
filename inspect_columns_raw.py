#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def summarize_points(name, pts):
    pts = np.asarray(pts, float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"{name} has shape {pts.shape}, expected (N,3)")
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    print(f"[{name}] N={pts.shape[0]}")
    print(f"  x: min={x.min():.2f}, max={x.max():.2f}")
    print(f"  y: min={y.min():.2f}, max={y.max():.2f}")
    print(f"  z: min={z.min():.2f}, max={z.max():.2f}")
    r = np.sqrt(x**2 + y**2 + z**2)
    print(f"  |r|: min={r.min():.2f}, max={r.max():.2f}")
    print("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--columns-mat",
        required=True,
        help="Path to *_column_*.mat (e.g., D0007_column_rh.mat)",
    )
    ap.add_argument(
        "--key",
        default=None,
        help="Variable name in MAT file (default: auto-detect first Nx3).",
    )
    args = ap.parse_args()

    mat_path = Path(args.columns_mat)
    if not mat_path.is_file():
        raise FileNotFoundError(mat_path)

    print(f"[INFO] Loading MAT: {mat_path}")
    data = loadmat(str(mat_path))

    if args.key is not None:
        if args.key not in data:
            raise KeyError(f"Key '{args.key}' not found in {mat_path.name}")
        pts = data[args.key]
        summarize_points(args.key, pts)
        return

    # auto-detect: pick first variable that looks like (N,3)
    for k, v in data.items():
        if k.startswith("__"):
            continue
        arr = np.asarray(v)
        if arr.ndim == 2 and arr.shape[1] == 3:
            print(f"[INFO] Auto-selected key '{k}' with shape {arr.shape}")
            summarize_points(k, arr)
            break
    else:
        raise RuntimeError(f"No (N,3) variable found in {mat_path.name}")


if __name__ == "__main__":
    main()
