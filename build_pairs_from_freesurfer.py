#!/usr/bin/env python

"""
build_pairs_from_freesurfer.py

Builds "pair" files for cortical thickness based on FreeSurfer
white and pial surfaces.

For each hemisphere, we assume:

    - lh.white and lh.pial share the same vertex count and ordering
    - rh.white and rh.pial share the same vertex count and ordering

We construct:

    lh_pair: (N_lh, 6) array
        [white_x, white_y, white_z,  pial_x, pial_y, pial_z]

    rh_pair: (N_rh, 6) array
        [white_x, white_y, white_z,  pial_x, pial_y, pial_z]

and save them as:

    <output_dir>/<ID>/columns_1mm/<ID>_pair_lh.mat   (var: lh_pair)
    <output_dir>/<ID>/columns_1mm/<ID>_pair_rh.mat   (var: rh_pair)

These are then used by get_thickness.py, which computes:

    thickness = ||pial - white|| (per vertex)

Assumed directory layout (mirrors your other scripts):

    Surfaces (from FreeSurfer recon-all) in:
        <output_dir>/<ID>/<ID>/surf/
            lh.white
            lh.pial
            rh.white
            rh.pial

    Pair files will be written to:
        <output_dir>/<ID>/columns_1mm/

Usage example:

    python build_pairs_from_freesurfer.py \
        --ID S00775 \
        --output-dir /mnt/newStor/paros/paros_WORK/column_code_tester/
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.io import savemat
import nibabel as nib


def _load_fs_surface(surf_path: Path) -> np.ndarray:
    """
    Load a FreeSurfer surface and return vertex coordinates (N, 3).

    surf_path: path to lh.white, lh.pial, etc.
    """
    if not surf_path.is_file():
        raise FileNotFoundError(f"Surface file not found: {surf_path}")

    coords, faces = nib.freesurfer.io.read_geometry(str(surf_path))
    print(f"[SURF] Loaded {surf_path.name}: {coords.shape[0]} vertices, {faces.shape[0]} faces")
    return coords


def _build_pair(white_coords: np.ndarray, pial_coords: np.ndarray) -> np.ndarray:
    """
    Given white and pial vertex coordinates, build pair array:

        [white_x, white_y, white_z,  pial_x, pial_y, pial_z]

    Both white_coords and pial_coords should be shape (N, 3).
    """
    if white_coords.shape != pial_coords.shape:
        raise ValueError(
            f"White and pial shapes differ: {white_coords.shape} vs {pial_coords.shape}"
        )

    pair = np.hstack([white_coords, pial_coords])  # (N, 6)
    return pair


def build_pairs_from_freesurfer(ID: str, output_dir):
    """
    Build pair files for subject ID using FreeSurfer surfaces stored under:

        <output_dir>/<ID>/<ID>/surf/

    Writes:

        <output_dir>/<ID>/columns_1mm/<ID>_pair_lh.mat  (lh_pair)
        <output_dir>/<ID>/columns_1mm/<ID>_pair_rh.mat  (rh_pair)
    """
    output_dir = Path(output_dir)

    subj_root = output_dir / ID / ID
    surf_dir = subj_root / "surf"

    lh_white_path = surf_dir / "lh.white"
    lh_pial_path  = surf_dir / "lh.pial"
    rh_white_path = surf_dir / "rh.white"
    rh_pial_path  = surf_dir / "rh.pial"

    print(f"[INFO] Subject:    {ID}")
    print(f"[INFO] Surf dir:   {surf_dir}")

    # Load surfaces
    lh_white = _load_fs_surface(lh_white_path)
    lh_pial  = _load_fs_surface(lh_pial_path)
    rh_white = _load_fs_surface(rh_white_path)
    rh_pial  = _load_fs_surface(rh_pial_path)

    # Build pairs
    lh_pair = _build_pair(lh_white, lh_pial)
    rh_pair = _build_pair(rh_white, rh_pial)

    print(f"[PAIR] LH pair shape: {lh_pair.shape}")
    print(f"[PAIR] RH pair shape: {rh_pair.shape}")

    # Output directory for pair files
    pair_dir = output_dir / ID / "columns_1mm"
    pair_dir.mkdir(parents=True, exist_ok=True)

    lh_out_mat = pair_dir / f"{ID}_pair_lh.mat"
    rh_out_mat = pair_dir / f"{ID}_pair_rh.mat"

    savemat(str(lh_out_mat), {"lh_pair": lh_pair})
    savemat(str(rh_out_mat), {"rh_pair": rh_pair})

    print(f"[SAVE] Wrote LH pair to: {lh_out_mat}")
    print(f"[SAVE] Wrote RH pair to: {rh_out_mat}")

    # Optional: also drop quick CSVs for debugging (one row per vertex)
    lh_csv = pair_dir / f"{ID}_pair_lh.csv"
    rh_csv = pair_dir / f"{ID}_pair_rh.csv"

    np.savetxt(lh_csv, lh_pair, delimiter=",")
    np.savetxt(rh_csv, rh_pair, delimiter=",")

    print(f"[SAVE] Wrote LH pair CSV: {lh_csv}")
    print(f"[SAVE] Wrote RH pair CSV: {rh_csv}")

    print("\n[DONE] Pair construction complete.")


def _cli():
    parser = argparse.ArgumentParser(
        description="Build cortical white–pial vertex pairs from FreeSurfer surfaces."
    )
    parser.add_argument("--ID", required=True, help="Subject ID, e.g. S00775")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Root dir where <ID>/<ID>/surf/ lives and where columns_1mm will be written.",
    )
    args = parser.parse_args()

    build_pairs_from_freesurfer(args.ID, args.output_dir)


if __name__ == "__main__":
    _cli()
