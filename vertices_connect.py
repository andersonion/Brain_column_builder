#!/usr/bin/env python

"""
vertices_connect.py

Connect FreeSurfer white and pial vertices into cortical columns,
sample points along those columns, and transform them into DWI CRS space.

Original MATLAB behavior:

    output_dir = '/.../ad_decode_test/output/';
    ID = 'S04491';
    lh_white = strcat(output_dir,ID,'/surf/lh.white');
    lh_pial  = strcat(output_dir,ID,'/surf/lh.pial');
    rh_white = strcat(output_dir,ID,'/surf/rh.white');
    rh_pial  = strcat(output_dir,ID,'/surf/rh.pial');
    trans_M_dir = strcat(output_dir,ID,'/DWI2T1_dti_upsampled.dat');

In this Python version we **decouple** the locations:

    - FreeSurfer surfaces come from a FreeSurfer subjects dir:

          <fs_subjects_dir>/<ID>/surf/lh.white
          <fs_subjects_dir>/<ID>/surf/lh.pial
          <fs_subjects_dir>/<ID>/surf/rh.white
          <fs_subjects_dir>/<ID>/surf/rh.pial

      where fs_subjects_dir is either:
          * explicitly passed, or
          * taken from $SUBJECTS_DIR, or
          * falls back to root_dir (for legacy layout).

    - Column and transform files live under root_dir:

          <root_dir>/<ID>/DWI2T1_dti_upsampled.dat
          <root_dir>/<ID>/columns/...

Outputs:

    <root_dir>/<ID>/columns/<ID>_pair_lh.mat
    <root_dir>/<ID>/columns/<ID>_pair_rh.mat
    <root_dir>/<ID>/columns/<ID>_column_lh.mat
    <root_dir>/<ID>/columns/<ID>_column_rh.mat
    <root_dir>/<ID>/columns/<ID>_column_lh_dwi.mat
    <root_dir>/<ID>/columns/<ID>_column_rh_dwi.mat
    <root_dir>/<ID>/columns/<ID>lh_t.white
    <root_dir>/<ID>/columns/<ID>rh_t.white
"""

import argparse
import os
from pathlib import Path

import numpy as np
from scipy.io import savemat
import nibabel.freesurfer.io as fsio


POINTS_NUM = 21  # number of samples along each column (including endpoints)


# ----------------- FreeSurfer-style transforms ----------------- #

def vox2ras_tkreg(voldim, voxres):
    """
    Approximate FreeSurfer vox2ras-tkreg transform.

    voldim: iterable (Nx, Ny, Nz)
    voxres: iterable (dx, dy, dz)

    Returns a 4×4 matrix mapping voxel indices (i,j,k) in tkreg convention
    to RAS coordinates.

        M = [[-dx,   0,   0,  dx*nx/2],
             [  0,   0,  dz, -dz*nz/2],
             [  0, -dy,   0,  dy*ny/2],
             [  0,   0,   0,        1]]
    """
    nx, ny, nz = [float(v) for v in voldim]
    dx, dy, dz = [float(v) for v in voxres]

    M = np.zeros((4, 4), dtype=float)
    M[0, 0] = -dx
    M[0, 3] = dx * nx / 2.0

    M[1, 2] = dz
    M[1, 3] = -dz * nz / 2.0

    M[2, 1] = -dy
    M[2, 3] = dy * ny / 2.0

    M[3, 3] = 1.0

    return M


def vox2ras_0to1(M):
    """
    Convert a vox2ras-tkreg matrix from 0-based to 1-based indexing.

    The MATLAB code uses: T_mov = vox2ras_tkreg(...); T_mov = vox2ras_0to1(T_mov);
    Here we implement a standard shift by +0.5 voxels in each dimension.

    R1 = M * ([I  -0.5; 0 1]), effectively moving from indices centered
    at voxel corners to voxel centers.
    """
    shift = np.eye(4, dtype=float)
    shift[0:3, 3] = -0.5
    return M @ shift


# ----------------- Core helpers ----------------- #

def generate_inner_points(pair: np.ndarray, points_num: int) -> np.ndarray:
    """
    Generate points along each white→pial column.

    pair: (N, 6), rows are [xw, yw, zw, xp, yp, zp]
    points_num: number of points (including both endpoints)

    Returns:
        column_points: (N * points_num, 3), stacked vertex-major:

            [v0_d0, v0_d1, ..., v0_dK, v1_d0, ...]
    """
    pair = np.asarray(pair, float)
    if pair.ndim != 2 or pair.shape[1] != 6:
        raise ValueError(f"pair must be (N,6), got {pair.shape}")

    white = pair[:, 0:3]   # (N,3)
    pial = pair[:, 3:6]    # (N,3)

    N = white.shape[0]
    t = np.linspace(0.0, 1.0, points_num, dtype=float)  # 0..1 inclusive

    # (N, points_num, 3)
    pts = white[:, None, :] * (1.0 - t)[None, :, None] + pial[:, None, :] * t[None, :, None]

    # Flatten to (N*points_num, 3)
    column_points = pts.reshape(N * points_num, 3)
    return column_points


def columns_to_cp_matrix(column_points: np.ndarray) -> np.ndarray:
    """
    Convert column_points (N*points_num, 3) into a 3×(N*points_num) matrix,
    as used by the rest of the pipeline.

    Here we simply transpose:

        column_points: (N*points_num, 3)
        -> (3, N*points_num)
    """
    column_points = np.asarray(column_points, float)
    if column_points.ndim != 2 or column_points.shape[1] != 3:
        raise ValueError(f"column_points must be (N*points_num, 3), got {column_points.shape}")

    return column_points.T  # (3, N*points_num)


def load_trans_M(dat_path: Path) -> np.ndarray:
    """
    Load DWI2T1 transform matrix from a .dat file, mimicking:

        data = importdata(trans_M_dir);
        trans_M = data.data(4:19);
        trans_M = reshape(trans_M,[4, 4])';

    Here we:
        - load all numeric values,
        - take elements [3:19] (0-based),
        - reshape 4×4, transpose.
    """
    vals = np.loadtxt(str(dat_path), dtype=float)
    vals = np.atleast_1d(vals).flatten()

    if vals.size < 19:
        raise ValueError(
            f"Expected at least 19 numeric values in {dat_path}, got {vals.size}"
        )

    trans_vals = vals[3:19]  # MATLAB 4:19 → zero-based 3:19
    if trans_vals.size != 16:
        raise ValueError(
            f"Expected 16 values for trans_M, got {trans_vals.size} from {dat_path}"
        )

    trans_M = trans_vals.reshape((4, 4)).T
    return trans_M


# ----------------- Main worker ----------------- #

def vertices_connect(
    ID: str,
    root_dir,
    fs_subjects_dir=None,
    voldim=(512, 512, 272),
    voxres=(0.5, 0.5, 0.5),
):
    """
    Main driver for building cortical columns and DWI-space coordinates
    for one subject.

    Parameters
    ----------
    ID : str
        Subject ID, e.g. 'S04491'.
    root_dir : str or Path
        Root directory for **output products** and transform file, containing:

            <root_dir>/<ID>/DWI2T1_dti_upsampled.dat
            <root_dir>/<ID>/columns/...

    fs_subjects_dir : str or Path, optional
        FreeSurfer SUBJECTS_DIR-like root containing <ID>/surf.
        If None:
            - If $SUBJECTS_DIR is set, use that.
            - Else, fall back to root_dir (legacy behavior).

    voldim : (Nx, Ny, Nz)
        Volume dimensions for T_mov (default: [512, 512, 272])
    voxres : (dx, dy, dz)
        Voxel resolution for T_mov (default: [0.5, 0.5, 0.5])
    """
    root_dir = Path(root_dir)

    if fs_subjects_dir is None:
        env_sd = os.getenv("SUBJECTS_DIR")
        if env_sd:
            fs_subjects_dir = Path(env_sd)
        else:
            fs_subjects_dir = root_dir
    else:
        fs_subjects_dir = Path(fs_subjects_dir)

    subj_fs_dir = fs_subjects_dir / ID
    surf_dir = subj_fs_dir / "surf"

    columns_dir = root_dir / ID / "columns"
    columns_dir.mkdir(parents=True, exist_ok=True)

    trans_M_path = root_dir / ID / "DWI2T1_dti_upsampled.dat"

    print("\n[INFO] Subject:", ID)
    print("[INFO] Root dir (outputs/transform):", root_dir)
    print("[INFO] FS subjects dir:             ", fs_subjects_dir)
    print("[INFO] FS subject dir:              ", subj_fs_dir)
    print("[INFO] Surf dir:                    ", surf_dir)
    print("[INFO] Columns dir:                 ", columns_dir)

    # ---- Surface paths ----
    lh_white_path = surf_dir / "lh.white"
    lh_pial_path = surf_dir / "lh.pial"
    rh_white_path = surf_dir / "rh.white"
    rh_pial_path = surf_dir / "rh.pial"

    for p in [lh_white_path, lh_pial_path, rh_white_path, rh_pial_path]:
        if not p.is_file():
            raise FileNotFoundError(f"Missing surface file: {p}")

    # ---- Transform path ----
    if not trans_M_path.is_file():
        raise FileNotFoundError(f"Missing transform file: {trans_M_path}")

    # ========================
    # 1) Load surfaces
    # ========================
    print("\n[STEP 1] Load white/pial surfaces")

    lh_white_vertices, lh_white_faces = fsio.read_geometry(str(lh_white_path))
    lh_pial_vertices, lh_pial_faces = fsio.read_geometry(str(lh_pial_path))
    rh_white_vertices, rh_white_faces = fsio.read_geometry(str(rh_white_path))
    rh_pial_vertices, rh_pial_faces = fsio.read_geometry(str(rh_pial_path))

    print(f"[SURF] LH white: {lh_white_vertices.shape[0]} vertices")
    print(f"[SURF] LH pial:  {lh_pial_vertices.shape[0]} vertices")
    print(f"[SURF] RH white: {rh_white_vertices.shape[0]} vertices")
    print(f"[SURF] RH pial:  {rh_pial_vertices.shape[0]} vertices")

    if lh_white_vertices.shape != lh_pial_vertices.shape:
        raise ValueError("LH white and pial have different shapes.")
    if rh_white_vertices.shape != rh_pial_vertices.shape:
        raise ValueError("RH white and pial have different shapes.")

    # ========================
    # 2) Build pairs
    # ========================
    print("\n[STEP 2] Build white–pial pairs")

    lh_pair = np.hstack([lh_white_vertices, lh_pial_vertices])  # (N,6)
    rh_pair = np.hstack([rh_white_vertices, rh_pial_vertices])  # (N,6)

    lh_pair_mat = columns_dir / f"{ID}_pair_lh.mat"
    rh_pair_mat = columns_dir / f"{ID}_pair_rh.mat"

    savemat(str(lh_pair_mat), {"lh_pair": lh_pair})
    savemat(str(rh_pair_mat), {"rh_pair": rh_pair})

    print(f"[SAVE] LH pair → {lh_pair_mat}")
    print(f"[SAVE] RH pair → {rh_pair_mat}")

    # ========================
    # 3) Generate column points
    # ========================
    print("\n[STEP 3] Generate column points (points_num =", POINTS_NUM, ")")

    lh_column_points = generate_inner_points(lh_pair, POINTS_NUM)
    rh_column_points = generate_inner_points(rh_pair, POINTS_NUM)

    lh_column_mat = columns_dir / f"{ID}_column_lh.mat"
    rh_column_mat = columns_dir / f"{ID}_column_rh.mat"

    savemat(str(lh_column_mat), {"lh_column_points": lh_column_points})
    savemat(str(rh_column_mat), {"rh_column_points": rh_column_points})

    print(f"[SAVE] LH column points → {lh_column_mat} ({lh_column_points.shape})")
    print(f"[SAVE] RH column points → {rh_column_mat} ({rh_column_points.shape})")

    # ========================
    # 4) Load transform, build T_mov
    # ========================
    print("\n[STEP 4] Load transform and build T_mov")

    trans_M = load_trans_M(trans_M_path)
    print("[TRANS] trans_M (DWI2T1) =\n", trans_M)

    T_mov = vox2ras_tkreg(voldim, voxres)
    T_mov = vox2ras_0to1(T_mov)
    print("[TRANS] T_mov (vox2ras-tkreg, 0->1) =\n", T_mov)

    T_mov_inv = np.linalg.inv(T_mov)

    # ========================
    # 5) Transform columns to DWI CRS
    # ========================
    print("\n[STEP 5] Transform column points to DWI CRS")

    # LH
    lh_cp = columns_to_cp_matrix(lh_column_points)  # (3, Npts)
    lh_cp_h = np.vstack([lh_cp, np.ones((1, lh_cp.shape[1]), dtype=float)])  # (4, Npts)

    lh_cp_dwi = T_mov_inv @ (trans_M @ lh_cp_h)  # (4, Npts)

    # RH
    rh_cp = columns_to_cp_matrix(rh_column_points)  # (3, Npts)
    rh_cp_h = np.vstack([rh_cp, np.ones((1, rh_cp.shape[1]), dtype=float)])  # (4, Npts)

    rh_cp_dwi = T_mov_inv @ (trans_M @ rh_cp_h)  # (4, Npts)

    if np.isnan(lh_cp_dwi).any():
        print(f"[WARN] {ID} LH cp_dwi has NaN values")
    if np.isnan(rh_cp_dwi).any():
        print(f"[WARN] {ID} RH cp_dwi has NaN values")

    lh_cp_dwi_mat = columns_dir / f"{ID}_column_lh_dwi.mat"
    rh_cp_dwi_mat = columns_dir / f"{ID}_column_rh_dwi.mat"

    savemat(str(lh_cp_dwi_mat), {"lh_cp_dwi": lh_cp_dwi})
    savemat(str(rh_cp_dwi_mat), {"rh_cp_dwi": rh_cp_dwi})

    print(f"[SAVE] LH cp_dwi → {lh_cp_dwi_mat} ({lh_cp_dwi.shape})")
    print(f"[SAVE] RH cp_dwi → {rh_cp_dwi_mat} ({rh_cp_dwi.shape})")

    # ========================
    # 6) Transform white surfaces and save
    # ========================
    print("\n[STEP 6] Transform white surfaces and save")

    def _transform_white(white_vertices):
        # white_vertices: (N,3)
        v = white_vertices.T  # (3,N)
        v_h = np.vstack([v, np.ones((1, v.shape[1]), dtype=float)])  # (4,N)
        v_t = trans_M @ v_h   # (4,N)   (MATLAB: lh_white_f = trans_M * lh_white_f)
        return v_t[0:3, :].T  # (N,3)

    lh_white_t = _transform_white(lh_white_vertices)
    rh_white_t = _transform_white(rh_white_vertices)

    lh_white_out = columns_dir / f"{ID}lh_t.white"
    rh_white_out = columns_dir / f"{ID}rh_t.white"

    fsio.write_geometry(str(lh_white_out), lh_white_t, lh_white_faces)
    fsio.write_geometry(str(rh_white_out), rh_white_t, rh_white_faces)

    print(f"[SAVE] LH transformed white surface → {lh_white_out}")
    print(f"[SAVE] RH transformed white surface → {rh_white_out}")

    print("\n[DONE] vertices_connect complete for", ID)


# ----------------- CLI ----------------- #

def _cli():
    parser = argparse.ArgumentParser(
        description="Connect white/pial vertices into columns and transform to DWI CRS."
    )
    parser.add_argument("--ID", required=True, help="Subject ID, e.g. S04491")
    parser.add_argument(
        "--root-dir",
        required=True,
        help="Root dir for outputs and transform (contains <ID>/DWI2T1_dti_upsampled.dat).",
    )
    parser.add_argument(
        "--fs-subjects-dir",
        default=None,
        help="FreeSurfer SUBJECTS_DIR-style root containing <ID>/surf. "
             "If omitted, uses $SUBJECTS_DIR or falls back to root-dir.",
    )
    parser.add_argument(
        "--voldim",
        nargs=3,
        type=int,
        default=[512, 512, 272],
        help="Volume dimensions (Nx Ny Nz) for T_mov (default: 512 512 272).",
    )
    parser.add_argument(
        "--voxres",
        nargs=3,
        type=float,
        default=[0.5, 0.5, 0.5],
        help="Voxel resolution (dx dy dz) for T_mov (default: 0.5 0.5 0.5).",
    )

    args = parser.parse_args()

    vertices_connect(
        ID=args.ID,
        root_dir=args.root_dir,
        fs_subjects_dir=args.fs_subjects_dir,
        voldim=args.voldim,
        voxres=args.voxres,
    )


if __name__ == "__main__":
    _cli()
