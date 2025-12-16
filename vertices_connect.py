#!/usr/bin/env python3

"""
vertices_connect.py

Connect FreeSurfer white and pial vertices into cortical columns,
sample points along those columns, and transform them into DWI CRS space.

Assumed directory layout
------------------------

We assume a per-subject layout like:

    <root_dir>/<ID>/
        DWI2T1_dti_*.dat         (or some other transform .dat file)
        <ID>_dwi_masked.nii.gz   (DWI volume used for transform)
        columns/                 (created by this script)
        <ID>/
            surf/
                lh.white
                lh.pial
                rh.white
                rh.pial
            label/
                lh.aparc.annot
                rh.aparc.annot

So FreeSurfer-style surfaces live under `<root_dir>/<ID>/<ID>/surf`,
while this script writes its outputs under `<root_dir>/<ID>/columns`.

Transform file flexibility
--------------------------

By default we try to locate a transform .dat file inside `<root_dir>/<ID>`:

Priority order:

    1) DWI2T1_dti_upsampled.dat
    2) <ID>_DWI2T1_dti_upsampled.dat
    3) DWI2T1_dti.dat
    4) <ID>_DWI2T1_dti.dat
    5) Any single *.dat file in <root_dir>/<ID>

You can override with:

    --transform-file my_transform.dat

T_mov (vox2ras-tkreg) is computed from the *actual* DWI NIfTI if present:
    <root_dir>/<ID>/<ID>_dwi_masked.nii.gz
Otherwise we fall back to CLI voldim/voxres.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.io import savemat
import nibabel as nib
import nibabel.freesurfer.io as fsio


POINTS_NUM = 21  # samples per column (including endpoints)


# ----------------- FreeSurfer-style transforms ----------------- #

def vox2ras_tkreg(voldim, voxres):
    """
    Approximate FreeSurfer vox2ras-tkreg transform for a given volume.
    voldim: (Nx, Ny, Nz), voxres: (dx, dy, dz)
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


def vox2ras_0to1_matlab_form(M0: np.ndarray) -> np.ndarray:
    M0 = np.asarray(M0, dtype=float)
    Q = np.zeros((4, 4), dtype=float)
    Q[0:3, 3] = 1.0
    return np.linalg.inv(np.linalg.inv(M0) + Q)


# ----------------- Helpers ----------------- #

def generate_inner_points(pair: np.ndarray, points_num: int) -> np.ndarray:
    """
    Given white/pial pairs (N,6), generate column sample points (N*points_num,3).
    """
    pair = np.asarray(pair, float)
    if pair.ndim != 2 or pair.shape[1] != 6:
        raise ValueError(f"pair must be (N,6), got {pair.shape}")

    white = pair[:, 0:3]
    pial = pair[:, 3:6]

    N = white.shape[0]
    t = np.linspace(0.0, 1.0, points_num, dtype=float)

    pts = white[:, None, :] * (1.0 - t)[None, :, None] + pial[:, None, :] * t[None, :, None]
    return pts.reshape(N * points_num, 3)


def columns_to_cp_matrix(column_points: np.ndarray) -> np.ndarray:
    """
    Convert column_points (N*points_num,3) into a 3×(N*points_num) matrix.
    """
    column_points = np.asarray(column_points, float)
    if column_points.ndim != 2 or column_points.shape[1] != 3:
        raise ValueError(f"column_points must be (N*points_num, 3), got {column_points.shape}")
    return column_points.T  # (3, N)


def _find_first_4x4_block(dat_path: Path):
    """
    Look for the first 4 consecutive lines in the file that each contain
    at least 4 float-like tokens. Interpret that as a 4×4 matrix.

    We treat the 4×4 as already in standard FreeSurfer-style row-major:
        [ R | T ]
        [ 0 | 1 ]
    i.e., NO transpose is applied when using it.
    """
    lines = dat_path.read_text().splitlines()
    n = len(lines)

    for start in range(n - 3):
        block = lines[start:start + 4]
        rows = []
        ok = True
        for line in block:
            parts = line.strip().replace(",", " ").split()
            if len(parts) < 4:
                ok = False
                break
            try:
                vals = [float(p) for p in parts[:4]]
            except ValueError:
                ok = False
                break
            rows.append(vals)

        if ok and len(rows) == 4:
            M = np.array(rows, dtype=float)
            if M.shape == (4, 4):
                return M

    return None


def _fallback_load_matrix(dat_path: Path):
    """
    Fallback: scrape all numeric tokens in the file and take the 4×4
    matrix from them, in a way roughly compatible with the old MATLAB
    importdata(data.data(4:19)) behavior.
    """
    text = dat_path.read_text()
    tokens = text.replace(",", " ").split()

    floats = []
    for tok in tokens:
        try:
            floats.append(float(tok))
        except ValueError:
            pass

    floats = np.asarray(floats, dtype=float)

    if floats.size >= 19:
        vals = floats[3:19]   # 16 values
    elif floats.size >= 16:
        vals = floats[:16]
    else:
        raise ValueError(
            f"Could not extract 16 numbers from {dat_path}; found only {floats.size} tokens."
        )

    M = vals.reshape((4, 4))
    return M


def load_trans_M(dat_path: Path) -> np.ndarray:
    """
    Load DWI2T1/T1→DWI transform matrix from .dat file.

    Strategy:
      1) Look for the first 4×4 float block (4 consecutive lines, 4 floats each).
      2) If not found, fall back to scraping numeric tokens.

    IMPORTANT: We now treat the 4×4 as already in the correct
    row-major arrangement and DO NOT transpose it. Transposing
    was turning an affine into a projective-like transform and
    blowing voxel indices up to ~1e4–1e5.
    """
    M = _find_first_4x4_block(dat_path)
    if M is not None:
        print("[PARSE] Found 4×4 block in transform file:")
        print(M)
    else:
        print("[PARSE] No clean 4×4 block found; falling back to float scrape.")
        M = _fallback_load_matrix(dat_path)
        print("[PARSE] Scraped 4×4 block:")
        print(M)

    # NO TRANSPOSE HERE.
    trans_M = M
    print("[PARSE] Final trans_M (NO transpose applied):")
    print(trans_M)
    return trans_M


def find_transform_file(subj_dir: Path, ID: str, transform_file: str | None) -> Path:
    """
    Decide which .dat file to use for the DWI↔T1 transform.
    """
    if transform_file is not None:
        candidate = subj_dir / transform_file
        if not candidate.is_file():
            raise FileNotFoundError(f"Requested transform file not found:\n  {candidate}")
        print(f"[TRANS] Using explicitly specified transform file: {candidate}")
        return candidate

    candidates = [
        subj_dir / "DWI2T1_dti_upsampled.dat",
        subj_dir / f"{ID}_DWI2T1_dti_upsampled.dat",
        subj_dir / "DWI2T1_dti.dat",
        subj_dir / f"{ID}_DWI2T1_dti.dat",
    ]
    for c in candidates:
        if c.is_file():
            print(f"[TRANS] Using transform file: {c}")
            return c

    dats = sorted(subj_dir.glob("*.dat"))
    if len(dats) == 1:
        print(f"[TRANS] Using sole .dat file in {subj_dir}: {dats[0]}")
        return dats[0]
    elif len(dats) > 1:
        msg_lines = [
            f"No preferred transform filename found in {subj_dir}.",
            "Multiple .dat files detected, cannot auto-resolve:",
        ]
        for d in dats:
            msg_lines.append(f"  - {d.name}")
        msg_lines.append(
            "Please rerun with --transform-file <name> to specify which one to use."
        )
        raise FileNotFoundError("\n".join(msg_lines))
    else:
        raise FileNotFoundError(
            f"No transform .dat file found in {subj_dir}. "
            f"Expected e.g. DWI2T1_dti.dat or DWI2T1_dti_upsampled.dat."
        )


def compute_T_mov(subj_dir: Path, ID: str, voldim_fallback, voxres_fallback):
    """
    Compute T_mov (DWI vox→RAS, 0->1) from the actual DWI NIfTI if possible.

    Preferred:
        <subj_dir>/<ID>_dwi_masked.nii.gz

    Fallback:
        use voldim_fallback / voxres_fallback from CLI.
    """
    candidates = [
        subj_dir / f"{ID}_dwi_masked.nii.gz",
        subj_dir / f"{ID}_dwi.nii.gz",
    ]

    for p in candidates:
        if p.is_file():
            print(f"[TRANS] Using mov NIfTI for T_mov: {p}")
            nii = nib.load(str(p))
            shape = nii.shape[:3]
            zooms = nii.header.get_zooms()[:3]
            print(f"[TRANS] mov shape={shape}, zooms={zooms}")
            M = vox2ras_tkreg(shape, zooms)
            M = vox2ras_0to1(M)
            print("[TRANS] T_mov computed from mov NIfTI (vox2ras-tkreg, 0->1):")
            print(M)
            return M

    print("[TRANS] No mov NIfTI found; falling back to CLI voldim/voxres.")
    print(f"[TRANS] voldim={voldim_fallback}, voxres={voxres_fallback}")
    M = vox2ras_tkreg(voldim_fallback, voxres_fallback)
    M = vox2ras_0to1(M)
    print("[TRANS] T_mov from fallback voldim/voxres (vox2ras-tkreg, 0->1):")
    print(M)
    return M


# ----------------- Main worker ----------------- #

def vertices_connect(
    ID: str,
    root_dir,
    transform_file: str | None = None,
    voldim=(512, 512, 272),
    voxres=(0.5, 0.5, 0.5),
):
    """
    Build columns and DWI-space coordinates for one subject.
    """
    root_dir = Path(root_dir)
    subj_dir = root_dir / ID
    subj_fs_dir = subj_dir / ID          # extra ID layer
    surf_dir = subj_fs_dir / "surf"
    columns_dir = subj_dir / "columns"
    columns_dir.mkdir(parents=True, exist_ok=True)

    trans_M_path = find_transform_file(subj_dir, ID, transform_file)

    print("\n[INFO] Subject:", ID)
    print("[INFO] Root dir (outputs/transform):", root_dir)
    print("[INFO] Subject dir:                 ", subj_dir)
    print("[INFO] FS subject dir:              ", subj_fs_dir)
    print("[INFO] Surf dir:                    ", surf_dir)
    print("[INFO] Columns dir:                 ", columns_dir)
    print("[INFO] Transform path:              ", trans_M_path)

    # Surfaces
    lh_white_path = surf_dir / "lh.white"
    lh_pial_path = surf_dir / "lh.pial"
    rh_white_path = surf_dir / "rh.white"
    rh_pial_path = surf_dir / "rh.pial"
    for p in [lh_white_path, lh_pial_path, rh_white_path, rh_pial_path]:
        if not p.is_file():
            raise FileNotFoundError(f"Missing surface file: {p}")

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

    print("\n[STEP 2] Build white–pial pairs")
    lh_pair = np.hstack([lh_white_vertices, lh_pial_vertices])
    rh_pair = np.hstack([rh_white_vertices, rh_pial_vertices])

    lh_pair_mat = columns_dir / f"{ID}_pair_lh.mat"
    rh_pair_mat = columns_dir / f"{ID}_pair_rh.mat"
    savemat(str(lh_pair_mat), {"lh_pair": lh_pair})
    savemat(str(rh_pair_mat), {"rh_pair": rh_pair})
    print(f"[SAVE] LH pair → {lh_pair_mat}")
    print(f"[SAVE] RH pair → {rh_pair_mat}")

    print("\n[STEP 3] Generate column points (points_num =", POINTS_NUM, ")")
    lh_column_points = generate_inner_points(lh_pair, POINTS_NUM)
    rh_column_points = generate_inner_points(rh_pair, POINTS_NUM)

    lh_column_mat = columns_dir / f"{ID}_column_lh.mat"
    rh_column_mat = columns_dir / f"{ID}_column_rh.mat"
    savemat(str(lh_column_mat), {"lh_column_points": lh_column_points})
    savemat(str(rh_column_mat), {"rh_column_points": rh_column_points})
    print(f"[SAVE] LH column points → {lh_column_mat} ({lh_column_points.shape})")
    print(f"[SAVE] RH column points → {rh_column_mat} ({rh_column_points.shape})")

    print("\n[STEP 4] Load transform and build T_mov")
    trans_M = load_trans_M(trans_M_path)
    print("[TRANS] trans_M (T1→DWI RAS) =\n", trans_M)

    T_mov = compute_T_mov(subj_dir, ID, voldim, voxres)
    print("[TRANS] T_mov (DWI vox→RAS, 0->1) =\n", T_mov)

    T_mov_inv = np.linalg.inv(T_mov)

    print("\n[STEP 5] Transform column points to DWI CRS")
    lh_cp = columns_to_cp_matrix(lh_column_points)
    rh_cp = columns_to_cp_matrix(rh_column_points)

    lh_cp_h = np.vstack([lh_cp, np.ones((1, lh_cp.shape[1]), dtype=float)])
    rh_cp_h = np.vstack([rh_cp, np.ones((1, rh_cp.shape[1]), dtype=float)])

    # original MATLAB: inv(T_mov) * trans_M * [col_RAS; 1]
    lh_cp_dwi = T_mov_inv @ (trans_M @ lh_cp_h)
    rh_cp_dwi = T_mov_inv @ (trans_M @ rh_cp_h)

    # quick range debug
    for name, arr in [("LH", lh_cp_dwi), ("RH", rh_cp_dwi)]:
        i = arr[0, :]
        j = arr[1, :]
        k = arr[2, :]
        print(
            f"[DEBUG] {name} cp_dwi voxel coord ranges: "
            f"i[min={i.min():.2f}, max={i.max():.2f}], "
            f"j[min={j.min():.2f}, max={j.max():.2f}], "
            f"k[min={k.min():.2f}, max={k.max():.2f}]"
        )

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

    print("\n[STEP 6] Transform white surfaces and save")

    def _transform_white(white_vertices):
        v = white_vertices.T
        v_h = np.vstack([v, np.ones((1, v.shape[1]), dtype=float)])
        v_t = trans_M @ v_h
        return v_t[0:3, :].T

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
    parser.add_argument("--ID", required=True, help="Subject ID, e.g. D0007")
    parser.add_argument(
        "--root-dir",
        required=True,
        help="Root dir for outputs and transform (contains <ID>/ and <ID>/<ID>/surf).",
    )
    parser.add_argument(
        "--transform-file",
        default=None,
        help="Optional transform .dat filename inside <root-dir>/<ID>. "
             "If omitted, auto-detect based on common names.",
    )
    parser.add_argument(
        "--voldim",
        nargs=3,
        type=int,
        default=[512, 512, 272],
        help="Fallback volume dimensions (Nx Ny Nz) if mov NIfTI not found.",
    )
    parser.add_argument(
        "--voxres",
        nargs=3,
        type=float,
        default=[0.5, 0.5, 0.5],
        help="Fallback voxel resolution (dx dy dz) if mov NIfTI not found.",
    )

    args = parser.parse_args()
    vertices_connect(
        ID=args.ID,
        root_dir=args.root_dir,
        transform_file=args.transform_file,
        voldim=args.voldim,
        voxres=args.voxres,
    )


if __name__ == "__main__":
    _cli()
