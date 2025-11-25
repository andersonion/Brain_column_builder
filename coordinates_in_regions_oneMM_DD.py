#!/usr/bin/env python

"""
Python translation of MATLAB function coordinates_in_regions_oneMM_DD.

Original MATLAB signature:
    coordinates_in_regions_oneMM_DD(ID, input_dir, output_dir)

This script:
- Loads FreeSurfer-style column coordinate matrices for lh/rh (DWI space).
- Loads FreeSurfer aparc annotations.
- For each region (i = 2:36, skipping 5), extracts coordinates for the
  vertices in that region, across 21 "depth" samples per vertex.
- Saves per-region lh/rh coordinates as both .mat and .csv.

Dependencies:
    numpy
    scipy (scipy.io.loadmat, scipy.io.savemat)
    nibabel (for reading .annot files)
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat
import nibabel.freesurfer as fsio


def _load_cp_dwi_mats(qsm_dir: Path, ID: str):
    """
    Load the *_column_* DWI coordinate matrices.

    MATLAB:
        load([input_dir ID '/QSM/' ID '_column_lh.mat']);
        load([input_dir ID '/QSM/' ID '_column_rh.mat']);
        load([input_dir ID '/QSM/' ID '_column_lh_dwi.mat']);
        load([input_dir ID '/QSM/' ID '_column_rh_dwi.mat']);

    In practice, we only need lh_cp_dwi and rh_cp_dwi (DWI space),
    but we mirror the original loads for completeness.
    """
    lh_mat = qsm_dir / f"{ID}_column_lh.mat"
    rh_mat = qsm_dir / f"{ID}_column_rh.mat"
    lh_dwi_mat = qsm_dir / f"{ID}_column_lh_dwi.mat"
    rh_dwi_mat = qsm_dir / f"{ID}_column_rh_dwi.mat"

    if not lh_dwi_mat.is_file():
        raise FileNotFoundError(f"Missing required file: {lh_dwi_mat}")
    if not rh_dwi_mat.is_file():
        raise FileNotFoundError(f"Missing required file: {rh_dwi_mat}")

    # These two are loaded in MATLAB; we do the same even if we don't use them directly.
    if lh_mat.is_file():
        _ = loadmat(lh_mat)
    if rh_mat.is_file():
        _ = loadmat(rh_mat)

    lh_dwi = loadmat(lh_dwi_mat)
    rh_dwi = loadmat(rh_dwi_mat)

    # Expect variables named exactly as in MATLAB:
    #   lh_cp_dwi, rh_cp_dwi
    try:
        ori_lh_cp_dwi = np.asarray(lh_dwi["lh_cp_dwi"])
    except KeyError as e:
        raise KeyError(f"{lh_dwi_mat} does not contain 'lh_cp_dwi'") from e

    try:
        ori_rh_cp_dwi = np.asarray(rh_dwi["rh_cp_dwi"])
    except KeyError as e:
        raise KeyError(f"{rh_dwi_mat} does not contain 'rh_cp_dwi'") from e

    return ori_lh_cp_dwi, ori_rh_cp_dwi


def _load_aparc_annot(subject_dir: Path, hemi: str):
    """
    Load FreeSurfer aparc annotation for one hemisphere.

    MATLAB equivalent:
        annot_file_lh = 'lh.aparc.annot';
        file_name_lh = [input_dir ID '/' ID '/label/' annot_file_lh];
        [~, label_lh, colortable_lh] = read_annotation(file_name_lh);
    """
    annot_fname = subject_dir / "label" / f"{hemi}.aparc.annot"

    if not annot_fname.is_file():
        raise FileNotFoundError(f"Annotation file not found: {annot_fname}")

    # nibabel.freesurfer.read_annot returns:
    #   labels: (N,) int array (per-vertex label ID)
    #   ctab:   (K, 5) array (RGBA + ID)
    #   names:  list/array of region names (bytes)
    labels, ctab, names = fsio.read_annot(str(annot_fname))

    struct_names = [
        n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else str(n)
        for n in names
    ]

    return labels, ctab, struct_names


def _region_indices_for_vertices(vertex_indices_0based: np.ndarray, depth_samples: int = 21):
    """
    Replicate the MATLAB logic:

        vertx_num = find(label == region_num);      % 1-based
        index = vertx_num .* 21 + (-20:1:0);        % m x 21
        index = reshape(index', 1, []);             % 1 x (m*21)

    where vertx_num is 1-based vertex indices, and we want final column indices
    (still 1-based in MATLAB). We return 1-based indices here to mirror MATLAB,
    and the caller can convert to 0-based for NumPy indexing.
    """
    # Convert 0-based vertex IDs (from np.where) to 1-based
    vertx_1b = vertex_indices_0based.astype(np.int64) + 1  # (m,)

    offsets = np.arange(-20, 1, dtype=np.int64)  # -20 .. 0 inclusive (21 values)

    indices = []
    for v in vertx_1b:
        base = v * depth_samples  # v*21
        indices.extend(base + offsets)

    return np.array(indices, dtype=np.int64)  # 1-based indices, as in MATLAB


def _process_hemi(
    hemi: str,
    labels: np.ndarray,
    ctab: np.ndarray,
    struct_names: list[str],
    ori_cp_dwi: np.ndarray,
    out_dir: Path,
):
    """
    Process one hemisphere across regions i = 2:36 (skip 5),
    mirroring the MATLAB loop.

    Writes:
        {hemi}_{region_name}.mat  (variable 'lh_cp_dwi' or 'rh_cp_dwi')
        {hemi}_{region_name}.csv
    """
    max_index = ori_cp_dwi.shape[1]  # number of columns

    for i in range(2, 37):  # MATLAB 2:36
        if i == 5:
            continue

        # MATLAB is 1-based; nibabel arrays are 0-based.
        # So MATLAB row i corresponds to Python index i-1.
        row_idx = i - 1

        if row_idx >= len(struct_names) or row_idx >= ctab.shape[0]:
            # If the color table has fewer entries than expected, skip extras.
            continue

        region_name = struct_names[row_idx]
        # nibabel ctab: RGBA + ID in column 4
        region_num = int(ctab[row_idx, 4])

        vertex_indices = np.where(labels == region_num)[0]  # 0-based
        if vertex_indices.size == 0:
            # No vertices in this region for this subject/hemisphere
            continue

        index_1b = _region_indices_for_vertices(vertex_indices, depth_samples=21)

        # Filter out-of-range indices (MATLAB: index(:, any(index > max_index, 1)) = [])
        valid_mask = index_1b <= max_index
        index_1b = index_1b[valid_mask]
        if index_1b.size == 0:
            continue

        idx_0b = index_1b - 1
        cp_dwi = ori_cp_dwi[:, idx_0b]

        # File naming: 'lh_{region_name}' or 'rh_{region_name}'
        output_name = f"{hemi}_{region_name}"
        mat_path = out_dir / f"{output_name}.mat"
        csv_path = out_dir / f"{output_name}.csv"

        # Variable name in MAT must match MATLAB expectations
        var_name = f"{hemi}_cp_dwi"
        savemat(mat_path, {var_name: cp_dwi})

        # Save CSV (mirrors MATLAB writematrix)
        np.savetxt(csv_path, cp_dwi, delimiter=",")


def coordinates_in_regions_oneMM_DD(ID: str, input_dir, output_dir):
    """
    Python version of coordinates_in_regions_oneMM_DD(ID, input_dir, output_dir).

    READS from:
        input_dir / ID / QSM
        input_dir / ID / ID / label

    WRITES to:
        output_dir / ID / QSM / label_coord_1mm
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    data_root = input_dir
    write_root = output_dir

    # Where to write per-region label coordinate outputs
    label_coord_dir = write_root / ID / "QSM" / "label_coord_1mm"
    label_coord_dir.mkdir(parents=True, exist_ok=True)

    # Where to read QSM columns from
    qsm_dir = data_root / ID / "QSM"

    col_lh_mat = qsm_dir / f"{ID}_column_lh.mat"
    if not col_lh_mat.is_file():
        print(f"Subject {ID} doesnt have columns result; looked in {col_lh_mat}")
        return

    ori_lh_cp_dwi, ori_rh_cp_dwi = _load_cp_dwi_mats(qsm_dir, ID)

    # Where to read annotations from: input_dir / ID / ID / label
    subject_dir = data_root / ID / ID
    labels_lh, ctab_lh, names_lh = _load_aparc_annot(subject_dir, "lh")
    labels_rh, ctab_rh, names_rh = _load_aparc_annot(subject_dir, "rh")

    _process_hemi(
        hemi="lh",
        labels=labels_lh,
        ctab=ctab_lh,
        struct_names=names_lh,
        ori_cp_dwi=ori_lh_cp_dwi,
        out_dir=label_coord_dir,
    )

    _process_hemi(
        hemi="rh",
        labels=labels_rh,
        ctab=ctab_rh,
        struct_names=names_rh,
        ori_cp_dwi=ori_rh_cp_dwi,
        out_dir=label_coord_dir,
    )


def _cli():
    """
    Command-line interface wrapper.

    Example:
        python coordinates_in_regions_oneMM_DD.py \\
            --ID S00775 \\
            --input-dir  /mnt/newStor/paros/paros_WORK/hanwen/ad_decode_test/output/ \\
            --output-dir /mnt/newStor/paros/paros_WORK/column_code_tester/
    """
    parser = argparse.ArgumentParser(
        description="Generate cortical column coordinates in different regions (1mm, DWI)."
    )
    parser.add_argument("--ID", required=True, help="Subject ID (e.g., S00775)")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Input directory root (contains ID/QSM and ID/ID/label).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory root (where ID/QSM/label_coord_1mm will be created).",
    )

    args = parser.parse_args()
    coordinates_in_regions_oneMM_DD(args.ID, args.input_dir, args.output_dir)


if __name__ == "__main__":
    _cli()
