#!/usr/bin/env python

import argparse
from pathlib import Path
import numpy as np
from scipy.io import loadmat, savemat
import nibabel.freesurfer as fsio


def _load_cp_dwi_mats(qsm_dir: Path, ID: str):
    lh_mat = qsm_dir / f"{ID}_column_lh.mat"
    rh_mat = qsm_dir / f"{ID}_column_rh.mat"
    lh_dwi_mat = qsm_dir / f"{ID}_column_lh_dwi.mat"
    rh_dwi_mat = qsm_dir / f"{ID}_column_rh_dwi.mat"

    print(f"\n[LOAD] Looking for:")
    print(f"  {lh_mat}")
    print(f"  {rh_mat}")
    print(f"  {lh_dwi_mat}")
    print(f"  {rh_dwi_mat}")

    if not lh_dwi_mat.is_file():
        raise FileNotFoundError(f"Missing required: {lh_dwi_mat}")
    if not rh_dwi_mat.is_file():
        raise FileNotFoundError(f"Missing required: {rh_dwi_mat}")

    if lh_mat.is_file():
        _ = loadmat(lh_mat)
    if rh_mat.is_file():
        _ = loadmat(rh_mat)

    lh_dwi = loadmat(lh_dwi_mat)
    rh_dwi = loadmat(rh_dwi_mat)

    ori_lh_cp_dwi = np.asarray(lh_dwi["lh_cp_dwi"])
    ori_rh_cp_dwi = np.asarray(rh_dwi["rh_cp_dwi"])

    print(f"[LOAD] LH cp_dwi shape: {ori_lh_cp_dwi.shape}")
    print(f"[LOAD] RH cp_dwi shape: {ori_rh_cp_dwi.shape}")

    return ori_lh_cp_dwi, ori_rh_cp_dwi


def _load_aparc_annot(subject_dir: Path, hemi: str):
    annot_fname = subject_dir / "label" / f"{hemi}.aparc.annot"
    print(f"\n[ANNOT] Loading annotation: {annot_fname}")

    if not annot_fname.is_file():
        raise FileNotFoundError(f"Missing annotation: {annot_fname}")

    labels, ctab, names = fsio.read_annot(str(annot_fname))

    struct_names = [
        n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else str(n)
        for n in names
    ]

    print(f"[ANNOT] {hemi}: {len(struct_names)} region names, labels shape: {labels.shape}")
    return labels, ctab, struct_names


def _region_indices_for_vertices(vertex_indices_0b: np.ndarray, depth_samples=21):
    vertx_1b = vertex_indices_0b.astype(np.int64) + 1
    offsets = np.arange(-20, 1, dtype=np.int64)
    indices = []

    for v in vertx_1b:
        base = v * depth_samples
        indices.extend(base + offsets)

    return np.array(indices, dtype=np.int64)


def _process_hemi(
    hemi: str,
    labels: np.ndarray,
    ctab: np.ndarray,
    struct_names: list[str],
    ori_cp_dwi: np.ndarray,
    out_dir: Path,
):

    max_index = ori_cp_dwi.shape[1]

    print(f"\n===== PROCESSING {hemi.upper()} HEMISPHERE =====")
    print(f"Total columns available: {max_index}")
    print(f"Total struct_names: {len(struct_names)}")

    n_written = 0
    n_regions_with_vertices = 0

    for i in range(2, 37):  # MATLAB 2:36
        if i == 5:
            print(f"[SKIP] MATLAB skip region i = 5")
            continue

        row_idx = i - 1  # 0-based alignment
        if row_idx >= len(struct_names) or row_idx >= ctab.shape[0]:
            print(f"[WARN] Row {row_idx} out of bounds for struct_names/ctab, skipping.")
            continue

        region_name = struct_names[row_idx]
        region_num = int(ctab[row_idx, 4])

        # Find vertices
        vertex_indices = np.where(labels == region_num)[0]
        n_vertices = len(vertex_indices)

        print(f"\n[REGION] i={i} (row_idx={row_idx})  name='{region_name}'  label={region_num}")
        print(f"         vertices={n_vertices}")

        if n_vertices == 0:
            print("         -> No vertices in this region, skipping.")
            continue

        n_regions_with_vertices += 1

        index_1b = _region_indices_for_vertices(vertex_indices)
        valid_mask = index_1b <= max_index
        index_1b = index_1b[valid_mask]

        print(f"         depth-sample indices kept={len(index_1b)}")

        if len(index_1b) == 0:
            print("         -> No valid depth samples, skipping.")
            continue

        idx_0b = index_1b - 1
        cp_dwi = ori_cp_dwi[:, idx_0b]

        output_name = f"{hemi}_{region_name}"
        mat_path = out_dir / f"{output_name}.mat"
        csv_path = out_dir / f"{output_name}.csv"

        savemat(mat_path, {f"{hemi}_cp_dwi": cp_dwi})
        np.savetxt(csv_path, cp_dwi, delimiter=",")

        print(f"         -> WROTE files: {mat_path.name}, {csv_path.name}")

        n_written += 1

    print(f"\n>>> SUMMARY {hemi.upper()} <<<")
    print(f"Regions with vertices: {n_regions_with_vertices}")
    print(f"Files written: {n_written}")


def coordinates_in_regions_oneMM_DD(ID: str, input_dir, output_dir):

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Paths
    qsm_dir = input_dir / ID / "QSM"
    subject_dir = input_dir / ID / ID

    out_dir = output_dir / ID / "QSM" / "label_coord_1mm"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[PATH] Reading from input_dir:  {input_dir}")
    print(f"[PATH] Writing to output_dir: {output_dir}")
    print(f"[PATH] Output folder will be: {out_dir}")

    # Column coordinate matrices
    ori_lh_cp_dwi, ori_rh_cp_dwi = _load_cp_dwi_mats(qsm_dir, ID)

    # Annotation files
    labels_lh, ctab_lh, names_lh = _load_aparc_annot(subject_dir, "lh")
    labels_rh, ctab_rh, names_rh = _load_aparc_annot(subject_dir, "rh")

    # Process hemispheres
    _process_hemi("lh", labels_lh, ctab_lh, names_lh, ori_lh_cp_dwi, out_dir)
    _process_hemi("rh", labels_rh, ctab_rh, names_rh, ori_rh_cp_dwi, out_dir)


def _cli():
    parser = argparse.ArgumentParser(description="Verbose QA cortical column region extraction.")
    parser.add_argument("--ID", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    coordinates_in_regions_oneMM_DD(args.ID, args.input_dir, args.output_dir)


if __name__ == "__main__":
    _cli()
