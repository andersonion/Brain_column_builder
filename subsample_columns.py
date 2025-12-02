#!/usr/bin/env python3
"""
Subsample cortical columns (vertices) per region using farthest-point sampling.

Modes:

1) ref_single:
   Build k_r table from a single reference subject.

   Example:
   python subsample_columns.py \
       --mode ref_single \
       --surf $SUBJECTS_DIR/fsaverage/surf/lh.white \
       --annot $SUBJECTS_DIR/fsaverage/label/lh.aparc.annot \
       --k-table lh_aparc_k_table_single.json \
       --frac 0.05 \
       --k-min 2 \
       --k-max 8

2) ref_multi:
   Build k_r table from the mean per-region vertex count across multiple
   reference subjects (recommended for robustness).

   Example:
   python subsample_columns.py \
       --mode ref_multi \
       --surf-list subj1_lh.white,subj2_lh.white,subj3_lh.white,subj4_lh.white,subj5_lh.white \
       --annot-list subj1_lh.aparc.annot,subj2_lh.aparc.annot,subj3_lh.aparc.annot,subj4_lh.aparc.annot,subj5_lh.aparc.annot \
       --k-table lh_aparc_k_table_multi.json \
       --frac 0.05 \
       --k-min 2 \
       --k-max 8

3) apply:
   Apply an existing k_r table to subsample a subject.

   Example:
   python subsample_columns.py \
       --mode apply \
       --surf $SUBJECTS_DIR/subj01/surf/lh.white \
       --annot $SUBJECTS_DIR/subj01/label/lh.aparc.annot \
       --k-table lh_aparc_k_table_multi.json \
       --out-indices subj01_lh_subsample_idx.npy \
       --seed 42

Outputs:
- ref_single / ref_multi: JSON file with k_table and counts.
- apply: NPY file with selected vertex indices to keep.
"""

import argparse
import json
import os
import sys

import numpy as np

try:
    import nibabel as nib
except ImportError:
    print("ERROR: This script requires nibabel (pip install nibabel).", file=sys.stderr)
    sys.exit(1)


def farthest_point_sampling(coords, k, seed=None):
    """
    Quasi-uniformly sample k points from coords using farthest-point sampling.

    Parameters
    ----------
    coords : (N, 3) array
        XYZ coordinates for N points (for example, vertex centers).
    k : int
        Number of points to sample.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    selected_idx : (k,) array of int
        Indices into coords of the selected points.
    """
    coords = np.asarray(coords)
    N = coords.shape[0]
    if k >= N:
        return np.arange(N, dtype=int)

    rng = np.random.default_rng(seed)

    # Start with a random index
    first = rng.integers(0, N)
    selected = [first]

    # Initialize distances to that first point
    diff = coords - coords[first]
    min_dist_sq = np.einsum("ij,ij->i", diff, diff)  # squared Euclidean

    for _ in range(1, k):
        # Farthest point from the current selected set
        next_idx = int(np.argmax(min_dist_sq))
        selected.append(next_idx)

        # Update distances with the new point
        diff = coords - coords[next_idx]
        dist_sq_new = np.einsum("ij,ij->i", diff, diff)
        min_dist_sq = np.minimum(min_dist_sq, dist_sq_new)

    return np.array(selected, dtype=int)


def load_surf_and_annot(surf_path, annot_path):
    """
    Load FreeSurfer surface coordinates and annotation labels.

    Returns
    -------
    coords : (N, 3) float32
    labels : (N,) int
    ctab   : (L, 5) int (color table)
    names  : list of label names (bytes)
    """
    coords, faces = nib.freesurfer.read_geometry(surf_path)
    labels, ctab, names = nib.freesurfer.read_annot(annot_path)
    return coords.astype(np.float32), labels.astype(int), ctab, names


def build_k_table_from_single(labels, frac=0.05, k_min=2, k_max=8):
    """
    Given region labels on a single reference brain, compute k_r per region.

    Parameters
    ----------
    labels : (N,) int
        Region labels per vertex.
    frac : float
        Fraction of vertices per region to keep before clamping.
    k_min : int
        Minimum columns to keep per region.
    k_max : int
        Maximum columns to keep per region.

    Returns
    -------
    k_table : dict
        Mapping label_int (as string) -> k_r (int).
    counts : dict
        Mapping label_int (as string) -> N_r (int).
    """
    labels = np.asarray(labels)
    unique_labels, counts_raw = np.unique(labels, return_counts=True)

    k_table = {}
    counts = {}

    for lab, n in zip(unique_labels, counts_raw):
        if lab == -1:
            # Often used for unknown / medial wall
            continue

        raw_k = frac * n
        k = int(round(raw_k))
        k = max(k_min, min(k_max, k))
        k_table[str(int(lab))] = k
        counts[str(int(lab))] = int(n)

    return k_table, counts


def build_k_table_from_multi(label_list, frac=0.05, k_min=2, k_max=8):
    """
    Given region labels from multiple reference subjects, compute k_r per region
    from the mean per-region vertex count.

    Parameters
    ----------
    label_list : list of (N_s,) arrays
        List of label arrays, one per reference subject.
    frac : float
        Fraction of average vertices per region to keep before clamping.
    k_min : int
        Minimum columns to keep per region.
    k_max : int
        Maximum columns to keep per region.

    Returns
    -------
    k_table : dict
        Mapping label_int (as string) -> k_r (int), where k_r is based on N̄_r.
    counts_mean : dict
        Mapping label_int (as string) -> mean N_r across reference subjects.
    counts_per_subject : list of dict
        One dict per subject: label_int (as string) -> N_r_s.
    """
    # Count per region per subject
    counts_per_subject = []
    all_labels = set()

    for labels in label_list:
        labels = np.asarray(labels)
        unique_labels, counts_raw = np.unique(labels, return_counts=True)
        d = {}
        for lab, n in zip(unique_labels, counts_raw):
            if lab == -1:
                continue
            lab_str = str(int(lab))
            d[lab_str] = int(n)
            all_labels.add(lab_str)
        counts_per_subject.append(d)

    all_labels = sorted(all_labels, key=lambda x: int(x))

    # Compute mean N_r across subjects for each label
    counts_mean = {}
    for lab_str in all_labels:
        vals = []
        for d in counts_per_subject:
            if lab_str in d:
                vals.append(d[lab_str])
        if not vals:
            continue
        counts_mean[lab_str] = float(np.mean(vals))

    # Now derive k_r from mean counts
    k_table = {}
    for lab_str, mean_n in counts_mean.items():
        raw_k = frac * mean_n
        k = int(round(raw_k))
        k = max(k_min, min(k_max, k))
        k_table[lab_str] = k

    return k_table, counts_mean, counts_per_subject


def subsample_subject(coords, labels, k_table, seed=0):
    """
    Subsample vertices per region for a subject using k_table.

    Parameters
    ----------
    coords : (N, 3) array
        Surface coordinates.
    labels : (N,) array
        Region labels per vertex.
    k_table : dict
        Mapping label_int (string) -> k_r (int).
    seed : int
        RNG seed base.

    Returns
    -------
    keep_idx : 1D array of int
        Global vertex indices to keep for this subject.
    """
    coords = np.asarray(coords)
    labels = np.asarray(labels)

    all_keep = []
    rng = np.random.default_rng(seed)

    unique_labels = np.unique(labels)

    for lab in unique_labels:
        if lab == -1:
            # Skip unknown / medial wall etc.
            continue

        key = str(int(lab))
        if key not in k_table:
            # Region not present in table, skip it.
            continue

        k = int(k_table[key])

        roi_mask = labels == lab
        roi_idx = np.where(roi_mask)[0]
        n_roi = roi_idx.size

        if n_roi == 0:
            continue

        if k > n_roi:
            # If region is smaller in this subject, keep all vertices
            k_eff = n_roi
        else:
            k_eff = k

        roi_coords = coords[roi_idx]
        roi_seed = int(rng.integers(0, 1_000_000))

        keep_local = farthest_point_sampling(roi_coords, k_eff, seed=roi_seed)
        keep_global = roi_idx[keep_local]
        all_keep.append(keep_global)

    if not all_keep:
        return np.array([], dtype=int)

    keep_idx = np.concatenate(all_keep)
    keep_idx = np.unique(keep_idx)
    return keep_idx


def parse_comma_list(s):
    """
    Split a comma-separated string into a list, handling trivial edge cases.
    """
    if s is None:
        return []
    parts = [x.strip() for x in s.split(",") if x.strip() != ""]
    return parts


def main():
    parser = argparse.ArgumentParser(description="Subsample cortical columns per region.")
    parser.add_argument(
        "--mode",
        choices=["ref_single", "ref_multi", "apply"],
        required=True,
        help="ref_single: single reference subject; "
             "ref_multi: multiple reference subjects; "
             "apply: subsample a subject using existing k_table."
    )

    # Common args
    parser.add_argument("--surf", help="Path to FreeSurfer surface (for ref_single or apply).")
    parser.add_argument("--annot", help="Path to FreeSurfer annot (for ref_single or apply).")
    parser.add_argument("--k-table", required=True,
                        help="Path to JSON file for k_table "
                             "(output in ref_* modes, input in apply mode).")

    # Multi-ref args
    parser.add_argument("--surf-list",
                        help="Comma-separated list of surfaces for ref_multi mode.")
    parser.add_argument("--annot-list",
                        help="Comma-separated list of annots for ref_multi mode.")

    # Apply output
    parser.add_argument("--out-indices", default=None,
                        help="Path to save npy array of selected vertex indices (apply mode only).")

    # Parameters
    parser.add_argument("--frac", type=float, default=0.05,
                        help="Fraction of (mean) vertices per region (ref_* modes). Default 0.05.")
    parser.add_argument("--k-min", type=int, default=2,
                        help="Minimum columns per region (ref_* modes).")
    parser.add_argument("--k-max", type=int, default=8,
                        help="Maximum columns per region (ref_* modes).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed base for FPS (apply mode).")

    args = parser.parse_args()

    if args.mode == "ref_single":
        if args.surf is None or args.annot is None:
            print("ERROR: --surf and --annot are required for ref_single mode.", file=sys.stderr)
            sys.exit(1)

        coords, labels, ctab, names = load_surf_and_annot(args.surf, args.annot)
        k_table, counts = build_k_table_from_single(
            labels,
            frac=args.frac,
            k_min=args.k_min,
            k_max=args.k_max,
        )

        out = {
            "mode": "ref_single",
            "k_table": k_table,
            "counts": counts,
            "params": {
                "frac": args.frac,
                "k_min": args.k_min,
                "k_max": args.k_max,
            },
        }

        with open(args.k_table, "w") as f:
            json.dump(out, f, indent=2, sort_keys=True)

        print(f"Saved k_table to {args.k_table}")
        print("Example entries (label_int: k_r):")
        for i, (lab, k) in enumerate(sorted(k_table.items(), key=lambda x: int(x[0]))):
            if i >= 10:
                break
            print(f"  {lab}: {k}")

    elif args.mode == "ref_multi":
        surf_list = parse_comma_list(args.surf_list)
        annot_list = parse_comma_list(args.annot_list)

        if not surf_list or not annot_list:
            print("ERROR: --surf-list and --annot-list are required for ref_multi mode.", file=sys.stderr)
            sys.exit(1)

        if len(surf_list) != len(annot_list):
            print("ERROR: --surf-list and --annot-list must have the same number of entries.", file=sys.stderr)
            sys.exit(1)

        label_list = []
        for surf_path, annot_path in zip(surf_list, annot_list):
            if not os.path.exists(surf_path):
                print(f"ERROR: surf path does not exist: {surf_path}", file=sys.stderr)
                sys.exit(1)
            if not os.path.exists(annot_path):
                print(f"ERROR: annot path does not exist: {annot_path}", file=sys.stderr)
                sys.exit(1)
            coords, labels, ctab, names = load_surf_and_annot(surf_path, annot_path)
            label_list.append(labels)

        k_table, counts_mean, counts_per_subject = build_k_table_from_multi(
            label_list,
            frac=args.frac,
            k_min=args.k_min,
            k_max=args.k_max,
        )

        out = {
            "mode": "ref_multi",
            "k_table": k_table,
            "counts_mean": counts_mean,
            "counts_per_subject": counts_per_subject,
            "params": {
                "frac": args.frac,
                "k_min": args.k_min,
                "k_max": args.k_max,
                "n_ref": len(label_list),
            },
        }

        with open(args.k_table, "w") as f:
            json.dump(out, f, indent=2, sort_keys=True)

        print(f"Saved multi-subject k_table to {args.k_table}")
        print(f"Number of reference subjects: {len(label_list)}")
        print("Example entries (label_int: k_r, mean_N_r):")
        for i, lab in enumerate(sorted(k_table.keys(), key=lambda x: int(x))):
            if i >= 10:
                break
            print(f"  {lab}: k_r={k_table[lab]}, mean_N_r={counts_mean[lab]:.1f}")

    elif args.mode == "apply":
        if args.surf is None or args.annot is None:
            print("ERROR: --surf and --annot are required for apply mode.", file=sys.stderr)
            sys.exit(1)

        if not os.path.exists(args.k_table):
            print(f"ERROR: k_table file does not exist: {args.k_table}", file=sys.stderr)
            sys.exit(1)

        with open(args.k_table, "r") as f:
            data = json.load(f)

        if "k_table" in data:
            k_table = data["k_table"]
        else:
            # Fallback if JSON is just a raw mapping
            k_table = data

        coords, labels, ctab, names = load_surf_and_annot(args.surf, args.annot)
        keep_idx = subsample_subject(coords, labels, k_table, seed=args.seed)

        if keep_idx.size == 0:
            print("WARNING: No vertices selected; check labels and k_table.", file=sys.stderr)

        if args.out_indices is None:
            base = os.path.splitext(os.path.basename(args.surf))[0]
            args.out_indices = base + "_subsample_idx.npy"

        np.save(args.out_indices, keep_idx)
        print(f"Saved {keep_idx.size} vertex indices to {args.out_indices}")

    else:
        print(f"Unknown mode {args.mode}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
