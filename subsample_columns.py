#!/usr/bin/env python3
"""
Subsample cortical columns (vertices) per region using Farthest-Point Sampling,
always treating left and right hemispheres together.

Key ideas:
- Build a single k_r per region NAME (e.g., "bankssts"), shared by LH and RH.
- Always process both hemispheres for every subject.
- Apply the same k_r for that region to LH and RH separately.

MODES
=====

1) ref_multi  (build k_r from multiple subjects)

Example:
    python subsample_columns.py \
        --mode ref_multi \
        --subjects subj01,subj02,subj03,subj04,subj05 \
        --subjects-dir /path/to/SUBJECTS_DIR \
        --k-table cortex_k_table.json \
        --frac 0.05 \
        --k-min 2 \
        --k-max 8

2) apply  (subsample one subject using an existing k_r table)

Example:
    python subsample_columns.py \
        --mode apply \
        --subject subj10 \
        --subjects-dir /path/to/SUBJECTS_DIR \
        --k-table cortex_k_table.json \
        --out-prefix subj10_subsample \
        --seed 42
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


###############################################################################
# Helpers
###############################################################################

def resolve_subject_base(subjects_dir, subj):
    """
    Return the actual base directory that contains surf/ and label/.
    Handles both:
        subjects_dir/subj/
        subjects_dir/subj/subj/
    """
    base1 = os.path.join(subjects_dir, subj)
    surf1 = os.path.join(base1, "surf")
    if os.path.isdir(surf1):
        return base1

    # Nested case: subjects_dir/subj/subj/surf
    base2 = os.path.join(subjects_dir, subj, subj)
    surf2 = os.path.join(base2, "surf")
    if os.path.isdir(surf2):
        return base2

    raise FileNotFoundError(
        f"Could not find surf/ for subject {subj}. Tried:\n"
        f"  {surf1}\n"
        f"  {surf2}"
    )


def farthest_point_sampling(coords, k, seed=None):
    """
    Quasi-uniform sampling using farthest-point sampling.

    Parameters
    ----------
    coords : (N, 3) array
    k      : int, number of points to keep
    seed   : int or None
    """
    coords = np.asarray(coords)
    N = coords.shape[0]

    if k >= N:
        return np.arange(N, dtype=int)

    rng = np.random.default_rng(seed)
    first = rng.integers(0, N)
    selected = [first]

    diff = coords - coords[first]
    min_dist_sq = np.einsum("ij,ij->i", diff, diff)  # squared distances

    for _ in range(1, k):
        next_idx = int(np.argmax(min_dist_sq))
        selected.append(next_idx)

        diff = coords - coords[next_idx]
        dist_sq_new = np.einsum("ij,ij->i", diff, diff)
        min_dist_sq = np.minimum(min_dist_sq, dist_sq_new)

    return np.array(selected, dtype=int)


def load_fs_hemi(subjects_dir, subj, hemi, surf_name="white", annot_name="aparc.annot"):
    """
    Load surface coordinates and annotation for one hemisphere of one subject,
    handling possible nested subject directories.
    """
    base = resolve_subject_base(subjects_dir, subj)

    surf_path = os.path.join(base, "surf", f"{hemi}.{surf_name}")
    annot_path = os.path.join(base, "label", f"{hemi}.{annot_name}")

    if not os.path.exists(surf_path):
        raise FileNotFoundError(f"Missing surface file: {surf_path}")
    if not os.path.exists(annot_path):
        raise FileNotFoundError(f"Missing annot file: {annot_path}")

    coords, _ = nib.freesurfer.read_geometry(surf_path)
    labels, ctab, names = nib.freesurfer.read_annot(annot_path)

    return coords.astype(np.float32), labels.astype(int), ctab, names


def make_label_value_to_name(ctab, names):
    """
    Build a mapping from label value (int) to region name (str).
    """
    value_to_name = {}
    for i, nm in enumerate(names):
        if isinstance(nm, bytes):
            nm_str = nm.decode("utf-8")
        else:
            nm_str = str(nm)
        val = int(ctab[i, 4])  # label value in last column
        value_to_name[val] = nm_str
    return value_to_name


def count_regions_by_name(labels, value_to_name):
    """
    Count number of vertices per region NAME for a single hemi of a subject.

    Returns
    -------
    counts : dict
        region_name -> count
    """
    labels = np.asarray(labels)
    uniq, cnts = np.unique(labels, return_counts=True)
    out = {}

    for lab, n in zip(uniq, cnts):
        if lab == -1:
            continue
        name = value_to_name.get(int(lab), "")
        if name.lower() in ("unknown", "???"):
            continue
        out[name] = out.get(name, 0) + int(n)

    return out


###############################################################################
# Build k_r from multiple subjects, both hemis
###############################################################################

def build_k_table_multi(subjects, subjects_dir, frac, k_min, k_max,
                        surf_name="white", annot_name="aparc.annot"):
    """
    Build a single k_r per region NAME using multiple subjects and both hemispheres.

    For each subject:
        - Count vertices per region name in LH and RH separately.
        - Compute the mean per-hemisphere count for that subject
          (average of LH and RH counts for that region).
    Then:
        - Average these per-subject means across all subjects to get Nbar(region_name).
        - Compute k_r = round(frac * Nbar), clamp to [k_min, k_max].
    """
    per_subject_region_means = {}  # region_name -> list of subject-level means

    for subj in subjects:
        subj_counts = {}  # region_name -> list of [lh_count, rh_count, ...]

        for hemi in ("lh", "rh"):
            coords, labels, ctab, names = load_fs_hemi(
                subjects_dir, subj, hemi,
                surf_name=surf_name,
                annot_name=annot_name,
            )
            value_to_name = make_label_value_to_name(ctab, names)
            hemi_counts = count_regions_by_name(labels, value_to_name)

            for name, n in hemi_counts.items():
                if name not in subj_counts:
                    subj_counts[name] = []
                subj_counts[name].append(n)

        # Compute per-subject mean across hemis for each region
        for name, vals in subj_counts.items():
            subj_mean = float(np.mean(vals))
            if name not in per_subject_region_means:
                per_subject_region_means[name] = []
            per_subject_region_means[name].append(subj_mean)

    counts_mean = {}
    k_table = {}

    for name, subj_means in per_subject_region_means.items():
        Nbar = float(np.mean(subj_means))
        counts_mean[name] = Nbar
        raw_k = frac * Nbar
        k = int(round(raw_k))
        k = max(k_min, min(k_max, k))
        k_table[name] = k

    return k_table, counts_mean, per_subject_region_means


###############################################################################
# Apply k_r to one subject (both hemis)
###############################################################################

def subsample_hemi_by_region_name(coords, labels, ctab, names, k_table, seed_offset=0):
    """
    Subsample vertices for a single hemi of one subject, using k_r keyed by region NAME.

    Returns
    -------
    keep_idx : 1D array of vertex indices to keep for this hemi.
    """
    coords = np.asarray(coords)
    labels = np.asarray(labels)

    value_to_name = make_label_value_to_name(ctab, names)
    rng = np.random.default_rng(seed_offset)

    keep_lists = []

    uniq_labels = np.unique(labels)
    for lab in uniq_labels:
        if lab == -1:
            continue
        name = value_to_name.get(int(lab), "")
        if name.lower() in ("unknown", "???"):
            continue
        if name not in k_table:
            continue

        k = int(k_table[name])
        idx = np.where(labels == lab)[0]
        if idx.size == 0:
            continue

        if k >= idx.size:
            keep_lists.append(idx)
            continue

        roi_coords = coords[idx]
        roi_seed = int(rng.integers(0, 1_000_000))
        local_keep = farthest_point_sampling(roi_coords, k, seed=roi_seed)
        keep_lists.append(idx[local_keep])

    if not keep_lists:
        return np.array([], dtype=int)

    return np.unique(np.concatenate(keep_lists))


def apply_k_table_to_subject(subject, subjects_dir, k_table,
                             surf_name="white", annot_name="aparc.annot",
                             seed=0):
    """
    Apply k_table to both hemis of one subject.

    Returns
    -------
    keep_lh : 1D array of LH vertex indices to keep
    keep_rh : 1D array of RH vertex indices to keep
    """
    # Left hemisphere
    coords_lh, labels_lh, ctab_lh, names_lh = load_fs_hemi(
        subjects_dir, subject, "lh", surf_name=surf_name, annot_name=annot_name
    )
    keep_lh = subsample_hemi_by_region_name(
        coords_lh, labels_lh, ctab_lh, names_lh, k_table,
        seed_offset=seed * 2 + 0,
    )

    # Right hemisphere
    coords_rh, labels_rh, ctab_rh, names_rh = load_fs_hemi(
        subjects_dir, subject, "rh", surf_name=surf_name, annot_name=annot_name
    )
    keep_rh = subsample_hemi_by_region_name(
        coords_rh, labels_rh, ctab_rh, names_rh, k_table,
        seed_offset=seed * 2 + 1,
    )

    return keep_lh, keep_rh


###############################################################################
# Main
###############################################################################

def parse_subject_list(s):
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Subsample cortical columns per region, both hemispheres together."
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=["ref_multi", "apply"],
        help="ref_multi: build k_r from multiple subjects; apply: subsample one subject."
    )

    parser.add_argument("--subjects", help="Comma-separated subject IDs (ref_multi).")
    parser.add_argument("--subject", help="Single subject ID (apply).")
    parser.add_argument("--subjects-dir", required=True,
                        help="FreeSurfer SUBJECTS_DIR (top-level, even if nested).")

    parser.add_argument("--k-table", required=True,
                        help="Path to k_r JSON (output for ref_multi, input for apply).")

    parser.add_argument("--frac", type=float, default=0.05,
                        help="Fraction of mean per-hem vertices per region.")
    parser.add_argument("--k-min", type=int, default=2,
                        help="Minimum columns per region.")
    parser.add_argument("--k-max", type=int, default=8,
                        help="Maximum columns per region.")
    parser.add_argument("--surf-name", default="white",
                        help="Surface name (default: white).")
    parser.add_argument("--annot-name", default="aparc.annot",
                        help="Annotation name (default: aparc.annot).")

    parser.add_argument("--out-prefix", default=None,
                        help="Prefix for output NPY files in apply mode.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed for sampling.")

    args = parser.parse_args()

    if args.mode == "ref_multi":
        subjects = parse_subject_list(args.subjects)
        if not subjects:
            print("ERROR: --subjects is required for ref_multi mode.", file=sys.stderr)
            sys.exit(1)

        k_table, counts_mean, per_subject_region_means = build_k_table_multi(
            subjects=subjects,
            subjects_dir=args.subjects_dir,
            frac=args.frac,
            k_min=args.k_min,
            k_max=args.k_max,
            surf_name=args.surf_name,
            annot_name=args.annot_name,
        )

        out = {
            "mode": "ref_multi",
            "subjects": subjects,
            "params": {
                "frac": args.frac,
                "k_min": args.k_min,
                "k_max": args.k_max,
                "surf_name": args.surf_name,
                "annot_name": args.annot_name,
            },
            "counts_mean": counts_mean,                   # region_name -> mean per-hem count
            "per_subject_region_means": per_subject_region_means,
            "k_table": k_table,                           # region_name -> k_r
        }

        with open(args.k_table, "w") as f:
            json.dump(out, f, indent=2, sort_keys=True)

        print(f"Saved k_r table to {args.k_table}")
        print("Example k_r entries (region_name: k_r, mean_N):")
        for i, name in enumerate(sorted(k_table.keys())):
            if i >= 10:
                break
            print(f"  {name}: k_r={k_table[name]}, mean_N={counts_mean[name]:.1f}")

    elif args.mode == "apply":
        if not args.subject:
            print("ERROR: --subject is required for apply mode.", file=sys.stderr)
            sys.exit(1)

        if not os.path.exists(args.k_table):
            print(f"ERROR: k_table file does not exist: {args.k_table}", file=sys.stderr)
            sys.exit(1)

        with open(args.k_table, "r") as f:
            data = json.load(f)

        if "k_table" in data:
            k_table = data["k_table"]
        else:
            k_table = data

        keep_lh, keep_rh = apply_k_table_to_subject(
            subject=args.subject,
            subjects_dir=args.subjects_dir,
            k_table=k_table,
            surf_name=args.surf_name,
            annot_name=args.annot_name,
            seed=args.seed,
        )

        if args.out_prefix is None:
            args.out_prefix = f"{args.subject}_subsample"

        out_lh = f"{args.out_prefix}_lh_idx.npy"
        out_rh = f"{args.out_prefix}_rh_idx.npy"

        np.save(out_lh, keep_lh)
        np.save(out_rh, keep_rh)

        print(f"Saved LH indices ({keep_lh.size}) to {out_lh}")
        print(f"Saved RH indices ({keep_rh.size}) to {out_rh}")

    else:
        print(f"Unknown mode: {args.mode}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
