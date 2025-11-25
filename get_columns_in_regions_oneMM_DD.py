#!/usr/bin/env python

"""
Python translation of MATLAB function get_columns_in_regions_oneMM_DD.

Original MATLAB:

    get_columns_in_regions_oneMM_DD(ID, input_dir, output_dir)

Behavior:
- Loads a QSM image for the subject.
- For each cortical region in a fixed list, loads per-region column coordinates
  (lh_cp_dwi / rh_cp_dwi) in DWI/QSM voxel space.
- Samples the QSM image along each column (21 depth samples per column).
- Writes per-region LH and RH value matrices as CSV files.

Python usage example:

    python get_columns_in_regions_oneMM_DD.py \\
        --ID S00775 \\
        --input-dir  /mnt/newStor/paros/paros_WORK/hanwen/ad_decode_test/input/ \\
        --output-dir /mnt/newStor/paros/paros_WORK/column_code_tester/

Assumed layout:

    QSM image:
        <input_dir>/<ID>/<ID>_QSM_masked.nii.gz

    Column coordinate .mat files (previous step output):
        <output_dir>/<ID>/QSM/label_coord_1mm/lh_<region>.mat
        <output_dir>/<ID>/QSM/label_coord_1mm/rh_<region>.mat

    Output CSVs:
        <output_dir>/<ID>/QSM/<ID>_lh_<region>_cols_QSM.csv
        <output_dir>/<ID>/QSM/<ID>_rh_<region>_cols_QSM.csv
"""

import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.io import loadmat
from scipy.ndimage import map_coordinates


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


def _load_qsm_image(input_dir: Path, ID: str) -> np.ndarray:
    """
    Load the subject's QSM image as a 3D numpy array.

    MATLAB:
        image = .../input/ID/ID_QSM_masked.nii.gz
        dwi = MRIread(image);
        volumn = dwi.vol;

    Here:
        image path = input_dir / ID / f"{ID}_QSM_masked.nii.gz"
    """
    image_path = input_dir / ID / f"{ID}_QSM_masked.nii.gz"
    print(f"[QSM] Looking for QSM image at: {image_path}")

    if not image_path.is_file():
        raise FileNotFoundError(f"Subject {ID} doesnt have QSM image: {image_path}")

    img = nib.load(str(image_path))
    vol = img.get_fdata()

    # If 4D (e.g., time series), take first volume
    if vol.ndim == 4:
        vol = vol[..., 0]

    print(f"[QSM] Loaded QSM volume with shape: {vol.shape}")
    return vol


def _load_region_cp_dwi(mat_path: Path, hemi: str) -> np.ndarray:
    """
    Load per-region column coordinates from a .mat file.

    Each file is expected to contain:
        - lh_cp_dwi or rh_cp_dwi: shape (4, N), homogeneous coordinates [x; y; z; 1]

    We only use the first 3 rows for indexing into the QSM volume.
    """
    if not mat_path.is_file():
        raise FileNotFoundError(f"Missing label_coord file: {mat_path}")

    data = loadmat(mat_path)
    var_name = f"{hemi}_cp_dwi"

    if var_name not in data:
        raise KeyError(f"{mat_path} does not contain variable '{var_name}'")

    cp_dwi = np.asarray(data[var_name])
    if cp_dwi.shape[0] < 3:
        raise ValueError(f"{mat_path}: {var_name} has invalid shape {cp_dwi.shape}")

    print(f"[CP] Loaded {var_name} from {mat_path.name}, shape={cp_dwi.shape}")
    return cp_dwi


def _sample_columns_from_volume(
    vol: np.ndarray, cp_dwi: np.ndarray, points_num: int = 21
) -> np.ndarray:
    """
    Sample QSM volume at column coordinates.

    MATLAB logic:

        points_num = 21;
        columns_num = size(cp_dwi, 2) / 21;
        values = zeros(columns_num, 21);
        for i = 1:21
            index = i:21:(i + 21 * (columns_num-1));
            values(:, i) = interp3(volumn,
                                   cp_dwi(1, index),
                                   cp_dwi(2, index),
                                   cp_dwi(3, index));
        end

    Here we replicate that behavior using scipy.ndimage.map_coordinates
    with linear interpolation (order=1).

    IMPORTANT:
        - cp_dwi comes from MATLAB, so coordinates are 1-based.
        - numpy/map_coordinates expects 0-based indices.
        - We subtract 1.0 from x,y,z to convert to 0-based.
    """
    n_coords = cp_dwi.shape[1]
    if n_coords % points_num != 0:
        raise ValueError(
            f"cp_dwi has {n_coords} columns, not divisible by points_num={points_num}"
        )

    columns_num = n_coords // points_num
    print(f"[SAMPLE] columns_num={columns_num}, points_num={points_num}")

    # Output: (columns_num, points_num)
    values = np.zeros((columns_num, points_num), dtype=float)

    # For each depth index i = 1..21 (MATLAB), 0..20 in Python
    for depth_idx in range(points_num):
        # 0-based indices into cp_dwi columns
        index = depth_idx + np.arange(columns_num) * points_num  # shape (columns_num,)

        # Extract coordinates: cp_dwi[0:3, index] are [x; y; z]
        coords = cp_dwi[0:3, index].astype(float)

        # Convert 1-based MATLAB coords to 0-based numpy coords
        coords -= 1.0

        # map_coordinates expects coords in (ndim, N) with order (z,y,x) or (x,y,z)
        # Here we assume cp_dwi was built in the same axis order as the QSM volume
        sampled = map_coordinates(
            vol,
            coords,
            order=1,       # linear interpolation (like interp3 default)
            mode="nearest" # clamp at edges
        )

        values[:, depth_idx] = sampled

    return values


def get_columns_in_regions_oneMM_DD(ID: str, input_dir, output_dir):
    """
    Python implementation of get_columns_in_regions_oneMM_DD.

    Parameters
    ----------
    ID : str
        Subject ID (e.g. 'S00775').
    input_dir : str or Path
        Root directory that contains the QSM image:
            <input_dir>/<ID>/<ID>_QSM_masked.nii.gz
    output_dir : str or Path
        Root directory that contains label_coord_1mm and where QSM
        per-region CSV outputs will be written:
            <output_dir>/<ID>/QSM/label_coord_1mm
            <output_dir>/<ID>/QSM/<ID>_lh_<region>_cols_QSM.csv
            <output_dir>/<ID>/QSM/<ID>_rh_<region>_cols_QSM.csv
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    print(f"\n[INFO] Subject: {ID}")
    print(f"[INFO] Input dir (QSM):  {input_dir}")
    print(f"[INFO] Output dir (QSM+labels): {output_dir}")

    # Load QSM volume
    vol = _load_qsm_image(input_dir, ID)

    # Where label_coord_1mm files live
    label_coord_dir = output_dir / ID / "QSM" / "label_coord_1mm"
    if not label_coord_dir.is_dir():
        raise FileNotFoundError(f"Label coord dir not found: {label_coord_dir}")

    # Where to write region-wise QSM column values
    qsm_out_dir = output_dir / ID / "QSM"
    qsm_out_dir.mkdir(parents=True, exist_ok=True)

    points_num = 21  # must match vertices_connect.m / coordinates_in_regions_oneMM_DD.m

    for region_name in REGION_LIST:
        lh_mat_path = label_coord_dir / f"lh_{region_name}.mat"
        rh_mat_path = label_coord_dir / f"rh_{region_name}.mat"

        print(f"\n[REGION] {region_name}")
        if not lh_mat_path.is_file() or not rh_mat_path.is_file():
            print(f"  -> Missing label files for {region_name}, skipping.")
            continue

        # Load per-region coordinates
        lh_cp_dwi = _load_region_cp_dwi(lh_mat_path, hemi="lh")
        rh_cp_dwi = _load_region_cp_dwi(rh_mat_path, hemi="rh")

        # Sample QSM volume along columns
        lh_values = _sample_columns_from_volume(vol, lh_cp_dwi, points_num=points_num)
        rh_values = _sample_columns_from_volume(vol, rh_cp_dwi, points_num=points_num)

        # Write CSV outputs
        lh_csv_path = qsm_out_dir / f"{ID}_lh_{region_name}_cols_QSM.csv"
        rh_csv_path = qsm_out_dir / f"{ID}_rh_{region_name}_cols_QSM.csv"

        np.savetxt(lh_csv_path, lh_values, delimiter=",")
        np.savetxt(rh_csv_path, rh_values, delimiter=",")

        print(f"  -> WROTE {lh_csv_path.name} (shape {lh_values.shape})")
        print(f"  -> WROTE {rh_csv_path.name} (shape {rh_values.shape})")


def _cli():
    parser = argparse.ArgumentParser(
        description="Sample QSM along cortical columns in predefined regions."
    )
    parser.add_argument("--ID", required=True, help="Subject ID (e.g., S00775)")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Root for QSM input images (contains ID/ID_QSM_masked.nii.gz).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root for column label_coord_1mm and QSM per-region CSV outputs.",
    )

    args = parser.parse_args()
    get_columns_in_regions_oneMM_DD(args.ID, args.input_dir, args.output_dir)


if __name__ == "__main__":
    _cli()
