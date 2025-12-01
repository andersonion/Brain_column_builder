#!/usr/bin/env python3
import numpy as np
from pathlib import Path


def _find_first_4x4_block(dat_path: Path):
    """
    Scan file for the first 4 consecutive lines each containing >=4 floats.
    If found, return a 4×4 array; else return None.
    """
    lines = dat_path.read_text().splitlines()
    n = len(lines)

    for start in range(n - 3):
        block = lines[start:start+4]
        rows = []
        ok = True
        for line in block:
            # normalize spacing + remove commas
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
            return np.array(rows, dtype=float)

    return None


def _fallback_load_matrix(dat_path: Path):
    """
    Old-style scraping of any floats in the entire file.
    Then taking either:
        - floats[3:19] if length ≥19 (matching MATLAB data.data(4:19))
        - floats[0:16] if length ≥16
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
            f"Could not extract 16 numbers. Found only {floats.size} tokens."
        )

    M = vals.reshape((4, 4))
    return M


def load_trans_M(dat_path: Path):
    """
    Try structured 4×4 block first; fall back to scraping.
    ALWAYS returns the matrix transposed, because MATLAB code expects transposed.
    """
    block = _find_first_4x4_block(dat_path)
    if block is not None:
        print("[PARSE] Found 4×4 block in file:")
        print(block)
        return block.T

    print("[PARSE] No clean 4×4 block found — falling back to token scrape.")
    M = _fallback_load_matrix(dat_path)
    print("[PARSE] Scraped matrix:")
    print(M)
    return M.T


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dat",
        required=True,
        help="Path to .dat transform file (e.g., DWI2T1_dti.dat)"
    )
    args = ap.parse_args()

    dat = Path(args.dat)
    if not dat.is_file():
        raise FileNotFoundError(dat)

    print(f"[INFO] Reading: {dat}\n")
    trans_M = load_trans_M(dat)

    print("\n[RESULT] Final trans_M (returned 4×4, after transpose):")
    print(trans_M)


if __name__ == "__main__":
    main()
