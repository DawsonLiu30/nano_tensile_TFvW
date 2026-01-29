import argparse
from pathlib import Path

import numpy as np
from ase.io import read, write


def _ensure_nonempty(name: str, arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=int).ravel()
    if arr.size == 0:
        raise RuntimeError(f"{name} is empty. Check how you generated {name}.")
    return arr


def _remap_indices_after_removal(idx: np.ndarray, removed: int) -> np.ndarray:
    """
    After removing atom with index `removed`, ASE will reindex atoms:
      - removed index disappears
      - all indices > removed shift left by 1
      - indices < removed unchanged
    """
    idx = np.asarray(idx, dtype=int).ravel()

    # Drop the removed one if it was inside idx (should not happen if we remove from free region,
    # but this makes it robust)
    idx = idx[idx != removed]

    # Shift indices larger than removed
    idx = np.where(idx > removed, idx - 1, idx)

    # Keep unique + sorted (nice for debugging)
    idx = np.unique(idx)
    return idx


def parse_args():
    ap = argparse.ArgumentParser(description="Create ONE vacancy ONLY in the free (non-grip) region, and remap grip indices.")
    ap.add_argument("--input", default="test_structure.xyz", help="Input structure (default: test_structure.xyz)")
    ap.add_argument("--bottom", default="bottom_idx.npy", help="Bottom grip index file (default: bottom_idx.npy)")
    ap.add_argument("--top", default="top_idx.npy", help="Top grip index file (default: top_idx.npy)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    ap.add_argument("--tag", default="vac1", help="Output tag (default: vac1) -> writes vac1.xyz/vac1.vasp and *_vac1.npy")
    return ap.parse_args()


def main():
    args = parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise FileNotFoundError(f"[make_vacancy] Input not found: {inp.resolve()}")

    bottom_path = Path(args.bottom)
    top_path = Path(args.top)
    if not bottom_path.exists() or not top_path.exists():
        raise FileNotFoundError(
            f"[make_vacancy] Missing grip idx files.\n"
            f"  bottom: {bottom_path.resolve()}\n"
            f"  top   : {top_path.resolve()}"
        )

    # 1) Read structure
    atoms = read(str(inp))
    n0 = len(atoms)
    if n0 <= 0:
        raise RuntimeError("[make_vacancy] Empty structure: no atoms.")

    # 2) Read grips (these indices are defined on THIS input structure)
    bottom_idx = _ensure_nonempty("bottom_idx", np.load(str(bottom_path)))
    top_idx = _ensure_nonempty("top_idx", np.load(str(top_path)))

    # Defensive check: indices must be within range
    if bottom_idx.max() >= n0 or top_idx.max() >= n0:
        raise RuntimeError(
            "[make_vacancy] Grip indices out of range for the input structure.\n"
            f"  n_atoms={n0}\n"
            f"  bottom_idx max={bottom_idx.max()}\n"
            f"  top_idx    max={top_idx.max()}\n"
            "=> You are likely mixing idx from a different structure."
        )

    fixed_idx = np.unique(np.concatenate([bottom_idx, top_idx])).astype(int)

    # 3) Free region = all - fixed
    all_idx = np.arange(n0, dtype=int)
    free_idx = np.setdiff1d(all_idx, fixed_idx)

    if free_idx.size == 0:
        raise RuntimeError("[make_vacancy] No free atoms available (free_idx is empty). Check grips.")

    # 4) Remove EXACTLY ONE atom from free region
    rng = np.random.default_rng(int(args.seed))
    remove_i = int(rng.choice(free_idx, size=1, replace=False)[0])

    mask = np.ones(n0, dtype=bool)
    mask[remove_i] = False
    atoms_vac = atoms[mask]

    # 5) Remap grip indices so they match the NEW structure (n0-1 atoms)
    bottom_idx_new = _remap_indices_after_removal(bottom_idx, remove_i)
    top_idx_new = _remap_indices_after_removal(top_idx, remove_i)

    # Defensive check: remapped indices must be within new range
    n1 = len(atoms_vac)
    if bottom_idx_new.size == 0 or top_idx_new.size == 0:
        raise RuntimeError(
            "[make_vacancy] After remapping, bottom/top became empty. "
            "This should not happen unless grips were too small or something went very wrong."
        )
    if bottom_idx_new.max() >= n1 or top_idx_new.max() >= n1:
        raise RuntimeError(
            "[make_vacancy] Remapped indices out of range. Something is inconsistent.\n"
            f"  new n_atoms={n1}\n"
            f"  bottom_new max={bottom_idx_new.max()}\n"
            f"  top_new    max={top_idx_new.max()}"
        )

    # 6) Write outputs
    tag = str(args.tag)

    xyz_out = Path(f"{tag}.xyz")
    vasp_out = Path(f"{tag}.vasp")
    bottom_out = Path(f"bottom_idx_{tag}.npy")
    top_out = Path(f"top_idx_{tag}.npy")

    write(str(xyz_out), atoms_vac)
    write(str(vasp_out), atoms_vac, vasp5=True, direct=True)

    np.save("bottom_idx.npy", bottom_idx_new)
    np.save("top_idx.npy", top_idx_new)

    # 7) Print summary (human-friendly)
    print(f"[Vacancy] input        : {inp.name}")
    print(f"[Vacancy] total atoms  : {n0}")
    print(f"[Vacancy] fixed atoms  : {fixed_idx.size}  free atoms : {free_idx.size}")
    print(f"[Vacancy] removed index: {remove_i}  (removed from FREE region)")
    print(f"[Vacancy] after vacancy: {n1}")
    print(f"[Vacancy] Written      : {xyz_out.name}, {vasp_out.name}")
    print(f"[Vacancy] Updated idx  : {bottom_out.name}, {top_out.name}")
    print("[Vacancy] NOTE: main.py must use the UPDATED idx files for this vac structure.")


if __name__ == "__main__":
    main()
