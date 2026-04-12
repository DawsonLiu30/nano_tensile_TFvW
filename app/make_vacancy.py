from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.io import read, write


def _ensure_nonempty(name: str, arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=int).ravel()
    if arr.size == 0:
        raise RuntimeError(f"{name} is empty. Check how you generated {name}.")
    return arr


def _ensure_balanced_grips(bottom_idx: np.ndarray, top_idx: np.ndarray, *, stage: str) -> None:
    if bottom_idx.size != top_idx.size:
        raise RuntimeError(
            f"[make_vacancy] {stage}: grip atom counts must stay balanced, "
            f"got bottom={bottom_idx.size}, top={top_idx.size}."
        )


def _remap_indices_after_removals(idx: np.ndarray, removed_idx: np.ndarray, n_old: int) -> np.ndarray:
    idx = np.asarray(idx, dtype=int).ravel()
    removed_idx = np.unique(np.asarray(removed_idx, dtype=int).ravel())
    keep_mask = np.ones(int(n_old), dtype=bool)
    keep_mask[removed_idx] = False

    old_to_new = np.full(int(n_old), -1, dtype=int)
    old_to_new[np.where(keep_mask)[0]] = np.arange(int(keep_mask.sum()), dtype=int)

    remapped = old_to_new[idx]
    remapped = remapped[remapped >= 0]
    return np.unique(remapped)


def _compute_n_vacancies_from_concentration(
    concentration_pct: float,
    basis_count: int,
) -> int:
    if concentration_pct < 0:
        raise ValueError(f"concentration_pct must be >= 0, got {concentration_pct}")
    if basis_count <= 0:
        return 0

    target = (float(concentration_pct) / 100.0) * float(basis_count)
    n = int(np.rint(target))
    if concentration_pct > 0 and n == 0:
        n = 1
    return n


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Create vacancies only inside the tensile free region and remap grip indices."
        )
    )
    ap.add_argument("--input", default="test_structure.xyz", help="Input structure")
    ap.add_argument("--bottom", default="bottom_idx.npy", help="Bottom grip index file")
    ap.add_argument("--top", default="top_idx.npy", help="Top grip index file")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument("--tag", default="vac1", help="Output tag -> writes <tag>.xyz/.vasp and *_<tag>.npy")

    ap.add_argument(
        "--mode",
        choices=["one", "count", "conc"],
        default="one",
        help="Vacancy mode: one (1 atom), count (--n), conc (--conc-pct).",
    )
    ap.add_argument("--n", type=int, default=1, help="Number of vacancies when --mode count")
    ap.add_argument(
        "--conc-pct",
        type=float,
        default=0.0,
        help="Vacancy concentration in percent when --mode conc, e.g. 0.1 for 0.1%%",
    )
    ap.add_argument(
        "--conc-basis",
        choices=["total", "free"],
        default="free",
        help="Denominator for concentration: total atoms or free-region atoms",
    )
    ap.add_argument(
        "--region",
        choices=["free"],
        default="free",
        help="Vacancies are restricted to the tensile free region.",
    )
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

    atoms = read(str(inp))
    n0 = len(atoms)
    if n0 <= 0:
        raise RuntimeError("[make_vacancy] Empty structure: no atoms.")

    bottom_idx = _ensure_nonempty("bottom_idx", np.load(str(bottom_path)))
    top_idx = _ensure_nonempty("top_idx", np.load(str(top_path)))
    _ensure_balanced_grips(bottom_idx, top_idx, stage="before vacancy")

    if int(bottom_idx.max()) >= n0 or int(top_idx.max()) >= n0:
        raise RuntimeError(
            "[make_vacancy] Grip indices out of range for the input structure.\n"
            f"  n_atoms={n0}\n"
            f"  bottom_idx max={bottom_idx.max()}\n"
            f"  top_idx    max={top_idx.max()}\n"
            "=> You are likely mixing idx from a different structure."
        )

    fixed_idx = np.unique(np.concatenate([bottom_idx, top_idx])).astype(int)
    all_idx = np.arange(n0, dtype=int)
    free_idx = np.setdiff1d(all_idx, fixed_idx)

    if free_idx.size == 0:
        raise RuntimeError("[make_vacancy] No free atoms available. Check the grips.")

    if args.mode == "one":
        n_vac = 1
        mode_note = "one"
    elif args.mode == "count":
        n_vac = int(args.n)
        mode_note = f"count(n={n_vac})"
    else:
        basis_count = n0 if args.conc_basis == "total" else free_idx.size
        n_vac = _compute_n_vacancies_from_concentration(args.conc_pct, basis_count)
        mode_note = (
            f"conc({float(args.conc_pct):.6g}%, basis={args.conc_basis}, "
            f"basis_count={basis_count})"
        )

    if n_vac < 0:
        raise RuntimeError(f"[make_vacancy] n_vac must be >= 0, got {n_vac}")

    candidate_idx = free_idx
    if n_vac > candidate_idx.size:
        print(
            f"[Vacancy] Warning: requested {n_vac} vacancies but only "
            f"{candidate_idx.size} free atoms are available -> clamp to {candidate_idx.size}."
        )
        n_vac = int(candidate_idx.size)

    if n_vac == 0:
        print("[Vacancy] n_vac=0 -> no atoms removed. Writing pass-through outputs.")
        remove_idx = np.array([], dtype=int)
    else:
        rng = np.random.default_rng(int(args.seed))
        remove_idx = np.sort(rng.choice(candidate_idx, size=n_vac, replace=False).astype(int))

    overlap_with_fixed = np.intersect1d(remove_idx, fixed_idx)
    if overlap_with_fixed.size > 0:
        raise RuntimeError(
            "[make_vacancy] Vacancy selection touched fixed atoms, which is forbidden.\n"
            f"  offending indices: {overlap_with_fixed.tolist()}"
        )

    mask = np.ones(n0, dtype=bool)
    mask[remove_idx] = False
    atoms_vac = atoms[mask]

    bottom_idx_new = _remap_indices_after_removals(bottom_idx, remove_idx, n0)
    top_idx_new = _remap_indices_after_removals(top_idx, remove_idx, n0)
    _ensure_balanced_grips(bottom_idx_new, top_idx_new, stage="after vacancy")

    n1 = len(atoms_vac)
    if bottom_idx_new.size == 0 or top_idx_new.size == 0:
        raise RuntimeError(
            "[make_vacancy] After remapping, bottom/top became empty. "
            "This should not happen if vacancies stay in the free region."
        )
    if int(bottom_idx_new.max()) >= n1 or int(top_idx_new.max()) >= n1:
        raise RuntimeError(
            "[make_vacancy] Remapped indices out of range.\n"
            f"  new n_atoms={n1}\n"
            f"  bottom_new max={bottom_idx_new.max()}\n"
            f"  top_new    max={top_idx_new.max()}"
        )

    tag = str(args.tag)
    xyz_out = Path(f"{tag}.xyz")
    vasp_out = Path(f"{tag}.vasp")
    bottom_out = Path(f"bottom_idx_{tag}.npy")
    top_out = Path(f"top_idx_{tag}.npy")

    write(str(xyz_out), atoms_vac)
    write(str(vasp_out), atoms_vac, vasp5=True, direct=True)
    np.save(str(bottom_out), bottom_idx_new)
    np.save(str(top_out), top_idx_new)

    print(f"[Vacancy] input        : {inp.name}")
    print(f"[Vacancy] total atoms  : {n0}")
    print(f"[Vacancy] fixed atoms  : {fixed_idx.size}  free atoms : {free_idx.size}")
    print(f"[Vacancy] mode         : {mode_note}")
    print("[Vacancy] remove region: free")
    print(f"[Vacancy] n_vacancies  : {n_vac}")
    if remove_idx.size:
        print(
            f"[Vacancy] removed idx  : {remove_idx[:10].tolist()}"
            f"{' ...' if remove_idx.size > 10 else ''}"
        )
    print(f"[Vacancy] after vacancy: {n1}")
    print(
        f"[Vacancy] grip atoms   : bottom={bottom_idx_new.size}, "
        f"top={top_idx_new.size} (balanced={bottom_idx_new.size == top_idx_new.size})"
    )
    print(f"[Vacancy] Written      : {xyz_out.name}, {vasp_out.name}")
    print(f"[Vacancy] Updated idx  : {bottom_out.name}, {top_out.name}")
    print("[Vacancy] NOTE: main.py must use the updated idx files for this vacancy structure.")


if __name__ == "__main__":
    main()
