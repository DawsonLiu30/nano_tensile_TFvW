#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
from ase.io import read, write


# =========================
# Auto_prep-like helpers
# =========================

def _unique_levels(z: np.ndarray, eps: float):
    """Quantize z-coordinates into discrete levels (same as auto_prep)."""
    q = np.round(z / eps) * eps
    levels = np.unique(q)
    levels.sort()
    return q, levels


def _pick_levels(levels: np.ndarray, z_min: float, z_max: float,
                 target_thickness: float, min_layers: int, max_layers):
    """Pick bottom/top z-levels (same spirit as auto_prep)."""
    # Bottom
    bottom_levels = []
    for lv in levels:
        if lv <= z_min + target_thickness:
            bottom_levels.append(lv)
        else:
            break
    if len(bottom_levels) < min_layers:
        bottom_levels = list(levels[:min(min_layers, len(levels))])
    if max_layers is not None and max_layers > 0:
        bottom_levels = bottom_levels[:min(max_layers, len(bottom_levels))]

    # Top
    top_levels = []
    for lv in levels[::-1]:
        if lv >= z_max - target_thickness:
            top_levels.append(lv)
        else:
            break
    if len(top_levels) < min_layers:
        top_levels = list(levels[-min(min_layers, len(levels)):][::-1])
    if max_layers is not None and max_layers > 0:
        top_levels = top_levels[:min(max_layers, len(top_levels))]

    return np.array(bottom_levels, float), np.array(top_levels, float)


def _target_thickness(z_range: float, end_frac: float, min_thickness: float, max_thickness: float):
    raw = float(z_range * float(end_frac))
    used = float(np.clip(raw, float(min_thickness), float(max_thickness)))
    return used, raw


def get_grip_masks_like_autoprep(
    atoms,
    end_frac: float,
    min_thickness: float,
    max_thickness: float,
    min_layers: int,
    max_layers,     # None => unlimited
    eps: float,
):
    """Return bottom_mask, top_mask, middle_mask, plus some meta info."""
    pos = atoms.get_positions()
    z = pos[:, 2].astype(float)
    z_min = float(z.min())
    z_max = float(z.max())
    z_range = z_max - z_min
    if z_range <= 0:
        raise ValueError("Invalid structure: z_range <= 0")

    th, th_raw = _target_thickness(z_range, end_frac, min_thickness, max_thickness)

    zq, levels = _unique_levels(z, float(eps))
    if levels.size < 2:
        raise ValueError("Not enough distinct z-levels. Try larger eps.")

    bottom_levels, top_levels = _pick_levels(
        levels=levels,
        z_min=z_min, z_max=z_max,
        target_thickness=th,
        min_layers=int(min_layers),
        max_layers=max_layers,
    )

    bottom_mask = np.isin(zq, bottom_levels)
    top_mask = np.isin(zq, top_levels)
    overlap = np.any(bottom_mask & top_mask)
    if overlap:
        raise ValueError("Bottom/top grips overlap. Reduce end_frac or adjust eps/min_layers.")

    middle_mask = ~(bottom_mask | top_mask)

    meta = dict(
        z_min=z_min, z_max=z_max, z_range=z_range,
        thickness_raw=th_raw, thickness_used=th,
        n_levels=int(levels.size),
        bottom_levels=int(bottom_levels.size),
        top_levels=int(top_levels.size),
        bottom_atoms=int(np.sum(bottom_mask)),
        top_atoms=int(np.sum(top_mask)),
        middle_atoms=int(np.sum(middle_mask)),
        middle_frac_nominal=float(1.0 - 2.0 * float(end_frac)),
    )
    return bottom_mask, top_mask, middle_mask, meta


# =========================
# Optional constraints: wedge + r_xy
# =========================

def wedge_mask(atoms, theta_min_deg: float, theta_max_deg: float):
    pos = atoms.get_positions()
    center = pos.mean(axis=0)
    dx = pos[:, 0] - center[0]
    dy = pos[:, 1] - center[1]
    theta = np.degrees(np.arctan2(dy, dx))
    theta = (theta + 360.0) % 360.0
    return (theta >= theta_min_deg) & (theta <= theta_max_deg)


def rxy_values_and_mask(atoms, rmin=None, rmax=None):
    pos = atoms.get_positions()
    center = pos.mean(axis=0)
    dx = pos[:, 0] - center[0]
    dy = pos[:, 1] - center[1]
    r = np.sqrt(dx*dx + dy*dy)

    mask = np.ones(len(atoms), dtype=bool)
    if rmin is not None:
        mask &= (r >= float(rmin))
    if rmax is not None:
        mask &= (r <= float(rmax))
    return r, mask


def remove_one(atoms, idx: int):
    cp = atoms.copy()
    del cp[int(idx)]
    return cp


# =========================
# Core
# =========================

def generate_vacancies_v2(
    input_file: str,
    outdir: str,
    n: int,
    seed: int,
    prefix: str,
    # shared with auto_prep
    end_frac: float,
    min_thickness: float,
    max_thickness: float,
    min_layers: int,
    max_layers: int,   # 0 => unlimited
    eps: float,
    # extra filters (optional)
    use_wedge: bool,
    theta_min: float,
    theta_max: float,
    rmin: float | None,
    rmax: float | None,
    write_direct: bool,
    write_supercell_copy: bool,
    debug: bool,
):
    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)

    atoms = read(input_file)

    # Get masks using the SAME logic family as auto_prep
    max_layers_eff = None if int(max_layers) == 0 else int(max_layers)
    bmask, tmask, mmask, meta = get_grip_masks_like_autoprep(
        atoms,
        end_frac=end_frac,
        min_thickness=min_thickness,
        max_thickness=max_thickness,
        min_layers=min_layers,
        max_layers=max_layers_eff,
        eps=eps,
    )

    # Candidate pool starts from middle zone (guaranteed disjoint from grips)
    pool_mask = mmask.copy()

    # Optional wedge filter
    if use_wedge:
        pool_mask &= wedge_mask(atoms, theta_min, theta_max)

    # Optional r_xy filter
    rxy, rmask = rxy_values_and_mask(atoms, rmin=rmin, rmax=rmax)
    pool_mask &= rmask

    pool = np.where(pool_mask)[0].astype(int)
    if pool.size == 0:
        raise ValueError(
            "No atoms available for vacancy under current constraints.\n"
            f"middle_atoms={meta['middle_atoms']}, use_wedge={use_wedge}, rmin={rmin}, rmax={rmax}\n"
            "Try relaxing wedge/rmin/rmax or reduce grips (end_frac/min_thickness/min_layers)."
        )

    rng = np.random.default_rng(int(seed))
    n_pick = min(int(n), int(pool.size))
    picks = rng.choice(pool, size=n_pick, replace=False)

    # Optionally save a copy of the input (for tracking)
    if write_supercell_copy:
        write(str(outdir_p / f"input_copy.vasp"), atoms, vasp5=True, direct=write_direct)

    # Write summary
    summary_csv = outdir_p / "summary.csv"
    with open(summary_csv, "w", encoding="utf-8") as f:
        f.write("i,removed_index,r_xy_A,in_middle,in_bottom_grip,in_top_grip,outfile\n")
        for i, idx in enumerate(picks):
            vac = remove_one(atoms, int(idx))
            out = outdir_p / f"{prefix}{i:03d}.vasp"
            write(str(out), vac, vasp5=True, direct=write_direct)

            f.write(
                f"{i},{int(idx)},{rxy[int(idx)]:.6f},"
                f"{int(mmask[int(idx)])},{int(bmask[int(idx)])},{int(tmask[int(idx)])},"
                f"{out.name}\n"
            )

    if debug:
        print("[vacancy_v2] input =", input_file)
        print("[vacancy_v2] atoms =", len(atoms))
        print(f"[vacancy_v2] end_frac(per side)={end_frac:.3f} -> nominal middle={meta['middle_frac_nominal']*100:.1f}%")
        print(f"[vacancy_v2] thickness_used={meta['thickness_used']:.6f} A (raw={meta['thickness_raw']:.6f} A)")
        print(f"[vacancy_v2] grips: bottom_atoms={meta['bottom_atoms']} top_atoms={meta['top_atoms']}")
        print(f"[vacancy_v2] middle_atoms={meta['middle_atoms']}")
        print("[vacancy_v2] filters: use_wedge =", use_wedge, f"theta=[{theta_min},{theta_max}]" if use_wedge else "")
        print("[vacancy_v2] filters: r_xy =", f"[{rmin},{rmax}] A")
        print("[vacancy_v2] pool =", int(pool.size), "picked =", int(n_pick), "seed =", seed)
        print("[vacancy_v2] outdir =", str(outdir_p))
        print("[vacancy_v2] summary =", str(summary_csv))


# =========================
# CLI
# =========================

def main():
    p = argparse.ArgumentParser(
        description="vacancy_v2: generate vacancies ONLY in the middle zone that is disjoint from auto_prep grips (same selection logic)."
    )
    p.add_argument("input_file", help="Input structure (.vasp/.xyz/...)")
    p.add_argument("--outdir", default="vac_v2_out")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prefix", default="vac_mid_")

    # Shared with auto_prep
    p.add_argument("--end-frac", type=float, default=0.15, help="Per-side frozen fraction (auto_prep --end-frac). 0.15 => middle ~70% nominal.")
    p.add_argument("--min-thickness", type=float, default=2.0)
    p.add_argument("--max-thickness", type=float, default=8.0)
    p.add_argument("--min-layers", type=int, default=2)
    p.add_argument("--max-layers", type=int, default=0, help="0 => unlimited (same meaning as auto_prep)")
    p.add_argument("--eps", type=float, default=1e-3)

    # Optional wedge / r_xy
    p.add_argument("--use-wedge", action="store_true")
    p.add_argument("--theta-min", type=float, default=135.0)
    p.add_argument("--theta-max", type=float, default=180.0)
    p.add_argument("--rmin", type=float, default=None)
    p.add_argument("--rmax", type=float, default=None)

    # Output format
    p.add_argument("--cartesian", action="store_true", help="Write VASP cartesian (default direct)")
    p.add_argument("--write-input-copy", action="store_true", help="Write input_copy.vasp to outdir (for bookkeeping)")
    p.add_argument("--quiet", action="store_true")

    args = p.parse_args()

    generate_vacancies_v2(
        input_file=args.input_file,
        outdir=args.outdir,
        n=args.n,
        seed=args.seed,
        prefix=args.prefix,

        end_frac=float(args.end_frac),
        min_thickness=float(args.min_thickness),
        max_thickness=float(args.max_thickness),
        min_layers=int(args.min_layers),
        max_layers=int(args.max_layers),
        eps=float(args.eps),

        use_wedge=bool(args.use_wedge),
        theta_min=float(args.theta_min),
        theta_max=float(args.theta_max),
        rmin=(None if args.rmin is None else float(args.rmin)),
        rmax=(None if args.rmax is None else float(args.rmax)),

        write_direct=(not args.cartesian),
        write_supercell_copy=bool(args.write_input_copy),
        debug=(not args.quiet),
    )


if __name__ == "__main__":
    main()

