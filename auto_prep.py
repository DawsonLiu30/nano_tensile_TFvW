import argparse
import numpy as np
from pathlib import Path
from ase.io import read

# =========================
# Internal Helper Functions
# =========================

def _unique_levels(z, eps):
    """Quantize z-coordinates into discrete levels."""
    q = np.round(z / eps) * eps
    levels = np.unique(q)
    levels.sort()
    return q, levels

def _pick_levels(levels, z_min, z_max, target_thickness, min_layers, max_layers):
    """
    Select atomic layers from bottom and top based on thickness and layer count.
    """

    # ---- Bottom (bottom-up) ----
    bottom_levels = []
    for lv in levels:
        if lv <= z_min + target_thickness:
            bottom_levels.append(lv)
        else:
            break

    # Ensure minimum layers
    if len(bottom_levels) < min_layers:
        bottom_levels = list(levels[:min(min_layers, len(levels))])

    # Apply max_layers limit
    if max_layers is not None and max_layers > 0:
        bottom_levels = bottom_levels[:min(max_layers, len(bottom_levels))]

    # ---- Top (top-down) ----
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

    return np.array(bottom_levels, dtype=float), np.array(top_levels, dtype=float)

def _compute_grip_thickness(zq, bottom_idx, top_idx):
    """Return actual grip thickness per side (A)."""
    z = zq.astype(float)
    bot_th = float(z[bottom_idx].max() - z[bottom_idx].min()) if bottom_idx.size else 0.0
    top_th = float(z[top_idx].max() - z[top_idx].min()) if top_idx.size else 0.0
    return bot_th, top_th

# =========================
# Core Function: Select grips
# =========================

def get_grip_indices(
    atoms,
    end_frac=0.10,          # Fraction per side (default 10%)
    min_thickness=2.0,      # Safety lower bound
    max_thickness=8.0,      # Safety upper bound
    min_layers=2,
    max_layers=None,        # Default unlimited
    eps=1e-3,
    debug=True,
):
    pos = atoms.get_positions()
    if pos.size == 0:
        raise ValueError("Empty structure: no atoms found.")

    z = pos[:, 2].astype(float)
    z_min = float(z.min())
    z_max = float(z.max())
    z_range = z_max - z_min

    if z_range <= 0:
        raise ValueError("Invalid structure: z_range <= 0")

    # ---- Core: Target thickness for end_frac ----
    target_thickness_raw = float(z_range * float(end_frac))
    target_thickness = float(np.clip(target_thickness_raw, float(min_thickness), float(max_thickness)))

    # ---- Find z-levels ----
    zq, levels = _unique_levels(z, float(eps))

    if levels.size < 2:
        raise ValueError("Not enough distinct z-levels detected. Try increasing --eps.")

    bottom_levels, top_levels = _pick_levels(
        levels=levels,
        z_min=z_min,
        z_max=z_max,
        target_thickness=target_thickness,
        min_layers=int(min_layers),
        max_layers=max_layers,
    )

    bottom_mask = np.isin(zq, bottom_levels)
    top_mask = np.isin(zq, top_levels)

    bottom_idx = np.where(bottom_mask)[0].astype(int)
    top_idx = np.where(top_mask)[0].astype(int)

    if bottom_idx.size == 0:
        raise ValueError("No atoms selected for bottom grip.")
    if top_idx.size == 0:
        raise ValueError("No atoms selected for top grip.")

    overlap = np.intersect1d(bottom_idx, top_idx)
    if overlap.size > 0:
        raise ValueError(
            f"Bottom and top grips overlap (n={overlap.size}). "
            "Check eps or reduce end_frac."
        )

    bottom_z = z[bottom_idx]
    top_z = z[top_idx]
    grip_fraction = (bottom_idx.size + top_idx.size) / len(z)

    bot_th, top_th = _compute_grip_thickness(zq, bottom_idx, top_idx)

    if debug:
        print(f"[AutoPrep] Atomic Z-range: {z_range:.6f} A")
        print(f"           z_min={z_min:.6f}, z_max={z_max:.6f}")
        print(f"           End-fraction (per side): {float(end_frac):.4f}  -> target_thickness_raw={target_thickness_raw:.6f} A")
        print(f"           thickness clip: [{float(min_thickness):.3f}, {float(max_thickness):.3f}] -> target_thickness={target_thickness:.6f} A")
        print(f"           eps={float(eps):.6g}, distinct_z_levels={levels.size}")
        print(f"           min_layers={int(min_layers)}, max_layers={'unlimited' if (max_layers is None) else int(max_layers)}")
        print(f"[AutoPrep] Bottom: layers={bottom_levels.size}, atoms={bottom_idx.size}, "
              f"z=[{bottom_z.min():.6f}, {bottom_z.max():.6f}], thickness~={bot_th:.6f} A")
        print(f"[AutoPrep] Top:    layers={top_levels.size}, atoms={top_idx.size}, "
              f"z=[{top_z.min():.6f}, {top_z.max():.6f}], thickness~={top_th:.6f} A")
        print(f"[AutoPrep] Gap (top_min - bottom_max): {top_z.min() - bottom_z.max():.6f} A")
        print(f"[AutoPrep] Total grip fraction: {grip_fraction:.6f}")

        # Warning about max_layers truncating the grip
        if max_layers is not None and max_layers > 0:
            print("[AutoPrep][Hint] max_layers is limiting grip thickness. "
                  "For stiffer grips, consider --max-layers 0 (unlimited).")

    return bottom_idx, top_idx

# =========================
# CLI Interface
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Auto-generate grip indices (end-fraction of z-length, layer-aligned)."
    )

    parser.add_argument("input_file", help="Path to the structure file (.vasp, .xyz, etc.)")
    parser.add_argument("--output-dir", default=".", help="Directory to save .npy files")

    parser.add_argument("--end-frac", type=float, default=0.10,
                        help="Grip thickness fraction per side (default: 0.10)")

    parser.add_argument("--min-thickness", type=float, default=2.0,
                        help="Minimum grip thickness in Angstroms")
    parser.add_argument("--max-thickness", type=float, default=8.0,
                        help="Maximum grip thickness in Angstroms")
    parser.add_argument("--min-layers", type=int, default=2,
                        help="Minimum number of z-layers per grip")

    parser.add_argument("--max-layers", type=int, default=0,
                        help="Maximum number of z-layers per grip (0 for unlimited)")

    parser.add_argument("--eps", type=float, default=1e-3,
                        help="Z-level grouping tolerance in Angstroms")

    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"[Error] Input file not found: {input_path}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Processing {input_path.name} ---")

    try:
        atoms = read(str(input_path))
    except Exception as e:
        print(f"[Error] Failed to read structure: {e}")
        return

    max_layers = None if int(args.max_layers) == 0 else int(args.max_layers)

    try:
        bot_idx, top_idx = get_grip_indices(
            atoms,
            end_frac=float(args.end_frac),
            min_thickness=float(args.min_thickness),
            max_thickness=float(args.max_thickness),
            min_layers=int(args.min_layers),
            max_layers=max_layers,
            eps=float(args.eps),
            debug=True,
        )
    except Exception as e:
        print(f"[Error] Auto grip selection failed: {e}")
        return

    bot_path = output_dir / "bottom_idx.npy"
    top_path = output_dir / "top_idx.npy"

    np.save(str(bot_path), bot_idx)
    np.save(str(top_path), top_idx)

    print(f"[Success] Saved grips:")
    print(f"          Bottom atoms: {len(bot_idx)} -> {bot_path}")
    print(f"          Top atoms:    {len(top_idx)} -> {top_path}")

if __name__ == "__main__":
    main()
