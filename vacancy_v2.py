import numpy as np
from ase.io import write

# ==========================================
# Default Configuration for Vacancy
# ==========================================
DEFAULT_VAC_CONFIG = {
    "enabled": False,        # Set to True in notebook to activate
    "n": 10,                 # Number of vacancies to create
    "seed": 42,              # Random seed for reproducibility
    "end_frac": 0.15,        # Fraction of ends to protect (grips)
    "min_thickness": 4.0,    # Min grip thickness (A)
    "max_thickness": 8.0,    # Max grip thickness (A)
    "min_layers": 2,         # Min atomic layers for grip
    "max_layers": 0,         # Max atomic layers (0 = no limit)
    "eps": 0.5,              # Z-level grouping tolerance
    # Wedge / Radial options
    "use_wedge": False,      # Restrict to a wedge shape
    "theta_min": 135.0,      # Degrees
    "theta_max": 180.0,
    "rmin": 0.0,             # Min radius (A)
    "rmax": 100.0            # Max radius (A)
}

def _unique_levels(z, eps):
    """Quantize z-coordinates into discrete levels."""
    q = np.round(z / eps) * eps
    levels = np.unique(q)
    levels.sort()
    return q, levels

def _pick_levels(levels, z_min, z_max, target_thickness, min_layers, max_layers):
    """Identifies z-levels that belong to the grips."""
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
        top_levels = list(levels[-min(min_layers, len(levels))])
    if max_layers is not None and max_layers > 0:
        top_levels = top_levels[:min(max_layers, len(top_levels))]
        
    return set(bottom_levels), set(top_levels)

def remove_atoms(atoms, config=DEFAULT_VAC_CONFIG):
    """
    Removes 'n' atoms randomly from the 'free' region (excluding grips).
    Returns a NEW atoms object with vacancies.
    """
    # Safety check: if disabled, return copy of original
    if not config.get("enabled", False):
        return atoms.copy()

    # Use config values
    n = config.get("n", 10)
    seed = config.get("seed", 42)
    
    # 1. Identify Grip Regions (Protected)
    z = atoms.get_positions()[:, 2]
    z_min, z_max = z.min(), z.max()
    L = z_max - z_min
    
    # Determine grip thickness
    target_thick = L * config.get("end_frac", 0.15)
    target_thick = max(target_thick, config.get("min_thickness", 4.0))
    target_thick = min(target_thick, config.get("max_thickness", 8.0))
    
    _, levels = _unique_levels(z, config.get("eps", 0.5))
    bot_levs, top_levs = _pick_levels(levels, z_min, z_max, target_thick, 
                                      config.get("min_layers", 2), config.get("max_layers", 0))
    
    # 2. Identify Candidates (Free Atoms)
    positions = atoms.get_positions()
    candidates = []
    
    # Pre-calculate cylindrical coords if needed
    use_wedge = config.get("use_wedge", False)
    rmin = config.get("rmin", 0)
    rmax = config.get("rmax", 100)
    
    if use_wedge or rmin > 0 or rmax < 100:
        xy = positions[:, :2]
        r = np.linalg.norm(xy, axis=1)
        theta = np.degrees(np.arctan2(xy[:, 1], xy[:, 0]))
        theta = np.where(theta < 0, theta + 360, theta)

    for i in range(len(atoms)):
        z_val = z[i]
        z_quant = np.round(z_val / config.get("eps", 0.5)) * config.get("eps", 0.5)
        
        # Check if in grip
        if (z_quant in bot_levs) or (z_quant in top_levs):
            continue 
            
        # Check Wedge / Radius
        if use_wedge:
            th = theta[i]
            if not (config.get("theta_min", 135) <= th <= config.get("theta_max", 180)):
                continue
                
        if rmin > 0 and r[i] < rmin: continue
        if rmax < 100 and r[i] > rmax: continue
            
        candidates.append(i)
        
    candidates = np.array(candidates)
    
    if len(candidates) < n:
        print(f"Warning: Requested {n} vacancies but only {len(candidates)} valid atoms found.")
        n = len(candidates)
        
    if n <= 0:
        return atoms.copy()

    # 3. Random Selection
    rng = np.random.default_rng(seed)
    remove_idx = rng.choice(candidates, size=n, replace=False)
    
    # 4. Create new atoms object
    mask = np.ones(len(atoms), dtype=bool)
    mask[remove_idx] = False
    
    new_atoms = atoms[mask]
    return new_atoms