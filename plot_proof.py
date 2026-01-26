import numpy as np
from ase.io import read
import matplotlib.pyplot as plt

fixed_b = np.load("bottom_idx.npy").astype(int).ravel()
fixed_t = np.load("top_idx.npy").astype(int).ravel()

def span_z(atoms, idx):
    z = atoms.get_positions()[:, 2]
    zz = z[idx]
    return float(zz.max() - zz.min())

def grips(atoms):
    z = atoms.get_positions()[:, 2]
    zb = float(z[fixed_b].max())
    zt = float(z[fixed_t].min())
    return zb, zt, float(zt - zb)

def free_mask(atoms, eps=1e-4):
    z = atoms.get_positions()[:, 2]
    zb = float(z[fixed_b].max())
    zt = float(z[fixed_t].min())
    return (z > zb + eps) & (z < zt - eps)

def span_mask(atoms, mask):
    z = atoms.get_positions()[mask, 2]
    if z.size < 2:
        return 0.0
    return float(z.max() - z.min())

data = np.genfromtxt("results/summary.csv", delimiter=",", names=True, dtype=None, encoding=None)
cycles = np.array(data["cycle"], dtype=int)

L_grip = []
L_free = []
L_bot = []
L_top = []

for cyc in cycles:
    atoms = read(f"results/cycle_{cyc:03d}_relaxed.xyz")
    zb, zt, L = grips(atoms)
    L_grip.append(L)
    L_free.append(span_mask(atoms, free_mask(atoms)))
    L_bot.append(span_z(atoms, fixed_b))
    L_top.append(span_z(atoms, fixed_t))

plt.figure()
plt.plot(cycles, L_grip, marker="o", label="Grip distance L = zt - zb (should increase)")
plt.plot(cycles, L_free, marker="o", label="Free zone span (should increase)")
plt.plot(cycles, L_bot, marker="o", label="Bottom clamp thickness (should stay ~const)")
plt.plot(cycles, L_top, marker="o", label="Top clamp thickness (should stay ~const)")
plt.xlabel("Cycle")
plt.ylabel("Length in z (Å)")
plt.title("Proof of stretching: grips vs free zone vs clamp thickness (relaxed)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("results/proof_four_curves.png", dpi=150)
print("Saved: results/proof_four_curves.png")

