from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


FORMULAS = {
    "grip_distance": r"$L_i = z_{\mathrm{top,avg}}(i) - z_{\mathrm{bottom,avg}}(i)$",
    "engineering_strain": r"$\varepsilon_i = \dfrac{L_i - L_0}{L_0}$",
    "periodic_strain": r"$\varepsilon = \dfrac{L_z - L_{z,0}}{L_{z,0}}$",
    "periodic_strain_compare": r"$\varepsilon_{\mathrm{periodic}} = \dfrac{L_z - L_{z,0}}{L_{z,0}}$",
    "grip_strain_compare": r"$\varepsilon_{\mathrm{grip}} = \dfrac{L_g - L_{g,0}}{L_{g,0}}$",
    "reference_area": r"$A_{\mathrm{ref}} = \pi \,\Delta x_{\mathrm{ref}}\,\Delta y_{\mathrm{ref}}/4$",
    "sigma_top": r"$\sigma_{\mathrm{top}} = -\,F_{\mathrm{top},z}/A_{\mathrm{ref}} \times 160.2177$",
    "sigma_bottom": r"$\sigma_{\mathrm{bottom}} = +\,F_{\mathrm{bottom},z}/A_{\mathrm{ref}} \times 160.2177$",
    "sigma_grip_raw": r"$\sigma_{\mathrm{grip,raw}} = (\sigma_{\mathrm{top}} + \sigma_{\mathrm{bottom}})/2$",
    "sigma_primary": r"$\sigma_{\mathrm{primary}}(i)=\sigma_{\mathrm{grip,raw}}(i)-\sigma_{\mathrm{grip,raw}}(0)$",
    "sigma_cell_wire": r"$\sigma_{\mathrm{cell,wire}}=\sigma_{\mathrm{cell},zz}\times A_{\mathrm{cell}}/A_{\mathrm{ref}}$",
    "periodic_sigma_wire": r"$\sigma_{\mathrm{wire}}=\sigma_{\mathrm{cell},zz}\times A_{\mathrm{cell}}/A_{\mathrm{wire}}$",
    "periodic_sigma_compare": r"$\sigma_{\mathrm{wire}}^{\mathrm{periodic}}=\sigma_{\mathrm{cell},zz}\times A_{\mathrm{cell}}/A_{\mathrm{wire}}$",
    "grip_sigma_compare": r"$\sigma_{\mathrm{wire}}^{\mathrm{grip}}=\frac{1}{2}\left(-\frac{F_{\mathrm{top},z}}{A_{\mathrm{ref}}}+\frac{F_{\mathrm{bottom},z}}{A_{\mathrm{ref}}}\right)160.2177-\sigma_{\mathrm{grip,raw}}(0)$",
}

FORMULA_SIZES = {
    "grip_distance": 40,
    "engineering_strain": 46,
    "periodic_strain": 46,
    "periodic_strain_compare": 42,
    "grip_strain_compare": 42,
    "reference_area": 42,
    "sigma_top": 40,
    "sigma_bottom": 40,
    "sigma_grip_raw": 39,
    "sigma_primary": 37,
    "sigma_cell_wire": 37,
    "periodic_sigma_wire": 37,
    "periodic_sigma_compare": 31,
    "grip_sigma_compare": 22,
}

def render_formula_png(outdir: Path, name: str, formula: str, *, fontsize: int = 40) -> None:
    fig = plt.figure(figsize=(10, 1.3), dpi=300)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.text(
        0.0,
        0.5,
        formula,
        fontsize=fontsize,
        va="center",
        ha="left",
        color="#1D2329",
    )
    outpath = outdir / f"{name}.png"
    fig.savefig(outpath, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    outdir = root / "results" / "professor_review" / "ppt_ready" / "decks" / "stress_strain_methodology_defense" / "formula_assets"
    outdir.mkdir(parents=True, exist_ok=True)

    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.family"] = "Times New Roman"

    for name, formula in FORMULAS.items():
        render_formula_png(outdir, name, formula, fontsize=FORMULA_SIZES.get(name, 40))

    print(outdir)


if __name__ == "__main__":
    main()
