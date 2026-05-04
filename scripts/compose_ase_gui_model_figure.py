from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = Path.home() / "Downloads"


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _trim_white(image: Image.Image, *, threshold: int = 248, pad: int = 12) -> Image.Image:
    rgba = image.convert("RGBA")
    pixels = rgba.load()
    width, height = rgba.size
    xs: list[int] = []
    ys: list[int] = []
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            if a > 0 and (r < threshold or g < threshold or b < threshold):
                xs.append(x)
                ys.append(y)
    if not xs:
        return rgba
    left = max(0, min(xs) - pad)
    top = max(0, min(ys) - pad)
    right = min(width, max(xs) + pad)
    bottom = min(height, max(ys) + pad)
    return rgba.crop((left, top, right, bottom))


def _load_panel(path: Path, *, remove_top_px: int) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    if remove_top_px > 0:
        image = image.crop((0, int(remove_top_px), image.width, image.height))
    return _trim_white(image, pad=10)


def _fit(image: Image.Image, box: tuple[int, int]) -> Image.Image:
    width, height = image.size
    max_w, max_h = box
    scale = min(max_w / width, max_h / height)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _paste_center(canvas: Image.Image, image: Image.Image, box: tuple[int, int, int, int]) -> None:
    x, y, w, h = box
    fitted = _fit(image, (w, h))
    px = x + (w - fitted.width) // 2
    py = y + (h - fitted.height) // 2
    canvas.alpha_composite(fitted, (px, py))


def _draw_panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    image: Image.Image,
    *,
    box: tuple[int, int, int, int],
    label: str,
    title: str,
) -> None:
    x, y, w, h = box
    label_font = _font(34, bold=True)
    title_font = _font(30, bold=True)
    draw.text((x, y), label, fill=(0, 0, 0), font=label_font)
    draw.text((x + 54, y + 2), title, fill=(0, 0, 0), font=title_font)
    _paste_center(canvas, image, (x, y + 52, w, h - 52))


def _center_title(draw: ImageDraw.ImageDraw, width: int, y: int, title: str, *, size: int = 36) -> None:
    title_font = _font(size, bold=True)
    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((width - (title_box[2] - title_box[0])) // 2, y), title, fill=(0, 0, 0), font=title_font)


def compose_orientation_figure(*, orientations: Image.Image, out: Path, pdf: Path) -> None:
    width, height = 2400, 760
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    _center_title(draw, width, 28, "Orientation-dependent nanocolumn cross sections", size=36)
    _paste_center(canvas, orientations, (70, 95, width - 140, height - 130))
    rgb = canvas.convert("RGB")
    rgb.save(out, dpi=(300, 300))
    rgb.save(pdf, resolution=300)


def compose_column_vacancy_figure(
    *,
    bulk: Image.Image,
    pristine: Image.Image,
    vacancy: Image.Image,
    out: Path,
    pdf: Path,
) -> None:
    width, height = 2400, 820
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    _center_title(draw, width, 28, "Reference, pristine, and vacancy-containing Al models", size=36)

    margin_x = 65
    gap = 50
    panel_w = (width - 2 * margin_x - 2 * gap) // 3
    panel_y = 105
    panel_h = 670

    _draw_panel(canvas, draw, bulk, box=(margin_x, panel_y, panel_w, panel_h), label="A", title="Bulk [111] reference")
    _draw_panel(
        canvas,
        draw,
        pristine,
        box=(margin_x + panel_w + gap, panel_y, panel_w, panel_h),
        label="B",
        title="Pristine [111] model",
    )
    _draw_panel(
        canvas,
        draw,
        vacancy,
        box=(margin_x + 2 * (panel_w + gap), panel_y, panel_w, panel_h),
        label="C",
        title="Vacancy [111] model",
    )

    rgb = canvas.convert("RGB")
    rgb.save(out, dpi=(300, 300))
    rgb.save(pdf, resolution=300)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose ASE-GUI style model figures for Section 4.2.")
    parser.add_argument("--bulk", default=str(DOWNLOADS / "bulk_oriented_111.png"))
    parser.add_argument("--pristine", default=str(DOWNLOADS / "Al_xtal_111_r10.0_zr2_n336.png"))
    parser.add_argument("--vacancy", default=str(DOWNLOADS / "Al_vac_111_r10.0_zr2_n335.png"))
    parser.add_argument("--orientations", default=str(DOWNLOADS / "vacancy_comparison.png"))
    parser.add_argument("--outdir", default="results/professor_review")
    parser.add_argument("--out", default="", help="Backward-compatible single output; writes the column/vacancy figure.")
    parser.add_argument("--pdf", default="", help="Backward-compatible single PDF path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = _resolve(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bulk = _load_panel(_resolve(args.bulk), remove_top_px=72)
    pristine = _load_panel(_resolve(args.pristine), remove_top_px=60)
    vacancy = _load_panel(_resolve(args.vacancy), remove_top_px=60)
    orientations = _load_panel(_resolve(args.orientations), remove_top_px=82)

    orientation_out = outdir / "04_2_orientation_comparison_ase_gui.png"
    orientation_pdf = orientation_out.with_suffix(".pdf")
    compose_orientation_figure(orientations=orientations, out=orientation_out, pdf=orientation_pdf)

    column_out = _resolve(args.out) if str(args.out).strip() else outdir / "04_2_column_vacancy_models_ase_gui.png"
    column_pdf = _resolve(args.pdf) if str(args.pdf).strip() else column_out.with_suffix(".pdf")
    compose_column_vacancy_figure(
        bulk=bulk,
        pristine=pristine,
        vacancy=vacancy,
        out=column_out,
        pdf=column_pdf,
    )
    print(f"[ase-gui-figure] output: {orientation_out}")
    print(f"[ase-gui-figure] output: {orientation_pdf}")
    print(f"[ase-gui-figure] output: {column_out}")
    print(f"[ase-gui-figure] output: {column_pdf}")


if __name__ == "__main__":
    main()
