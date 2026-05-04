from __future__ import annotations

import sys
import tempfile
import zipfile
from pathlib import Path


def patch_theme_xml(xml: str) -> str:
    xml = xml.replace('typeface="Calibri Light"', 'typeface="Times New Roman"')
    xml = xml.replace('typeface="Calibri"', 'typeface="Times New Roman"')
    return xml


def patch_pptx_theme(pptx_path: Path) -> None:
    if not pptx_path.exists():
        raise FileNotFoundError(pptx_path)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pptx") as tmp:
        tmp_path = Path(tmp.name)

    try:
        with zipfile.ZipFile(pptx_path, "r") as src, zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as dst:
            for info in src.infolist():
                data = src.read(info.filename)
                if info.filename == "ppt/theme/theme1.xml":
                    data = patch_theme_xml(data.decode("utf-8")).encode("utf-8")
                dst.writestr(info, data)
        tmp_path.replace(pptx_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python patch_ppt_theme_times_new_roman.py <pptx_path>")
    patch_pptx_theme(Path(sys.argv[1]).resolve())


if __name__ == "__main__":
    main()
