import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, Presentation, PresentationFile } from "@oai/artifact-tool";

const OUT_ROOT = "C:/Users/dawso/Desktop/DFTPY_QE_VACLM_NAS_20260701";
const OUT_PPTX = `${OUT_ROOT}/04_EVALUATION_SLIDES/DFTpy_QE_VACLM_evaluation_20260701.pptx`;
const QA_DIR = `${OUT_ROOT}/04_EVALUATION_SLIDES/qa_rendered`;
const MAP_IMAGE = `${OUT_ROOT}/01_DFTPY_ISERVICE_VACLM/02_SIMPLE_MAPS/professor_three_maps_minimal_iservice.png`;

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

async function readImageBlob(imagePath) {
  const bytes = await fs.readFile(imagePath);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

function addText(slide, text, position, style = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    fontSize: style.fontSize ?? 22,
    color: style.color ?? "slate-900",
    bold: style.bold ?? false,
    alignment: style.alignment ?? "left",
  };
  return shape;
}

function addTitle(slide, title, kicker = "Al single-vacancy calibration") {
  addText(slide, kicker, { left: 64, top: 38, width: 640, height: 30 }, {
    fontSize: 16,
    bold: true,
    color: "slate-500",
  });
  addText(slide, title, { left: 64, top: 72, width: 1060, height: 64 }, {
    fontSize: 38,
    bold: true,
    color: "slate-950",
  });
}

function addFooter(slide, page) {
  addText(slide, `DFTpy/QE VACLM package | ${page}`, { left: 64, top: 682, width: 620, height: 22 }, {
    fontSize: 12,
    color: "slate-400",
  });
  addText(slide, "NAS target: /OFDFT/TFvW mu-lambda test/DFTPY_QE_VACLM_NAS_20260701", { left: 620, top: 682, width: 600, height: 22 }, {
    fontSize: 12,
    color: "slate-400",
    alignment: "right",
  });
}

function addBox(slide, position, fill = "white", line = "slate-200") {
  return slide.shapes.add({
    geometry: "roundRect",
    position,
    fill,
    line: { style: "solid", fill: line, width: 1 },
    borderRadius: "rounded-xl",
  });
}

function addMetric(slide, label, value, note, left, top, accent = "slate-900") {
  addBox(slide, { left, top, width: 350, height: 140 }, "white", "slate-200");
  addText(slide, label, { left: left + 24, top: top + 18, width: 300, height: 28 }, {
    fontSize: 17,
    bold: true,
    color: "slate-500",
  });
  addText(slide, value, { left: left + 24, top: top + 52, width: 300, height: 48 }, {
    fontSize: 34,
    bold: true,
    color: accent,
  });
  addText(slide, note, { left: left + 24, top: top + 104, width: 300, height: 28 }, {
    fontSize: 15,
    color: "slate-600",
  });
}

async function main() {
  await fs.mkdir(path.dirname(OUT_PPTX), { recursive: true });
  await fs.mkdir(QA_DIR, { recursive: true });

  const p = Presentation.create({ slideSize: { width: 1280, height: 720 } });

  // Slide 1
  {
    const slide = p.slides.add();
    slide.background.fill = "slate-50";
    addText(slide, "DFTpy TFvW λ–μ Calibration", { left: 72, top: 115, width: 980, height: 72 }, {
      fontSize: 50,
      bold: true,
      color: "slate-950",
    });
    addText(slide, "with QE vc-relax reference for Al single vacancy", { left: 76, top: 200, width: 960, height: 44 }, {
      fontSize: 28,
      color: "slate-600",
    });
    addMetric(slide, "QE vacancy reference", "0.63695 eV", "3×3×3 Al, 108 → 107 atoms", 76, 324, "blue-700");
    addMetric(slide, "nearest DFTpy point", "0.66166 eV", "λ=0.9, μ=0.1, qualified", 466, 324, "emerald-700");
    addMetric(slide, "difference", "+0.02471 eV", "DFTpy − QE", 856, 324, "amber-700");
    addFooter(slide, "1");
  }

  // Slide 2
  {
    const slide = p.slides.add();
    slide.background.fill = "white";
    addTitle(slide, "What is in the NAS package");
    addBox(slide, { left: 72, top: 168, width: 1136, height: 410 }, "slate-50", "slate-200");
    const rows = [
      ["01_DFTPY_ISERVICE_VACLM", "iService DFTpy λ–μ scan: raw cases, inputs, outputs, trajectories, scripts, tables, maps."],
      ["02_QE_REFERENCE_VCRELAX", "QE 7.5 pristine and vacancy vc-relax reference: pw.in, pw.out, pseudo, structures, run script."],
      ["03_COMPARISON_SUMMARY", "QE parsed energies and DFTpy-vs-QE ranking table."],
      ["04_EVALUATION_SLIDES", "This concise discussion deck."],
    ];
    rows.forEach((r, i) => {
      const y = 198 + i * 82;
      addText(slide, r[0], { left: 108, top: y, width: 330, height: 30 }, {
        fontSize: 23,
        bold: true,
        color: "slate-950",
      });
      addText(slide, r[1], { left: 462, top: y, width: 700, height: 42 }, {
        fontSize: 19,
        color: "slate-600",
      });
    });
    addFooter(slide, "2");
  }

  // Slide 3
  {
    const slide = p.slides.add();
    slide.background.fill = "white";
    addTitle(slide, "DFTpy λ–μ scan maps");
    const img = await readImageBlob(MAP_IMAGE);
    slide.images.add({
      blob: img,
      contentType: "image/png",
      alt: "DFTpy lambda-mu maps for lattice constant, vacancy formation energy, and KEDF",
      fit: "contain",
      position: { left: 72, top: 150, width: 1136, height: 455 },
    });
    addText(slide, "Middle map is vacancy formation energy using Gillan-style perfect/defect subtraction.", { left: 92, top: 612, width: 1080, height: 34 }, {
      fontSize: 18,
      color: "slate-600",
    });
    addFooter(slide, "3");
  }

  // Slide 4
  {
    const slide = p.slides.add();
    slide.background.fill = "white";
    addTitle(slide, "QE reference calculation");
    addBox(slide, { left: 72, top: 160, width: 1136, height: 380 }, "slate-50", "slate-200");
    addText(slide, "Formula", { left: 110, top: 190, width: 220, height: 32 }, { fontSize: 22, bold: true, color: "slate-500" });
    addText(slide, "Eᶠᵛᵃᶜ = Eᵥₐ𝚌(Al107) − (107/108) Eₚᵣᵢₛₜᵢₙₑ(Al108)", { left: 300, top: 188, width: 820, height: 40 }, { fontSize: 25, bold: true, color: "slate-950" });
    const data = [
      ["pristine vc-relax", "-4266.26536555 Ry", "JOB DONE"],
      ["vacancy vc-relax", "-4226.71609376 Ry", "JOB DONE"],
      ["formation energy", "0.636946 eV", "QE reference"],
    ];
    data.forEach((r, i) => {
      const y = 278 + i * 70;
      addText(slide, r[0], { left: 130, top: y, width: 300, height: 34 }, { fontSize: 22, bold: true, color: "slate-700" });
      addText(slide, r[1], { left: 470, top: y, width: 360, height: 34 }, { fontSize: 24, bold: true, color: i === 2 ? "blue-700" : "slate-950" });
      addText(slide, r[2], { left: 860, top: y, width: 230, height: 34 }, { fontSize: 20, color: "slate-500" });
    });
    addFooter(slide, "4");
  }

  // Slide 5
  {
    const slide = p.slides.add();
    slide.background.fill = "slate-50";
    addTitle(slide, "Current interpretation");
    addText(slide, "Coarse λ–μ scan result", { left: 88, top: 170, width: 440, height: 36 }, {
      fontSize: 26,
      bold: true,
      color: "slate-950",
    });
    addText(slide, "The nearest qualified DFTpy point to the QE reference is λ=0.9, μ=0.1.", { left: 88, top: 220, width: 1030, height: 44 }, {
      fontSize: 28,
      color: "slate-700",
    });
    addMetric(slide, "QE reference", "0.63695 eV", "baseline for selection", 88, 330, "blue-700");
    addMetric(slide, "DFTpy λ=0.9 μ=0.1", "0.66166 eV", "nearest qualified coarse-grid point", 466, 330, "emerald-700");
    addMetric(slide, "absolute difference", "0.02471 eV", "coarse grid only", 844, 330, "amber-700");
    addText(slide, "Next: use QE as the selection target, then refine locally around the low-μ / high-λ region before applying to divacancy or nanostructures.", { left: 88, top: 520, width: 1080, height: 72 }, {
      fontSize: 24,
      color: "slate-800",
    });
    addFooter(slide, "5");
  }

  // Slide 6
  {
    const slide = p.slides.add();
    slide.background.fill = "white";
    addTitle(slide, "Action items satisfied");
    const items = [
      ["Complete DFTpy input/output", "included under 01_DFTPY_ISERVICE_VACLM/03_RAW_CASES"],
      ["QE/DFT reference", "included under 02_QE_REFERENCE_VCRELAX"],
      ["Comparison table", "03_COMPARISON_SUMMARY/dftpy_nearest_to_qe_reference.csv"],
      ["Evaluation slides", "04_EVALUATION_SLIDES/DFTpy_QE_VACLM_evaluation_20260701.pptx"],
    ];
    items.forEach((item, i) => {
      const y = 170 + i * 92;
      addBox(slide, { left: 90, top: y, width: 1100, height: 66 }, i % 2 ? "slate-50" : "white", "slate-200");
      addText(slide, item[0], { left: 122, top: y + 16, width: 310, height: 30 }, { fontSize: 22, bold: true, color: "slate-950" });
      addText(slide, item[1], { left: 464, top: y + 16, width: 680, height: 30 }, { fontSize: 20, color: "slate-600" });
    });
    addFooter(slide, "6");
  }

  for (const [i, slide] of p.slides.items.entries()) {
    const png = await p.export({ slide, format: "png", scale: 1 });
    await writeBlob(`${QA_DIR}/slide-${String(i + 1).padStart(2, "0")}.png`, png);
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(`${QA_DIR}/slide-${String(i + 1).padStart(2, "0")}.layout.json`, await layout.text());
  }
  const montage = await p.export({ format: "webp", montage: true, scale: 1 });
  await writeBlob(`${QA_DIR}/deck-montage.webp`, montage);

  const pptx = await PresentationFile.exportPptx(p);
  await pptx.save(OUT_PPTX);
  console.log(OUT_PPTX);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
