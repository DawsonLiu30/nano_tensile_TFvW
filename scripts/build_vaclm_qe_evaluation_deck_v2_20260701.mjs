import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const OUT_ROOT = "C:/Users/dawso/Desktop/DFTPY_QE_VACLM_NAS_20260701";
const OUT_PPTX = `${OUT_ROOT}/04_EVALUATION_SLIDES/DFTpy_QE_VACLM_evaluation_20260701_v2.pptx`;
const QA_DIR = `${OUT_ROOT}/04_EVALUATION_SLIDES/qa_rendered_v2`;
const MAP_IMAGE = `${OUT_ROOT}/01_DFTPY_ISERVICE_VACLM/02_SIMPLE_MAPS/professor_three_maps_minimal_iservice.png`;

const W = 1280;
const H = 720;
const BG = "#F7F4EE";
const NAVY = "#12324A";
const BLUE = "#28608A";
const GREEN_BG = "#E9F4EA";
const BLUE_BG = "#EAF1F7";
const TAN_BG = "#F5E9D6";
const TEXT = "#111827";
const MUTED = "#4B5563";
const RED = "#B42318";

async function writeBlob(filePath, blob) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

async function readImageBlob(imagePath) {
  const bytes = await fs.readFile(imagePath);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

function shape(slide, position, fill = "none", line = "none", radius = "rounded-none") {
  return slide.shapes.add({
    geometry: radius === "rounded-none" ? "rect" : "roundRect",
    position,
    fill,
    line: { style: "solid", fill: line, width: line === "none" ? 0 : 1 },
    borderRadius: radius,
  });
}

function text(slide, value, position, style = {}) {
  const box = slide.shapes.add({
    geometry: "textbox",
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  box.text = value;
  box.text.style = {
    fontSize: style.fontSize ?? 20,
    bold: style.bold ?? false,
    italic: style.italic ?? false,
    color: style.color ?? TEXT,
    alignment: style.alignment ?? "left",
  };
  return box;
}

function title(slide, value, subtitle = "") {
  text(slide, value, { left: 32, top: 28, width: 1080, height: 48 }, {
    fontSize: 26,
    bold: false,
    color: TEXT,
  });
  if (subtitle) {
    text(slide, subtitle, { left: 32, top: 74, width: 920, height: 28 }, {
      fontSize: 13,
      color: MUTED,
    });
  }
  shape(slide, { left: 32, top: 104, width: 1160, height: 2 }, NAVY, NAVY);
}

function footer(slide, n) {
  text(slide, "NAS: /OFDFT/TFvW mu-lambda test/DFTPY_QE_VACLM_NAS_20260701", {
    left: 32, top: 684, width: 760, height: 18,
  }, { fontSize: 9, color: "#6B7280" });
  text(slide, `DFTpy/QE single vacancy calibration | ${n}`, {
    left: 920, top: 684, width: 280, height: 18,
  }, { fontSize: 9, color: "#6B7280", alignment: "right" });
}

function bullet(slide, value, left, top, width = 1040, color = TEXT) {
  text(slide, `- ${value}`, { left, top, width, height: 26 }, {
    fontSize: 16,
    color,
  });
}

function panel(slide, left, top, width, height, fill, heading, body, headingColor = TEXT) {
  shape(slide, { left, top, width, height }, fill, "#D7DEE6", "rounded-lg");
  text(slide, heading, { left: left + 18, top: top + 14, width: width - 36, height: 24 }, {
    fontSize: 16,
    bold: true,
    color: headingColor,
  });
  text(slide, body, { left: left + 18, top: top + 46, width: width - 36, height: height - 58 }, {
    fontSize: 14,
    color: TEXT,
  });
}

async function main() {
  await fs.mkdir(path.dirname(OUT_PPTX), { recursive: true });
  await fs.rm(QA_DIR, { recursive: true, force: true });
  await fs.mkdir(QA_DIR, { recursive: true });

  const deck = Presentation.create({ slideSize: { width: W, height: H } });

  // 1. Background and purpose
  {
    const slide = deck.slides.add();
    slide.background.fill = BG;
    title(slide, "DFTpy TFvW lambda-mu calibration", "Al single-vacancy benchmark with QE reference");
    text(slide, "Current purpose:", { left: 32, top: 142, width: 420, height: 24 }, {
      fontSize: 16,
      bold: true,
    });
    bullet(slide, "Provide the raw DFTpy input/output package requested by Professor.", 32, 182);
    bullet(slide, "Use QE vc-relax as the KSDFT/DFT reference for vacancy formation energy.", 32, 218);
    bullet(slide, "Select a defensible TFvW lambda-mu region before divacancy or nanostructure calculations.", 32, 254);
    text(slide, "Reference target:", { left: 32, top: 330, width: 420, height: 24 }, {
      fontSize: 16,
      bold: true,
    });
    text(slide, "QE/PBE single-vacancy formation energy = 0.636946 eV", {
      left: 32, top: 370, width: 720, height: 36,
    }, { fontSize: 22, bold: true, color: BLUE });
    text(slide, "Gillan-style formula: E_f^vac = E_vac(Al107) - (107/108) E_pristine(Al108)", {
      left: 32, top: 422, width: 920, height: 30,
    }, { fontSize: 15, color: MUTED });
    footer(slide, 1);
  }

  // 2. Computational details
  {
    const slide = deck.slides.add();
    slide.background.fill = BG;
    title(slide, "Computational / technical details");
    panel(slide, 42, 140, 550, 170, BLUE_BG, "DFTpy TFvW lambda-mu scan",
      "System: conventional fcc Al 3x3x3\nPristine: Al108; vacancy: Al107\nXC / pseudo: LDA / al.lda.recpot\nKEDF: TFvW with lambda, mu scan\nRelaxation: full atom + cell relaxation through ASE vc-relax-equivalent driver\nSpacing: 0.20 A");
    panel(slide, 648, 140, 550, 170, BLUE_BG, "QE reference",
      "Code: QE 7.5\nFunctional / pseudo: PBE / Al_PAW_PBE.UPF\nCalculations: pristine vc-relax and vacancy vc-relax\nFormula: E_vac - (107/108) E_pristine\nPurpose: KSDFT baseline for lambda-mu selection");
    panel(slide, 42, 358, 1156, 140, GREEN_BG, "Why perfect and defective systems are both included",
      "Each DFTpy point contains a perfect Al108 calculation and a defective Al107 calculation. The vacancy formation energy is not a raw total energy; it is the scaled perfect/defect subtraction used for vacancy supercell calculations.");
    footer(slide, 2);
  }

  // 3. NAS package map
  {
    const slide = deck.slides.add();
    slide.background.fill = BG;
    title(slide, "Package structure");
    const rows = [
      ["01_DFTPY_ISERVICE_VACLM", "iService DFTpy scan: raw cases, ini inputs, stdout/output logs, trajectories, result.json, scripts, tables, maps."],
      ["02_QE_REFERENCE_VCRELAX", "QE pristine/vacancy reference: pw.in, pw.out, pseudo, VASP structures, local run logs, iService run notes."],
      ["03_COMPARISON_SUMMARY", "Parsed QE reference, DFTpy-vs-QE ranking table, and all-point comparison CSV."],
      ["04_EVALUATION_SLIDES", "This revised discussion deck."],
    ];
    rows.forEach((r, i) => {
      const y = 150 + i * 94;
      shape(slide, { left: 42, top: y, width: 1156, height: 70 }, i % 2 === 0 ? "#FFFFFF" : "#F1F5F9", "#D7DEE6", "rounded-lg");
      text(slide, r[0], { left: 70, top: y + 18, width: 340, height: 28 }, { fontSize: 19, bold: true });
      text(slide, r[1], { left: 430, top: y + 16, width: 720, height: 38 }, { fontSize: 15, color: MUTED });
    });
    footer(slide, 3);
  }

  // 4. Raw table / provenance
  {
    const slide = deck.slides.add();
    slide.background.fill = BG;
    title(slide, "Raw table and provenance requested by Professor", "Direct table from DFTpy raw outputs plus source folders");
    bullet(slide, "Professor raw table: Total energy, KEDF/kinetic energy, lattice constant.", 56, 152);
    bullet(slide, "Rows/columns: lambda x mu, 0.1 to 1.0.", 56, 190);
    bullet(slide, "Raw values are kept separate from vacancy-formation maps.", 56, 228, 1040, RED);
    bullet(slide, "Each flat CSV row has the source directory and input/output paths.", 56, 266);
    panel(slide, 56, 340, 1080, 150, TAN_BG, "Key distinction",
      "The raw table answers: what does the final DFTpy output report for the pristine calculation?\nThe formation-energy maps answer: which lambda-mu region matches the vacancy reference after perfect/defect subtraction?");
    footer(slide, 4);
  }

  // 5. DFTpy maps
  {
    const slide = deck.slides.add();
    slide.background.fill = BG;
    title(slide, "DFTpy lambda-mu maps", "Qualified iService vc-relax points only");
    const img = await readImageBlob(MAP_IMAGE);
    slide.images.add({
      blob: img,
      contentType: "image/png",
      alt: "DFTpy lambda-mu maps",
      fit: "contain",
      position: { left: 66, top: 128, width: 1140, height: 455 },
    });
    text(slide, "Left: relaxed lattice constant. Middle: vacancy formation energy. Right: KEDF contribution.", {
      left: 88, top: 604, width: 980, height: 24,
    }, { fontSize: 15, color: MUTED });
    text(slide, "Red reference lines: a0 ~ 4.05 A and vacancy Ef ~ 0.56-0.66 eV literature / QE range.", {
      left: 88, top: 632, width: 1040, height: 24,
    }, { fontSize: 15, color: RED });
    footer(slide, 5);
  }

  // 6. QE reference details
  {
    const slide = deck.slides.add();
    slide.background.fill = BG;
    title(slide, "QE vacancy reference");
    panel(slide, 48, 145, 540, 190, BLUE_BG, "Pristine vc-relax",
      "Final total energy:\n-4266.26536555 Ry\n\nStatus: JOB DONE\nAtoms: Al108");
    panel(slide, 650, 145, 540, 190, BLUE_BG, "Vacancy vc-relax",
      "Final total energy:\n-4226.71609376 Ry\n\nStatus: JOB DONE\nAtoms: Al107");
    panel(slide, 48, 380, 1142, 130, GREEN_BG, "Formation energy",
      "E_f^vac = E_vac(Al107) - (107/108) E_pristine(Al108) = 0.636946 eV\nThis value is used as the QE/KSDFT baseline for selecting lambda and mu.");
    footer(slide, 6);
  }

  // 7. DFTpy vs QE decision
  {
    const slide = deck.slides.add();
    slide.background.fill = BG;
    title(slide, "DFTpy vs QE: current calibration result");
    panel(slide, 56, 148, 330, 150, "#FFFFFF", "QE reference",
      "Ef^vac = 0.636946 eV\n\nBaseline target");
    panel(slide, 450, 148, 330, 150, "#FFFFFF", "Nearest DFTpy point",
      "lambda = 0.9\nmu = 0.1\nEf^vac = 0.661661 eV");
    panel(slide, 844, 148, 330, 150, "#FFFFFF", "Difference",
      "DFTpy - QE = +0.024714 eV\n\nCoarse-grid error");
    panel(slide, 56, 360, 1118, 120, GREEN_BG, "Interpretation",
      "The coarse lambda-mu scan already identifies the low-mu / high-lambda corner as the best current region. This should be refined locally before using the parameters for divacancy, strain-field, or nanostructure production calculations.");
    footer(slide, 7);
  }

  // 8. Discussion / next
  {
    const slide = deck.slides.add();
    slide.background.fill = BG;
    title(slide, "Interpretation / next discussion");
    bullet(slide, "The original high TFvW vacancy energy was mainly a KEDF-weight issue, not simply a missing-output issue.", 56, 156);
    bullet(slide, "QE reference is now included and should be the basis for lambda-mu selection.", 56, 198);
    bullet(slide, "Do not treat divacancy/nanostructure plots as final until lambda-mu choice is fixed.", 56, 240);
    bullet(slide, "Next calculation priority: fine scan around high lambda / low mu, then repeat any downstream cases if the selected parameters change.", 56, 282);
    panel(slide, 56, 382, 1118, 110, TAN_BG, "Short conclusion",
      "For the current coarse grid, lambda = 0.9 and mu = 0.1 is the nearest qualified DFTpy point to the QE vacancy reference. The full raw DFTpy and QE input/output provenance is included in the NAS package.");
    footer(slide, 8);
  }

  for (const [i, slide] of deck.slides.items.entries()) {
    const stem = `slide-${String(i + 1).padStart(2, "0")}`;
    await writeBlob(`${QA_DIR}/${stem}.png`, await deck.export({ slide, format: "png", scale: 1 }));
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(`${QA_DIR}/${stem}.layout.json`, await layout.text(), "utf8");
  }
  await writeBlob(`${QA_DIR}/deck-montage.webp`, await deck.export({ format: "webp", montage: true, scale: 1 }));

  const pptx = await PresentationFile.exportPptx(deck);
  await pptx.save(OUT_PPTX);
  console.log(OUT_PPTX);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
