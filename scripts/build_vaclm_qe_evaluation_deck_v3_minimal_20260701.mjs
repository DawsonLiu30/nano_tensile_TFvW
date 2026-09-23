import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const OUT_ROOT = "C:/Users/dawso/Desktop/DFTPY_QE_VACLM_NAS_20260701";
const OUT_PPTX = `${OUT_ROOT}/04_EVALUATION_SLIDES/DFTpy_QE_VACLM_minimal_discussion_20260701.pptx`;
const QA_DIR = `${OUT_ROOT}/04_EVALUATION_SLIDES/qa_minimal_v3`;
const COMPARE_CSV = `${OUT_ROOT}/03_COMPARISON_SUMMARY/dftpy_qe_all_points_comparison.csv`;

const BG = "#F7F4EE";
const TEXT = "#111827";
const MUTED = "#4B5563";
const NAVY = "#12324A";
const BLUE = "#246B8F";
const GREEN = "#16794C";
const AMBER = "#B45309";
const RED = "#B42318";
const LINE = "#CBD5E1";

const QE_EF = 0.6369464632544929;

async function writeBlob(filePath, blob) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

function addText(slide, value, position, style = {}) {
  const s = slide.shapes.add({
    geometry: "textbox",
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  s.text = value;
  s.text.style = {
    fontSize: style.fontSize ?? 20,
    bold: style.bold ?? false,
    italic: style.italic ?? false,
    color: style.color ?? TEXT,
    alignment: style.alignment ?? "left",
  };
  return s;
}

function rect(slide, position, fill, line = "none", radius = "rounded-none") {
  return slide.shapes.add({
    geometry: radius === "rounded-none" ? "rect" : "roundRect",
    position,
    fill,
    line: { style: "solid", fill: line, width: line === "none" ? 0 : 1 },
    borderRadius: radius,
  });
}

function title(slide, value, subtitle = "") {
  addText(slide, value, { left: 48, top: 40, width: 1050, height: 54 }, {
    fontSize: 38,
    bold: false,
    color: TEXT,
  });
  if (subtitle) {
    addText(slide, subtitle, { left: 50, top: 92, width: 950, height: 26 }, {
      fontSize: 16,
      color: MUTED,
    });
  }
  rect(slide, { left: 48, top: 126, width: 1160, height: 2 }, NAVY, NAVY);
}

function footer(slide, page) {
  addText(slide, "NAS: /OFDFT/TFvW mu-lambda test/DFTPY_QE_VACLM_NAS_20260701", {
    left: 48, top: 682, width: 760, height: 18,
  }, { fontSize: 10, color: "#6B7280" });
  addText(slide, `${page}`, { left: 1160, top: 682, width: 48, height: 18 }, {
    fontSize: 10,
    color: "#6B7280",
    alignment: "right",
  });
}

function bullet(slide, value, left, top, width = 1050, style = {}) {
  addText(slide, `- ${value}`, { left, top, width, height: style.height ?? 34 }, {
    fontSize: style.fontSize ?? 20,
    color: style.color ?? TEXT,
    bold: style.bold ?? false,
  });
}

function metric(slide, label, value, note, left, top, color) {
  rect(slide, { left, top, width: 342, height: 126 }, "#FFFFFF", LINE, "rounded-lg");
  addText(slide, label, { left: left + 18, top: top + 16, width: 300, height: 24 }, {
    fontSize: 16,
    bold: true,
    color: MUTED,
  });
  addText(slide, value, { left: left + 18, top: top + 46, width: 305, height: 42 }, {
    fontSize: 30,
    bold: true,
    color,
  });
  addText(slide, note, { left: left + 18, top: top + 92, width: 305, height: 22 }, {
    fontSize: 14,
    color: MUTED,
  });
}

function parseCsvLine(line) {
  const out = [];
  let cur = "";
  let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      if (quoted && line[i + 1] === '"') {
        cur += '"';
        i++;
      } else {
        quoted = !quoted;
      }
    } else if (ch === "," && !quoted) {
      out.push(cur);
      cur = "";
    } else {
      cur += ch;
    }
  }
  out.push(cur);
  return out;
}

async function loadCandidates() {
  const text = await fs.readFile(COMPARE_CSV, "utf8");
  const lines = text.trim().split(/\r?\n/);
  const headers = parseCsvLine(lines[0]);
  const rows = lines.slice(1).map((line) => {
    const vals = parseCsvLine(line);
    return Object.fromEntries(headers.map((h, i) => [h, vals[i] ?? ""]));
  });
  return rows
    .map((r) => ({
      setting: r.setting,
      lambda: Number(r.lambda),
      mu: Number(r.mu),
      ef: Number(r.dftpy_Ef_vac_eV),
      delta: Number(r.delta_vs_qe_eV),
      absDelta: Number(r.abs_delta_vs_qe_eV),
      a0: Number(r.lattice_constant_A),
      pfmax: Number(r.pristine_final_fmax_eV_A),
      vfmax: Number(r.vacancy_final_fmax_eV_A),
    }))
    .filter((r) => Number.isFinite(r.ef) && Number.isFinite(r.absDelta))
    .sort((a, b) => a.absDelta - b.absDelta);
}

function addCandidateTable(slide, candidates, left, top) {
  const cols = [
    ["lambda", 95],
    ["mu", 80],
    ["DFTpy Ef", 150],
    ["DFTpy-QE", 150],
    ["a0", 120],
    ["force note", 240],
  ];
  const totalW = cols.reduce((s, [, w]) => s + w, 0);
  rect(slide, { left, top, width: totalW, height: 42 }, "#EAF1F7", LINE);
  let x = left;
  cols.forEach(([name, w]) => {
    addText(slide, name, { left: x + 8, top: top + 10, width: w - 16, height: 20 }, {
      fontSize: 15,
      bold: true,
      color: NAVY,
    });
    x += w;
  });
  candidates.slice(0, 5).forEach((r, idx) => {
    const y = top + 42 + idx * 42;
    rect(slide, { left, top: y, width: totalW, height: 42 }, idx === 0 ? "#E9F4EA" : "#FFFFFF", LINE);
    const vals = [
      r.lambda.toFixed(1),
      r.mu.toFixed(1),
      `${r.ef.toFixed(3)} eV`,
      `${r.delta >= 0 ? "+" : ""}${r.delta.toFixed(3)} eV`,
      `${r.a0.toFixed(3)} A`,
      `${Math.max(r.pfmax, r.vfmax).toFixed(4)} eV/A`,
    ];
    x = left;
    vals.forEach((v, i) => {
      addText(slide, v, { left: x + 8, top: y + 10, width: cols[i][1] - 16, height: 20 }, {
        fontSize: 14,
        color: idx === 0 ? GREEN : TEXT,
        bold: idx === 0,
      });
      x += cols[i][1];
    });
  });
}

async function main() {
  await fs.rm(QA_DIR, { recursive: true, force: true });
  await fs.mkdir(QA_DIR, { recursive: true });
  const candidates = await loadCandidates();
  const best = candidates[0];

  const deck = Presentation.create({ slideSize: { width: 1280, height: 720 } });

  // Slide 1
  {
    const slide = deck.slides.add();
    slide.background.fill = BG;
    addText(slide, "DFTpy TFvW lambda-mu calibration", { left: 56, top: 118, width: 1040, height: 66 }, {
      fontSize: 50,
      bold: false,
      color: TEXT,
    });
    addText(slide, "single Al vacancy, calibrated against QE vc-relax", { left: 60, top: 194, width: 980, height: 38 }, {
      fontSize: 24,
      color: MUTED,
    });
    rect(slide, { left: 60, top: 286, width: 1080, height: 2 }, NAVY, NAVY);
    addText(slide, "Main result", { left: 60, top: 330, width: 300, height: 34 }, {
      fontSize: 26,
      bold: true,
    });
    addText(slide, `Best current DFTpy match: lambda = ${best.lambda.toFixed(1)}, mu = ${best.mu.toFixed(1)}`, {
      left: 60, top: 374, width: 760, height: 38,
    }, { fontSize: 28, bold: true, color: GREEN });
    addText(slide, `DFTpy Ef = ${best.ef.toFixed(6)} eV; QE Ef = ${QE_EF.toFixed(6)} eV; difference = +${best.absDelta.toFixed(6)} eV`, {
      left: 60, top: 426, width: 1030, height: 34,
    }, { fontSize: 20, color: TEXT });
    footer(slide, 1);
  }

  // Slide 2
  {
    const slide = deck.slides.add();
    slide.background.fill = BG;
    title(slide, "Gillan formation-energy method", "Perfect and defective supercells are compared with atom-number scaling");
    addText(slide, "Why not compare raw total energies?", { left: 62, top: 166, width: 640, height: 30 }, {
      fontSize: 24,
      bold: true,
    });
    bullet(slide, "The pristine cell has 108 Al atoms; the vacancy cell has 107 Al atoms.", 62, 212);
    bullet(slide, "Raw total energies therefore cannot be compared directly.", 62, 252);
    bullet(slide, "Gillan rewrites the vacancy problem using two periodic systems: perfect N atoms and defective N-1 atoms plus one vacancy.", 62, 292, 1060, { height: 56 });
    rect(slide, { left: 110, top: 390, width: 1010, height: 86 }, "#FFFFFF", LINE, "rounded-lg");
    addText(slide, "E_f^vac = E_vac(Al107) - (107/108) E_pristine(Al108)", {
      left: 150, top: 416, width: 940, height: 34,
    }, { fontSize: 28, bold: true, color: BLUE, alignment: "center" });
    addText(slide, "This is the formation energy used for both QE and DFTpy comparisons in this package.", {
      left: 122, top: 518, width: 1000, height: 28,
    }, { fontSize: 20, color: MUTED, alignment: "center" });
    footer(slide, 2);
  }

  // Slide 3
  {
    const slide = deck.slides.add();
    slide.background.fill = BG;
    title(slide, "QE vc-relax gives the reference value");
    metric(slide, "Pristine Al108", "-4266.26536555 Ry", "QE vc-relax, JOB DONE", 76, 176, BLUE);
    metric(slide, "Vacancy Al107", "-4226.71609376 Ry", "QE vc-relax, JOB DONE", 468, 176, BLUE);
    metric(slide, "Formation energy", "0.636946 eV", "baseline for DFTpy selection", 860, 176, GREEN);
    addText(slide, "Interpretation", { left: 76, top: 370, width: 240, height: 30 }, {
      fontSize: 24,
      bold: true,
    });
    bullet(slide, "This QE result lies in the literature scale for Al vacancy formation energies.", 76, 416);
    bullet(slide, "It is used here as the numerical target for choosing lambda and mu.", 76, 456);
    bullet(slide, "The same Gillan-style subtraction is applied to DFTpy so the comparison is like-for-like.", 76, 496);
    footer(slide, 3);
  }

  // Slide 4
  {
    const slide = deck.slides.add();
    slide.background.fill = BG;
    title(slide, "The closest DFTpy point is lambda = 0.9, mu = 0.1", "Ranked by absolute difference from the QE vacancy formation energy");
    addCandidateTable(slide, candidates, 92, 166);
    addText(slide, "What this means", { left: 92, top: 442, width: 300, height: 30 }, {
      fontSize: 24,
      bold: true,
    });
    bullet(slide, "Increasing mu strongly raises the vacancy formation energy; the best match is at the lowest sampled mu.", 92, 486);
    bullet(slide, "The closest point is only 0.0247 eV above QE, much smaller than the multi-eV error from uncalibrated TFvW.", 92, 526);
    bullet(slide, "Force residuals are not yet at the strict 0.002 eV/A criterion, so this is the best energy match, not the final production parameter.", 92, 566, 1040, { color: RED, height: 50 });
    footer(slide, 4);
  }

  // Slide 5
  {
    const slide = deck.slides.add();
    slide.background.fill = BG;
    title(slide, "Conclusion and references");
    addText(slide, "Conclusion", { left: 70, top: 160, width: 240, height: 30 }, {
      fontSize: 24,
      bold: true,
    });
    bullet(slide, `Current best coarse-grid parameter: lambda = ${best.lambda.toFixed(1)}, mu = ${best.mu.toFixed(1)}.`, 70, 204);
    bullet(slide, `DFTpy vacancy formation energy is ${best.ef.toFixed(6)} eV, compared with QE ${QE_EF.toFixed(6)} eV.`, 70, 244);
    bullet(slide, "Next step should be a local fine scan / stricter re-relaxation around this region before divacancy work.", 70, 284);
    rect(slide, { left: 70, top: 370, width: 1080, height: 1 }, LINE, LINE);
    addText(slide, "References", { left: 70, top: 398, width: 240, height: 30 }, {
      fontSize: 24,
      bold: true,
    });
    bullet(slide, "M. J. Gillan, J. Phys.: Condens. Matter 1, 689 (1989). Vacancy formation energy in Al; perfect/defect supercell formulation.", 70, 444, 1080, { fontSize: 18, height: 46 });
    bullet(slide, "Gillan reports calculated Al vacancy energy around 0.56 eV and compares with experimental scale around 0.66 eV.", 70, 498, 1080, { fontSize: 18, height: 42 });
    bullet(slide, "This work: QE 7.5 PBE vc-relax reference, Al108/Al107 conventional 3x3x3 supercell, Ef = 0.636946 eV.", 70, 548, 1080, { fontSize: 18, height: 42 });
    footer(slide, 5);
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
