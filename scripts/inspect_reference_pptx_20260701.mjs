import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

async function main() {
  const pptxPath = process.argv[2];
  const outDir = process.argv[3];
  if (!pptxPath || !outDir) {
    throw new Error("Usage: node inspect_reference_pptx_20260701.mjs <pptx> <outDir>");
  }
  await fs.mkdir(outDir, { recursive: true });
  const deck = await PresentationFile.importPptx(await FileBlob.load(pptxPath));
  const inspect = await deck.inspect({
    kind: "slide,textbox,shape,image,chart,table,layout",
    maxChars: 24000,
  });
  await fs.writeFile(path.join(outDir, "inspect.ndjson"), inspect.ndjson, "utf8");
  for (const [index, slide] of deck.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    await writeBlob(
      path.join(outDir, `${stem}.png`),
      await deck.export({ slide, format: "png", scale: 1 }),
    );
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(outDir, `${stem}.layout.json`), await layout.text(), "utf8");
  }
  await writeBlob(
    path.join(outDir, "deck-montage.webp"),
    await deck.export({ format: "webp", montage: true, scale: 1 }),
  );
  console.log(JSON.stringify({ pptxPath, outDir, slides: deck.slides.items.length }, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
