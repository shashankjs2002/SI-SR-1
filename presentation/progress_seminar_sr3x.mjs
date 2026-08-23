import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = path.resolve(SCRIPT_DIR, "..");
const SOURCE_ASSET_DIR = process.env.GEODIFF_SOURCE_ASSET_DIR ||
  path.join(PROJECT_ROOT, "outs");
const BUILD_ROOT = process.env.GEODIFF_PPT_BUILD_ROOT ||
  "F:/codex_geodiff_benchmark_sources_20260619/progress_seminar_build";
const OUTPUT_DIR = path.join(BUILD_ROOT, "output");
const RENDER_DIR = path.join(BUILD_ROOT, "rendered");
const ASSET_DIR = path.join(BUILD_ROOT, "assets");
const FINAL_PPTX = process.env.GEODIFF_FINAL_PPTX ||
  path.join(OUTPUT_DIR, "GeoDiff_GAN_SR3x_Progress_Seminar.pptx");

const W = 1280;
const H = 720;
const C = {
  navy: "#173A5E",
  teal: "#177E7A",
  orange: "#D9772B",
  gray: "#5F6B76",
  dark: "#1F2933",
  mid: "#D9E0E6",
  light: "#F3F6F8",
  paleTeal: "#E8F4F3",
  paleOrange: "#FCF1E8",
  paleNavy: "#EDF2F7",
  white: "#FFFFFF",
};
const FONT = "Aptos";

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

async function readImageBlob(filePath) {
  const bytes = await fs.readFile(filePath);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

function addImage(slide, blob, x, y, w, h, alt, fit = "cover") {
  return slide.images.add({
    blob,
    contentType: "image/webp",
    alt,
    fit,
    position: { left: x, top: y, width: w, height: h },
  });
}

function addText(slide, text, x, y, w, h, size = 20, color = C.dark, options = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    name: options.name,
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    fontFamily: FONT,
    fontSize: size,
    color,
    bold: Boolean(options.bold),
    italic: Boolean(options.italic),
    alignment: options.align || "left",
  };
  return shape;
}

function addRect(slide, x, y, w, h, fill = C.white, line = C.mid, width = 1, name) {
  return slide.shapes.add({
    geometry: "rect",
    name,
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { style: "solid", fill: line, width },
  });
}

function addCircle(slide, x, y, diameter, fill = C.white, line = C.navy, width = 2) {
  return slide.shapes.add({
    geometry: "ellipse",
    position: { left: x, top: y, width: diameter, height: diameter },
    fill,
    line: { style: "solid", fill: line, width },
  });
}

function addRule(slide, x, y, w, color = C.mid, thickness = 2) {
  return addRect(slide, x, y, w, thickness, color, color, 0);
}

function addArrow(slide, x, y, w, h = 18, color = C.gray) {
  return slide.shapes.add({
    geometry: "rightArrow",
    position: { left: x, top: y, width: w, height: h },
    fill: color,
    line: { style: "solid", fill: color, width: 0 },
  });
}

function addDownArrow(slide, x, y, w = 18, h = 44, color = C.gray) {
  return slide.shapes.add({
    geometry: "downArrow",
    position: { left: x, top: y, width: w, height: h },
    fill: color,
    line: { style: "solid", fill: color, width: 0 },
  });
}

function addUpArrow(slide, x, y, w = 18, h = 44, color = C.gray) {
  return slide.shapes.add({
    geometry: "upArrow",
    position: { left: x, top: y, width: w, height: h },
    fill: color,
    line: { style: "solid", fill: color, width: 0 },
  });
}

function addSlide(presentation, title, section, number, subtitle = "") {
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  addText(slide, section.toUpperCase(), 64, 27, 280, 22, 14, C.teal, { bold: true });
  addText(slide, title, 64, 54, 1152, 52, 36, C.navy, { bold: true, name: "slide-title" });
  if (subtitle) addText(slide, subtitle, 64, 108, 1120, 30, 18, C.gray);
  addRule(slide, 64, 680, 1152, C.mid, 1);
  addText(slide, "GeoDiff-GAN | SR-3x", 64, 687, 260, 20, 12, C.gray);
  addText(slide, String(number).padStart(2, "0"), 1164, 687, 52, 20, 12, C.gray, { align: "right" });
  return slide;
}

function setNotes(slide, text, sources = []) {
  const sourceBlock = sources.length
    ? `\n\n[Sources]\n${sources.map((source) => `- ${source}`).join("\n")}`
    : "\n\n[Sources]\n- Project SR-3x source code and researcher-authored experiment notes.";
  slide.speakerNotes.textFrame.setText(`${text}${sourceBlock}`);
  slide.speakerNotes.setVisible(true);
}

function addBullets(slide, bullets, x = 86, y = 168, w = 1080, gap = 66, size = 22, accent = C.teal) {
  bullets.forEach((bullet, index) => {
    const yy = y + index * gap;
    addRect(slide, x, yy + 8, 10, 10, accent, accent, 0);
    addText(slide, bullet, x + 26, yy, w - 26, gap - 8, size, C.dark);
  });
}

function addCallout(slide, text, x, y, w, h, color = C.orange, size = 22) {
  addRect(slide, x, y, 5, h, color, color, 0);
  addText(slide, text, x + 20, y + 2, w - 20, h - 4, size, C.dark, { bold: true });
}

function addNode(slide, x, y, w, h, title, detail, palette = "teal", name) {
  const styles = {
    teal: { line: C.teal },
    navy: { line: C.navy },
    orange: { line: C.orange },
    gray: { line: C.gray },
  };
  const style = styles[palette];
  const box = addRect(slide, x, y, w, h, C.white, C.dark, 1, name);
  addRule(slide, x, y, w, style.line, 4);
  const multilineTitle = title.includes("\n") || title.length > 17;
  const titleHeight = multilineTitle ? 46 : 28;
  const titleSize = multilineTitle ? 18 : 20;
  addText(slide, title, x + 12, y + 12, w - 24, titleHeight, titleSize, style.line, { bold: true, align: "center" });
  if (detail) {
    const detailTop = y + 20 + titleHeight;
    addText(slide, detail, x + 10, detailTop, w - 20, h - (detailTop - y) - 10, 16, C.gray, { align: "center" });
  }
  return box;
}

function addEdgeLabel(slide, text, x, y, w, color = C.gray) {
  addText(slide, text, x, y, w, 22, 14, color, { align: "center", italic: true });
}

function addFigureLabel(slide, label, caption, x, y, w, color = C.dark) {
  addText(slide, label, x, y, 48, 24, 17, color, { bold: true });
  addText(slide, caption, x + 48, y, w - 48, 54, 17, color);
}

function addColumnHeader(slide, title, x, y, w, color = C.navy) {
  addText(slide, title, x, y, w, 32, 24, color, { bold: true });
  addRule(slide, x, y + 36, w, color, 3);
}

function addTwoColumnBullets(slide, leftTitle, leftBullets, rightTitle, rightBullets, options = {}) {
  const x1 = 72;
  const x2 = 664;
  const w = 544;
  addColumnHeader(slide, leftTitle, x1, 160, w, options.leftColor || C.navy);
  addColumnHeader(slide, rightTitle, x2, 160, w, options.rightColor || C.teal);
  addBullets(slide, leftBullets, x1 + 8, 220, w - 16, 70, 20, options.leftColor || C.navy);
  addBullets(slide, rightBullets, x2 + 8, 220, w - 16, 70, 20, options.rightColor || C.teal);
}

function addSimpleTable(slide, headers, rows, x, y, widths, rowHeight = 52, fontSize = 17) {
  const total = widths.reduce((sum, value) => sum + value, 0);
  addRect(slide, x, y, total, rowHeight, C.navy, C.navy, 0);
  let xx = x;
  headers.forEach((header, index) => {
    addText(slide, header, xx + 8, y + 12, widths[index] - 16, rowHeight - 16, fontSize, C.white, { bold: true });
    xx += widths[index];
  });
  rows.forEach((row, rowIndex) => {
    const yy = y + rowHeight * (rowIndex + 1);
    addRect(slide, x, yy, total, rowHeight, rowIndex % 2 ? C.light : C.white, C.mid, 1);
    let cellX = x;
    row.forEach((cell, columnIndex) => {
      addText(slide, String(cell), cellX + 8, yy + 11, widths[columnIndex] - 16, rowHeight - 14, fontSize, C.dark, { bold: columnIndex === 0 });
      cellX += widths[columnIndex];
    });
  });
}

function addMetricBar(slide, label, valueText, fraction, x, y, w, color = C.teal) {
  addText(slide, label, x, y, 170, 26, 18, C.dark, { bold: true });
  addRect(slide, x + 180, y + 4, w - 350, 18, C.light, C.light, 0);
  addRect(slide, x + 180, y + 4, Math.max(4, (w - 350) * fraction), 18, color, color, 0);
  addText(slide, valueText, x + w - 150, y, 150, 26, 18, color, { bold: true, align: "right" });
}

function titleSlide(presentation, assets) {
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  addText(slide, "PROGRESS SEMINAR", 72, 60, 260, 24, 15, C.teal, { bold: true });
  addText(slide, "GeoDiff-GAN SR-3x", 72, 116, 680, 80, 52, C.navy, { bold: true });
  addText(slide, "Landsat 30 m to Sentinel-2 10 m super-resolution", 74, 226, 680, 56, 25, C.gray);
  addCallout(slide, "Progress seminar: dataset, model design and current results", 74, 326, 680, 48, C.orange, 22);

  // Real project imagery replaces the abstract pixel-grid figure.
  const imageY = 164;
  const imageSize = 126;
  const imageXs = [806, 962, 1118];
  addArrow(slide, 936, 216, 22, 14, C.gray);
  addArrow(slide, 1092, 216, 22, 14, C.gray);
  addRect(slide, imageXs[0] - 3, imageY - 3, imageSize + 6, imageSize + 6, C.white, C.orange, 2);
  addRect(slide, imageXs[1] - 3, imageY - 3, imageSize + 6, imageSize + 6, C.white, C.navy, 2);
  addRect(slide, imageXs[2] - 3, imageY - 3, imageSize + 6, imageSize + 6, C.white, C.teal, 2);
  addImage(slide, assets.landsat, imageXs[0], imageY, imageSize, imageSize, "Landsat 30 metre input patch");
  addImage(slide, assets.inferred, imageXs[1], imageY, imageSize, imageSize, "GeoDiff-GAN inferred 10 metre patch");
  addImage(slide, assets.sentinel, imageXs[2], imageY, imageSize, imageSize, "Sentinel-2 10 metre reference patch");
  addText(slide, "Landsat\n30 m input", imageXs[0] - 4, 302, imageSize + 8, 48, 15, C.orange, { bold: true, align: "center" });
  addText(slide, "GeoDiff-GAN\n10 m estimate", imageXs[1] - 8, 302, imageSize + 16, 48, 15, C.navy, { bold: true, align: "center" });
  addText(slide, "Sentinel-2\n10 m reference", imageXs[2] - 8, 302, imageSize + 16, 48, 15, C.teal, { bold: true, align: "center" });

  addText(slide, "[Your Name]  |  [Enrollment No.]", 74, 556, 560, 28, 18, C.dark, { bold: true });
  addText(slide, "[Department]  |  [Supervisor]  |  [Date]", 74, 594, 680, 26, 17, C.gray);
  addRule(slide, 74, 662, 1132, C.navy, 3);
  setNotes(slide,
    "Open with the scientific objective: improve spatial detail without silently violating the Landsat measurement. State that the output is a Sentinel-like estimate, not a new physical observation.",
    [
      "https://www.usgs.gov/landsat-missions/landsat-8",
      "https://documentation.dataspace.copernicus.eu/Data/SentinelMissions/Sentinel2.html",
      "Project assets: outs/landsat_30m.webp, outs/infered_10m.webp, outs/sentinel_10m.webp",
    ]);
}

function buildDeck(assets) {
  const p = Presentation.create({ slideSize: { width: W, height: H } });
  titleSlide(p, assets);

  // 2. One pixel.
  {
    const s = addSlide(p, "Why 3x super-resolution is an ill-posed problem", "Context", 2);
    addText(s, "Landsat input | 30 m", 132, 144, 380, 30, 22, C.orange, { bold: true, align: "center" });
    addText(s, "Sentinel reference | 10 m", 768, 144, 380, 30, 22, C.teal, { bold: true, align: "center" });
    addRect(s, 137, 181, 370, 370, C.white, C.orange, 2);
    addRect(s, 773, 181, 370, 370, C.white, C.teal, 2);
    addImage(s, assets.landsat, 142, 186, 360, 360, "Landsat 30 metre image over the same geographic footprint");
    addImage(s, assets.sentinel, 778, 186, 360, 360, "Sentinel-2 10 metre image over the same geographic footprint");

    // A corresponding ground region is one coarse cell versus a 3x3 fine grid.
    const coarseX = 262;
    const coarseY = 306;
    const region = 120;
    addRect(s, coarseX, coarseY, region, region, "none", C.orange, 4);
    addArrow(s, 566, 350, 142, 24, C.gray);
    addText(s, "same ground area", 556, 314, 164, 28, 16, C.gray, { align: "center" });
    const fineX = 898;
    const fineY = 306;
    const fineCell = region / 3;
    for (let row = 0; row < 3; row += 1) {
      for (let col = 0; col < 3; col += 1) {
        addRect(s, fineX + col * fineCell, fineY + row * fineCell, fineCell, fineCell, "none", C.teal, 2);
      }
    }
    addText(s, "1 Landsat sample", 240, 556, 164, 28, 17, C.orange, { bold: true, align: "center" });
    addText(s, "9 Sentinel samples", 876, 556, 164, 28, 17, C.teal, { bold: true, align: "center" });
    addCallout(s, "One 30 m measurement constrains, but does not determine, the nine 10 m values.", 156, 608, 968, 44, C.orange, 20);
    setNotes(s, "Both panels cover the same geographic footprint. The boxes explain sampling density, not physical display size. Expected answer: one coarse measurement cannot uniquely determine nine fine measurements; the inverse problem is ill-posed.", [
      "https://www.usgs.gov/landsat-missions/landsat-8",
      "Project assets: outs/landsat_30m.webp and outs/sentinel_10m.webp",
    ]);
  }

  // 3. Spatial resolution.
  {
    const s = addSlide(p, "Spatial sampling: Landsat 30 m and Sentinel-2 10 m", "Context", 3);
    addColumnHeader(s, "Landsat 8/9 OLI", 98, 176, 440, C.orange);
    addText(s, "30 m", 98, 236, 440, 80, 52, C.orange, { bold: true });
    addText(s, "RGB, NIR and SWIR bands used here", 98, 330, 440, 48, 21, C.gray);
    addColumnHeader(s, "Sentinel-2 MSI", 706, 176, 440, C.teal);
    addText(s, "10 m", 706, 236, 440, 80, 52, C.teal, { bold: true });
    addText(s, "RGB reference bands B4 / B3 / B2", 706, 330, 440, 48, 21, C.gray);
    addRule(s, 622, 176, 2, C.mid, 360);
    addCallout(s, "A smaller pixel spacing improves spatial sampling only. Spectral, radiometric and temporal resolution are separate properties.", 130, 500, 1010, 66, C.navy, 21);
    setNotes(s, "Use this slide to separate spatial resolution from the other three resolution dimensions. The model changes the output sampling grid; it does not upgrade the sensor itself.", [
      "https://www.usgs.gov/landsat-missions/landsat-8",
      "https://documentation.dataspace.copernicus.eu/Data/SentinelMissions/Sentinel2.html",
    ]);
  }

  // 4. SR definition.
  {
    const s = addSlide(p, "Super-resolution and interpolation are not the same", "Context", 4);
    addArrow(s, 330, 292, 110, 22, C.gray);
    addArrow(s, 750, 292, 110, 22, C.gray);
    addNode(s, 100, 225, 220, 160, "LR measurement", "3 x 128 x 128\nactual Landsat", "orange");
    addNode(s, 452, 225, 286, 160, "Learned prior", "patterns from aligned\ntraining pairs", "navy");
    addNode(s, 870, 225, 290, 160, "HR estimate", "3 x 384 x 384\nSentinel-like RGB", "teal");
    addText(s, "Bicubic", 100, 170, 220, 32, 23, C.gray, { bold: true, align: "center" });
    addText(s, "Learned SR", 556, 170, 520, 32, 23, C.teal, { bold: true, align: "center" });
    addCallout(s, "Bicubic changes the array size but adds no measured detail.", 122, 450, 470, 48, C.gray, 19);
    addCallout(s, "Learned SR estimates detail from training data and must be tested.", 680, 450, 480, 48, C.teal, 19);
    setNotes(s, "Emphasize that array enlargement is not equivalent to information recovery. The model output remains an estimate even when it is visually sharper.", ["https://arxiv.org/abs/2108.10257"]);
  }

  // 5. Why SR.
  {
    const s = addSlide(p, "When is satellite image super-resolution useful?", "Context", 5);
    addBullets(s, [
      "Historical continuity: Landsat provides a long archive for retrospective studies.",
      "Clouds, acquisition gaps, and timing can remove the desired Sentinel observation.",
      "Commercial very-high-resolution imagery may be restricted or expensive.",
      "A consistent lower-resolution record may exist when the higher-resolution record does not.",
    ], 90, 160, 1080, 82, 21, C.teal);
    addCallout(s, "Use the observed Sentinel-2 image whenever a valid scene is available for the required place and date.", 140, 542, 1000, 54, C.orange, 22);
    setNotes(s, "Ask the audience what should be preferred when a matching Sentinel image exists. The answer is the real measurement. SR addresses missing, cloudy, costly, or historical cases.", [
      "https://www.usgs.gov/landsat-missions/landsat-surface-reflectance",
      "https://documentation.dataspace.copernicus.eu/Data/SentinelMissions/Sentinel2.html",
    ]);
  }

  // 6. Applications.
  {
    const s = addSlide(p, "Intended use and limitations", "Context", 6);
    addTwoColumnBullets(s,
      "Potential uses",
      ["Change screening", "Field and urban boundary interpretation", "Coastline and water-body delineation", "Downstream research inputs"],
      "Do not use alone for",
      ["Legal boundaries", "Property disputes", "Navigation", "Safety-critical decisions"],
      { leftColor: C.teal, rightColor: C.orange });
    setNotes(s, "A generated SR product may help an analyst prioritize or inspect areas, but important decisions require independent measurements and provenance.");
  }

  // 7. Can / cannot.
  {
    const s = addSlide(p, "What the model can and cannot estimate", "Context", 7);
    addTwoColumnBullets(s,
      "Can estimate",
      ["Likely high-frequency structure", "Similarity to paired references", "Uncertainty across samples", "Approximate LR evidence consistency"],
      "Cannot guarantee",
      ["Every generated edge is real", "Information measured nowhere", "Correction of bad registration", "Zero uncertainty"],
      { leftColor: C.teal, rightColor: C.orange });
    setNotes(s, "Frame the scientific test around held-out fidelity, consistency, generalization, and uncertainty, not visual sharpness alone.");
  }

  // 8. Research problem.
  {
    const s = addSlide(p, "Research objective", "Problem", 8);
    const imageY = 164;
    const imageSize = 236;
    const imageXs = [94, 522, 950];
    addImage(s, assets.landsat, imageXs[0], imageY, imageSize, imageSize, "Native Landsat 30 metre input patch");
    addImage(s, assets.inferred, imageXs[1], imageY, imageSize, imageSize, "GeoDiff-GAN 3x output patch");
    addImage(s, assets.sentinel, imageXs[2], imageY, imageSize, imageSize, "Sentinel-2 10 metre paired reference patch");
    addArrow(s, 364, 274, 112, 12, C.dark);
    addArrow(s, 792, 274, 112, 12, C.dark);
    addEdgeLabel(s, "f_theta(y_L, optional MS)", 344, 244, 152, C.navy);
    addEdgeLabel(s, "paired evaluation", 786, 244, 124, C.teal);
    addFigureLabel(s, "(a)", "Landsat L2 RGB\n30 m | 3 x 128 x 128", imageXs[0], 414, imageSize, C.orange);
    addFigureLabel(s, "(b)", "GeoDiff-GAN estimate\n10 m | 3 x 384 x 384", imageXs[1], 414, imageSize, C.navy);
    addFigureLabel(s, "(c)", "Sentinel-2 L2A reference\n10 m | 3 x 384 x 384", imageXs[2], 414, imageSize, C.teal);
    addRule(s, 94, 494, 1092, C.mid, 1);
    addText(s, "Task definition", 94, 516, 172, 26, 20, C.navy, { bold: true });
    addText(s, "Estimate the paired Sentinel-like RGB field from the co-registered Landsat measurement; the reference is a different sensor observation, not hidden Landsat truth.", 270, 510, 916, 54, 19, C.dark);
    addCallout(s, "Success criterion: the complete model must improve test PSNR and SSIM over its own SwinIR base.", 164, 592, 952, 42, C.orange, 19);
    setNotes(s, "This is not synthetic downsampling. Landsat and Sentinel are different observations from different instruments, so sensor mismatch becomes part of the learning problem.", [
      "https://www.usgs.gov/landsat-missions/landsat-surface-reflectance",
      "https://documentation.dataspace.copernicus.eu/Data/SentinelMissions/Sentinel2.html",
      "Project assets: outs/landsat_30m.webp, outs/infered_10m.webp, outs/sentinel_10m.webp",
    ]);
  }

  // 9. Evolution.
  {
    const s = addSlide(p, "Development of the experiment", "Problem", 9);
    const centers = [248, 640, 1032];
    const colors = [C.gray, C.orange, C.teal];

    // Connectors first: a single evidence-maturity axis, not three UI cards.
    addArrow(s, 138, 224, 1014, 12, C.mid);
    addRect(s, 438, 146, 1, 426, C.mid, C.mid, 0);
    addRect(s, 830, 146, 1, 426, C.mid, C.mid, 0);

    const phases = [
      {
        title: "Synthetic validation",
        question: "Does the 4x system train end to end?",
        data: "Sentinel 40 m -> 10 m",
        result: "Established architecture, staged training and diagnostics.",
      },
      {
        title: "Real cross-sensor task",
        question: "Can Landsat 30 m predict Sentinel 10 m?",
        data: "Landsat L2 -> Sentinel-2 L2A | 3x",
        result: "Exposed registration, temporal and radiometric mismatch.",
      },
      {
        title: "Controlled harmonization",
        question: "Can calibrated RGB and MS guidance beat the base?",
        data: "RGB base + NIR/SWIR residual guidance",
        result: "Current test: held-out PSNR/SSIM and intervention ablations.",
      },
    ];

    phases.forEach((phase, index) => {
      const x = 72 + index * 392;
      addText(s, `Phase ${index + 1}`, x, 146, 352, 24, 17, colors[index], { bold: true, align: "center" });
      addText(s, phase.title, x, 174, 352, 34, 23, C.navy, { bold: true, align: "center" });
      addCircle(s, centers[index] - 25, 204, 50, C.white, colors[index], 3);
      addText(s, String(index + 1), centers[index] - 25, 214, 50, 28, 20, colors[index], { bold: true, align: "center" });

      addText(s, "Aim", x + 24, 286, 304, 22, 16, colors[index], { bold: true });
      addText(s, phase.question, x + 24, 314, 304, 56, 19, C.dark);
      addRule(s, x + 24, 382, 304, C.mid, 1);
      addText(s, "Data", x + 24, 398, 304, 22, 16, colors[index], { bold: true });
      addText(s, phase.data, x + 24, 426, 304, 48, 18, C.dark);
      addText(s, "Finding", x + 24, 486, 304, 22, 16, colors[index], { bold: true });
      addText(s, phase.result, x + 24, 514, 304, 58, 18, C.dark);
    });

    addCallout(s, "Each phase used more realistic data and exposed a different source of error.", 130, 606, 1020, 42, C.orange, 18);
    setNotes(s, "Explain that the current SR-3x branch was cut to address the supervisor-directed real Landsat-to-Sentinel problem. Each phase changes the scientific question and the evidence required, rather than merely increasing model size.");
  }

  // 10. Cross-sensor dimensions.
  {
    const s = addSlide(p, "Sources of Landsat-Sentinel mismatch", "Data", 10);
    addText(s, "Sensor observation", 78, 142, 176, 24, 17, C.teal, { bold: true });
    addText(s, "y_s = Q_s { P_s * sum_lambda R_s(lambda) L(x, lambda, t) } + e_s", 256, 138, 946, 32, 20, C.navy, { bold: true });
    addRule(s, 78, 180, 1124, C.mid, 1);
    addText(s, "Observed pair difference", 78, 196, 220, 24, 17, C.orange, { bold: true });
    addText(s, "d_obs = y_S2 - W(y_L8) = d_sp + d_spec + d_rad + d_time + e", 300, 190, 902, 38, 20, C.dark, { bold: true });

    const xs = [72, 364, 656, 948];
    const colors = [C.navy, C.teal, C.orange, C.gray];
    const factors = [
      ["d_spatial", "PSF, sampling grid and residual registration", "Road/river edge displacement; sCC", "Co-registration, masks and guard bands"],
      ["d_spectral", "Different relative spectral response functions", "Band ratios and spectral-angle error", "Band-aware inputs; avoid false channel equivalence"],
      ["d_radiometric", "Calibration, atmosphere, BRDF and illumination", "Mean/variance drift; ERGAS and SAM", "Surface-reflectance scaling and train-only harmonization"],
      ["d_temporal", "Land-cover or atmospheric change between dates", "Localized coherent change despite alignment", "Date-gap threshold, change audit and exclusion"],
    ];

    // Scientific decomposition columns: ruled sections instead of colored cards.
    [354, 646, 938].forEach((x) => addRect(s, x, 254, 1, 310, C.mid, C.mid, 0));
    factors.forEach((factor, index) => {
      const x = xs[index];
      addRule(s, x, 252, 248, colors[index], 4);
      addText(s, factor[0], x, 270, 248, 30, 22, colors[index], { bold: true });
      addText(s, "Physical origin", x, 320, 248, 20, 15, C.gray, { bold: true });
      addText(s, factor[1], x, 346, 248, 58, 18, C.dark);
      addText(s, "Observable signature", x, 422, 248, 20, 15, C.gray, { bold: true });
      addText(s, factor[2], x, 448, 248, 58, 18, C.dark);
      addText(s, "Experimental control", x, 524, 248, 20, 15, C.gray, { bold: true });
      addText(s, factor[3], x, 550, 248, 56, 18, C.dark);
    });
    addCallout(s, "A larger model may fit all four terms. Controlled experiments are needed to identify real spatial detail.", 172, 622, 936, 34, C.orange, 18);
    setNotes(s, "Use roads and riverbanks as alignment examples. Explain that a mismatch may be sensor color response, haze, change, or displacement rather than missing spatial detail.", [
      "https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf",
      "https://www.usgs.gov/landsat-missions/landsat-8",
      "https://documentation.dataspace.copernicus.eu/Data/SentinelMissions/Sentinel2.html",
    ]);
  }

  // 11. Questions.
  {
    const s = addSlide(p, "Research questions", "Problem", 11);
    addBullets(s, [
      "Does the final model beat bicubic and the exact embedded SwinIR base?",
      "Does diffusion improve measured detail, or only alter texture?",
      "Can evidence confidence suppress unsupported residuals?",
      "Does train-only radiometric harmonization reduce domain gap?",
      "Can NIR/SWIR guide the residual without shifting base RGB?",
      "Do gains survive spatial and complete-tile holdout tests?",
    ], 88, 142, 1110, 73, 20, C.teal);
    setNotes(s, "State the hypothesis: a conservative RGB base plus evidence-gated multispectral residual guidance should be safer than unrestricted full-image generation.");
  }

  // 12. Bands.
  {
    const s = addSlide(p, "Input and target bands", "Data", 12);
    addSimpleTable(s,
      ["Role", "Product", "Bands", "Resolution"],
      [
        ["LR RGB", "Landsat 8/9 C2 L2", "B4 / B3 / B2", "30 m"],
        ["Residual guidance", "Landsat 8/9 C2 L2", "B5 / B6 / B7", "30 m"],
        ["HR reference", "Sentinel-2 L2A", "B4 / B3 / B2", "10 m"],
        ["Quality", "Both products", "QA / SCL / no-data", "native"],
      ], 78, 170, [210, 350, 330, 190], 62, 18);
    addCallout(s, "The multispectral model uses six input bands but still predicts a three-channel RGB image.", 190, 542, 900, 48, C.teal, 22);
    setNotes(s, "Surface reflectance is used instead of display RGB or raw digital numbers. Auxiliary bands provide material and moisture cues but are never presented as output colors.", [
      "https://www.usgs.gov/landsat-missions/landsat-8",
      "https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S2L2A.html",
    ]);
  }

  // 13. Pair creation.
  {
    const s = addSlide(p, "Landsat-Sentinel pair preparation", "Data", 13);
    const nodes = [
      ["Discover", "complete products"], ["Match", "overlap + date"], ["Scale", "surface reflectance"],
      ["Reproject", "to paired grids"], ["Mask", "joint valid support"], ["Extract", "NPZ + manifest"],
    ];
    for (let i = 0; i < 5; i += 1) addArrow(s, 244 + i * 194, 280, 54, 16, C.gray);
    nodes.forEach((node, index) => addNode(s, 72 + index * 194, 228, 164, 126, node[0], node[1], index < 3 ? "navy" : "teal"));
    addText(s, "Key settings", 88, 430, 180, 30, 24, C.navy, { bold: true });
    addText(s, "max date gap: 3 days", 88, 476, 300, 28, 20, C.dark);
    addText(s, "HR: 384 x 384 | stride 288", 430, 476, 340, 28, 20, C.dark);
    addText(s, "LR: 128 x 128 | valid >= 95%", 830, 476, 360, 28, 20, C.dark);
    addCallout(s, "The manifest stores product IDs, date gap, transforms, masks and dataset split for every pair.", 150, 554, 980, 46, C.orange, 21);
    setNotes(s, "Walk through the processing sequence. Stress that a patch is not just two images; it is a paired measurement with provenance and validity support.", [
      "https://www.usgs.gov/landsat-missions/landsat-surface-reflectance",
      "https://documentation.dataspace.copernicus.eu/Data/SentinelMissions/Sentinel2.html",
    ]);
  }

  // 14. Equal footprint.
  {
    const s = addSlide(p, "Patch dimensions and ground footprint", "Data", 14);
    addText(s, "128 pixels x 30 m", 118, 198, 420, 48, 30, C.orange, { bold: true, align: "center" });
    addText(s, "= 3840 m", 118, 264, 420, 54, 38, C.navy, { bold: true, align: "center" });
    addText(s, "384 pixels x 10 m", 742, 198, 420, 48, 30, C.teal, { bold: true, align: "center" });
    addText(s, "= 3840 m", 742, 264, 420, 54, 38, C.navy, { bold: true, align: "center" });
    addRule(s, 632, 182, 2, C.mid, 200);
    addCallout(s, "The plots use equal display sizes, but the native sample counts differ. Bicubic resizing adds no new measurement.", 134, 448, 1012, 62, C.orange, 22);
    addText(s, "Audience question: why can the 128-pixel image look physically as large as the 384-pixel image?", 156, 548, 968, 42, 20, C.gray, { italic: true, align: "center" });
    setNotes(s, "The plotting library scales both arrays to the same display box. Native sample count and ground sampling distance do not change.");
  }

  // 15. QA.
  {
    const s = addSlide(p, "Quality control and rejected pairs", "Data", 15);
    addTwoColumnBullets(s,
      "Automatic rejection",
      ["cloud / cirrus / shadow / snow", "no-data and black borders", "radiometric saturation", "valid support below 95%"],
      "Manual pair audit",
      ["road and river displacement", "coastline misalignment", "large temporal change", "unexpected color response"],
      { leftColor: C.orange, rightColor: C.teal });
    addCallout(s, "A high valid-pixel fraction does not guarantee accurate registration.", 210, 558, 860, 44, C.navy, 22);
    setNotes(s, "Rejected samples are quarantined with a reason rather than silently deleted. This preserves an audit trail and supports later threshold tuning.", [
      "https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S2L2A.html",
    ]);
  }

  // 16. Splits.
  {
    const s = addSlide(p, "Train, validation and test split", "Data", 16);
    const x = 104, y = 205, totalW = 1070, h = 170;
    addRect(s, x, y, 642, h, C.paleNavy, C.navy, 2);
    addRect(s, x + 642, y, 214, h, C.paleTeal, C.teal, 2);
    addRect(s, x + 856, y, 214, h, C.paleOrange, C.orange, 2);
    addText(s, "TRAIN", x, y + 62, 642, 42, 28, C.navy, { bold: true, align: "center" });
    addText(s, "VAL", x + 642, y + 62, 214, 42, 28, C.teal, { bold: true, align: "center" });
    addText(s, "TEST", x + 856, y + 62, 214, 42, 28, C.orange, { bold: true, align: "center" });
    addRect(s, x + 630, y - 12, 24, h + 24, C.white, C.gray, 1);
    addRect(s, x + 844, y - 12, 24, h + 24, C.white, C.gray, 1);
    addText(s, "guard bands", 492, 400, 300, 28, 18, C.gray, { align: "center" });
    addBullets(s, [
      "Development: spatial blocks within every tile provide val/test even with few tiles.",
      "Final claim: complete-city or complete-tile holdout and spatial K-fold evaluation.",
    ], 124, 472, 1030, 66, 21, C.teal);
    setNotes(s, "Random patch splitting is invalid because nearby windows overlap and share textures. Within-tile splits are for development; full tile holdout is the stronger generalization test.");
  }

  // 17. Baselines.
  {
    const s = addSlide(p, "Baselines and acceptance criterion", "Evaluation", 17);
    addText(s, "Evaluation ladder", 88, 160, 300, 34, 24, C.navy, { bold: true });
    const labels = ["Bicubic", "SwinIR base", "RGB GeoDiff-GAN", "Multispectral guided", "External SR methods"];
    labels.forEach((label, i) => {
      addRect(s, 104 + i * 214, 236, 182, 118, i === 1 ? C.paleOrange : i >= 2 ? C.paleTeal : C.light, i === 1 ? C.orange : i >= 2 ? C.teal : C.gray, 2);
      addText(s, label, 116 + i * 214, 272, 158, 48, 20, C.dark, { bold: true, align: "center" });
    });
    addCallout(s, "All methods are evaluated on the same pairs, masks, splits, indices and metric code.", 152, 444, 976, 52, C.navy, 22);
    addText(s, "Safety baseline", 344, 376, 180, 26, 17, C.orange, { bold: true, align: "center" });
    setNotes(s, "SwinIR is both an internal branch and a baseline. A final output below the base in held-out PSNR/SSIM is a regression, even if it looks sharper.", ["https://arxiv.org/abs/2108.10257"]);
  }

  // 18. Current architecture overview.
  {
    const s = addSlide(p, "GeoDiff-GAN SR-3x architecture", "Model", 18);
    addText(s, "(a) Deterministic reconstruction path", 72, 140, 420, 24, 18, C.navy, { bold: true });
    addText(s, "(b) Conditional residual path", 72, 314, 420, 24, 18, C.teal, { bold: true });
    addText(s, "training supervision", 72, 504, 250, 22, 16, C.gray, { bold: true });

    // Connectors are created before modules so they remain visually behind them.
    addArrow(s, 240, 214, 72, 10, C.dark);
    addArrow(s, 492, 214, 72, 10, C.dark);
    addArrow(s, 754, 214, 240, 10, C.dark);
    addArrow(s, 220, 394, 48, 10, C.dark);
    addArrow(s, 428, 394, 38, 10, C.dark);
    addArrow(s, 636, 394, 38, 10, C.dark);
    addArrow(s, 844, 394, 38, 10, C.dark);
    addArrow(s, 1052, 394, 38, 10, C.dark);
    addUpArrow(s, 1108, 258, 12, 90, C.gray);
    addArrow(s, 220, 560, 32, 9, C.gray);
    addArrow(s, 412, 560, 32, 9, C.gray);
    addArrow(s, 604, 560, 32, 9, C.gray);
    addUpArrow(s, 516, 438, 12, 88, C.gray);

    addNode(s, 72, 174, 160, 84, "Landsat RGB", "3 x 128 x 128", "orange");
    addNode(s, 320, 174, 164, 84, "SwinIR base", "RGB-only | 3x", "navy");
    addNode(s, 572, 174, 174, 84, "Base estimate", "3 x 384 x 384", "navy");
    addNode(s, 1002, 174, 190, 84, "Merge + clip", "3 x 384 x 384", "teal");

    addNode(s, 72, 354, 140, 84, "Landsat MS", "6 x 128 x 128", "orange");
    addNode(s, 276, 354, 144, 84, "LR encoder", "4 scales", "teal");
    addNode(s, 474, 354, 154, 84, "Diffusion", "4 @ 48^2", "navy");
    addNode(s, 682, 354, 154, 84, "GeoMapper", "3 outputs", "teal");
    addNode(s, 890, 354, 154, 84, "SR decoder", "residual at 384^2", "orange");
    addNode(s, 1098, 354, 150, 84, "Gate + HPF", "alpha * c * HPF", "teal");

    addNode(s, 72, 526, 140, 72, "Sentinel HR", "3 x 384 x 384", "gray");
    addNode(s, 260, 526, 144, 72, "r_target", "x_S2 - x_base", "orange");
    addNode(s, 452, 526, 144, 72, "VAE encode", "latent moments", "navy");
    addNode(s, 644, 526, 144, 72, "z_0", "4 x 48 x 48", "teal");

    addText(s, "x_hat = clip[x_base + alpha c .* HP(r_hat), 0, 1]", 816, 540, 418, 32, 18, C.navy, { bold: true, align: "center" });
    addCallout(s, "Low-frequency RGB comes from the base branch. The residual is high-pass filtered and confidence weighted.", 168, 620, 944, 36, C.orange, 18);
    setNotes(s, "Explain the two responsibility paths. The top path reconstructs measured low-frequency structure. The bottom path proposes high-frequency residual detail using all six Landsat bands. The residual is gated and can be rejected.", [
      "Project source: src/geodiff_gan/models/system.py on branch SR-3x",
      "https://arxiv.org/abs/2108.10257",
      "https://arxiv.org/abs/2410.10812",
      "https://arxiv.org/abs/2405.04356",
    ]);
  }

  // 19. Base detailed.
  {
    const s = addSlide(p, "SwinIR base branch", "Model", 19);
    addText(s, "SwinIR-style deterministic branch", 72, 144, 420, 24, 18, C.navy, { bold: true });
    const xs = [64, 244, 438, 658, 858, 1064];
    const widths = [132, 146, 166, 152, 154, 148];
    for (let i = 0; i < xs.length - 1; i += 1) addArrow(s, xs[i] + widths[i] + 8, 270, xs[i + 1] - (xs[i] + widths[i]) - 16, 10, C.dark);
    // Feature-space residual connection.
    addRule(s, 314, 190, 424, C.gray, 2);
    addDownArrow(s, 306, 190, 12, 28, C.gray);
    addDownArrow(s, 730, 190, 12, 28, C.gray);
    // Image-space bicubic anchor.
    addRule(s, 130, 376, 862, C.gray, 2);
    addDownArrow(s, 122, 320, 12, 58, C.gray);
    addArrow(s, 990, 371, 62, 10, C.gray);

    addNode(s, xs[0], 226, widths[0], 96, "RGB input", "3 x 128 x 128", "orange");
    addNode(s, xs[1], 226, widths[1], 96, "3 x 3 conv", "C = 32", "navy");
    addNode(s, xs[2], 214, widths[2], 120, "Swin body", "4 RSTB | 8 x 8 windows\n4 attention heads", "navy");
    addNode(s, xs[3], 226, widths[3], 96, "Body conv", "+ feature residual", "navy");
    addNode(s, xs[4], 214, widths[4], 120, "Resize-conv", "bilinear 3x\n3 x 3 refinement", "teal");
    addNode(s, xs[5], 226, widths[5], 96, "x_base", "3 x 384 x 384", "teal");
    addEdgeLabel(s, "long feature skip", 438, 164, 190);
    addEdgeLabel(s, "bicubic RGB anchor", 482, 384, 190);
    addText(s, "x_base = bicubic(y_RGB) + B_theta(y_RGB)", 300, 440, 680, 38, 23, C.navy, { bold: true, align: "center" });
    addCallout(s, "If the residual is rejected, the model returns the SwinIR base. NIR/SWIR do not alter its low-frequency RGB output.", 168, 524, 944, 44, C.orange, 20);
    addText(s, "Image-space upsampling uses resize-convolution to avoid PixelShuffle phase artifacts.", 224, 598, 832, 28, 18, C.teal, { bold: true, align: "center" });
    setNotes(s, "The SwinIR-style branch predicts an RGB residual over bicubic interpolation. In the guided model base_input_channels=3, so NIR/SWIR are excluded from this path.", [
      "Project source: src/geodiff_gan/models/base.py",
      "https://arxiv.org/abs/2108.10257",
    ]);
  }

  // 20. VAE.
  {
    const s = addSlide(p, "Residual variational autoencoder", "Model", 20);
    addText(s, "(a) VAE training path", 72, 146, 300, 24, 18, C.orange, { bold: true });
    addText(s, "(b) Inference path", 72, 408, 300, 24, 18, C.teal, { bold: true });

    const trainX = [70, 270, 476, 682, 884, 1080];
    const trainW = [142, 148, 150, 146, 148, 130];
    for (let i = 0; i < trainX.length - 1; i += 1) addArrow(s, trainX[i] + trainW[i] + 8, 244, trainX[i + 1] - (trainX[i] + trainW[i]) - 16, 9, C.dark);
    addNode(s, trainX[0], 198, trainW[0], 94, "r_target", "x_S2 - x_base", "orange");
    addNode(s, trainX[1], 186, trainW[1], 118, "Encoder", "384 -> 192 -> 96 -> 48", "navy");
    addNode(s, trainX[2], 186, trainW[2], 118, "Moments", "mu and log sigma^2\n4 x 48 x 48", "navy");
    addNode(s, trainX[3], 198, trainW[3], 94, "Sample z_0", "mu + sigma epsilon", "teal");
    addNode(s, trainX[4], 186, trainW[4], 118, "Decoder", "48 -> 96 -> 192 -> 384", "teal");
    addNode(s, trainX[5], 198, trainW[5], 94, "r_tilde", "3 x 384 x 384", "teal");
    addText(s, "L_VAE = L1(r_tilde, r_target) + beta_KL KL[q(z|r) || N(0,I)]", 226, 334, 828, 36, 20, C.navy, { bold: true, align: "center" });

    addArrow(s, 238, 500, 58, 10, C.dark);
    addArrow(s, 500, 500, 58, 10, C.dark);
    addArrow(s, 762, 500, 58, 10, C.dark);
    addNode(s, 72, 456, 158, 90, "Noise z_T", "~ N(0, I)", "gray");
    addNode(s, 304, 444, 188, 114, "Conditional diffusion", "LR-conditioned DDIM", "navy");
    addNode(s, 566, 456, 188, 90, "Denoised latent", "z_hat_0: 4 x 48 x 48", "teal");
    addNode(s, 828, 444, 226, 114, "GeoMapper + decoder", "content, FiLM, confidence\ncontrolled HR residual", "orange");
    addText(s, "The VAE encoder is absent at inference; no Sentinel pixels enter the model.", 246, 594, 788, 30, 20, C.orange, { bold: true, align: "center" });
    setNotes(s, "The VAE itself is not the super-resolution model. It is trained on target-minus-base residuals and provides the latent space in which diffusion operates.", ["Project source: src/geodiff_gan/models/vae.py"]);
  }

  // 21. Diffusion.
  {
    const s = addSlide(p, "Conditional latent diffusion", "Model", 21);
    addText(s, "Condition vector", 76, 146, 176, 22, 17, C.gray, { bold: true });
    addRect(s, 246, 138, 944, 54, C.white, C.dark, 1);
    addRule(s, 246, 138, 944, C.navy, 4);
    addText(s, "timestep t   |   SR/edit mode   |   multi-scale LR features   |   optional text and degradation", 270, 154, 896, 24, 17, C.dark, { align: "center" });

    // U-shaped denoising path and skip routes.
    addArrow(s, 220, 294, 48, 9, C.dark);
    addDownArrow(s, 418, 312, 12, 72, C.dark);
    addArrow(s, 430, 380, 28, 9, C.dark);
    addDownArrow(s, 608, 400, 12, 72, C.dark);
    addArrow(s, 620, 468, 28, 9, C.dark);
    addArrow(s, 808, 468, 28, 9, C.dark);
    addUpArrow(s, 996, 400, 12, 72, C.dark);
    addArrow(s, 1008, 380, 28, 9, C.dark);
    addUpArrow(s, 1184, 312, 12, 72, C.dark);
    addRule(s, 344, 228, 764, C.gray, 1);
    addDownArrow(s, 336, 228, 12, 38, C.gray);
    addDownArrow(s, 1100, 228, 12, 38, C.gray);
    addRule(s, 534, 320, 382, C.gray, 1);
    addDownArrow(s, 526, 320, 12, 38, C.gray);
    addDownArrow(s, 908, 320, 12, 38, C.gray);

    addNode(s, 72, 254, 140, 82, "z_t", "4 x 48 x 48", "orange");
    addNode(s, 276, 266, 134, 82, "Down 1", "128 @ 48^2", "navy");
    addNode(s, 466, 358, 134, 82, "Down 2", "256 @ 24^2", "navy");
    addNode(s, 656, 450, 144, 82, "Bottleneck", "384 @ 12^2", "teal");
    addNode(s, 844, 358, 144, 82, "Up 2 + skip", "256 @ 24^2", "teal");
    addNode(s, 1032, 266, 166, 82, "v head", "4 @ 48^2", "orange");
    addEdgeLabel(s, "skip connection", 630, 204, 194);
    addEdgeLabel(s, "skip connection", 628, 296, 194);
    addText(s, "DDIM: z_T -> z_t -> z_hat_0; repeat with different seeds to estimate uncertainty", 174, 562, 932, 32, 20, C.navy, { bold: true, align: "center" });
    addCallout(s, "Current implementation: image-space upsampling uses resize-convolution; the latent U-Net upsampler remains an ablation item.", 150, 604, 980, 38, C.orange, 17);
    setNotes(s, "Diffusion represents one-to-many uncertainty, but plausibility is not fidelity. Multiple samples support uncertainty estimation. Mention the current internal upsampler accurately as an implementation detail for future ablation.", [
      "Project source: src/geodiff_gan/models/diffusion.py",
      "https://arxiv.org/abs/2410.10812",
    ]);
  }

  // 22. Mapper decoder.
  {
    const s = addSlide(p, "GeoMapper and residual decoder", "Model", 22);
    addText(s, "(a) Spatial mapping", 72, 142, 260, 24, 18, C.teal, { bold: true });
    addText(s, "(b) Residual synthesis", 72, 360, 260, 24, 18, C.orange, { bold: true });

    addArrow(s, 218, 224, 42, 9, C.dark);
    addArrow(s, 218, 304, 42, 9, C.dark);
    addArrow(s, 516, 234, 56, 9, C.dark);
    addArrow(s, 516, 278, 56, 9, C.dark);
    addArrow(s, 516, 322, 56, 9, C.dark);
    addDownArrow(s, 632, 338, 12, 66, C.gray);
    addDownArrow(s, 806, 338, 12, 66, C.gray);
    addDownArrow(s, 980, 338, 12, 66, C.gray);
    addArrow(s, 232, 454, 34, 9, C.dark);
    addArrow(s, 424, 454, 34, 9, C.dark);
    addArrow(s, 616, 454, 34, 9, C.dark);
    addArrow(s, 808, 454, 34, 9, C.dark);
    addArrow(s, 1000, 454, 34, 9, C.dark);

    addNode(s, 72, 184, 138, 80, "Denoised z", "4 x 48 x 48", "navy");
    addNode(s, 72, 274, 138, 80, "LR feature", "f64: 48 x 64^2", "teal");
    addNode(s, 268, 176, 240, 170, "GeoMapper", "concat + projection\n4 spatial residual blocks\nshared 48 x 48 representation", "teal");
    addNode(s, 580, 196, 172, 74, "Content tensor", "48 channels @ 48^2", "navy");
    addNode(s, 580, 276, 172, 74, "FiLM styles", "4 x 96 parameters", "orange");
    addNode(s, 792, 196, 172, 74, "Evidence c", "1 channel @ 48^2", "teal");
    addNode(s, 1004, 196, 172, 74, "Edit gate", "separate policy", "gray");

    addNode(s, 72, 412, 152, 88, "Content", "48 @ 48^2", "navy");
    addNode(s, 274, 400, 142, 112, "Stage 1", "48 -> 96 | C48\nFiLM + skip", "orange");
    addNode(s, 466, 400, 142, 112, "Stage 2", "96 -> 192 | C36\nFiLM + skip", "orange");
    addNode(s, 658, 400, 142, 112, "Stage 3", "192 -> 384 | C24\nFiLM + skip", "orange");
    addNode(s, 850, 400, 142, 112, "Detail head", "r_hat: 3 x 384^2", "orange");
    addNode(s, 1042, 400, 154, 112, "Gate + HPF", "alpha * c * HPF", "teal");
    addText(s, "resize-convolution at each image-space upsampling stage", 326, 530, 628, 26, 18, C.gray, { italic: true, align: "center" });
    addCallout(s, "The confidence map scales the SR residual. The edit gate is used only in synthetic edit mode.", 174, 592, 932, 42, C.navy, 19);
    setNotes(s, "The decoder has two heads in code: detail residual and edit residual. SR mode high-pass filters the detail head and multiplies it by evidence confidence before adding it to the base.", [
      "Project source: src/geodiff_gan/models/generator.py",
      "Project source: src/geodiff_gan/models/system.py",
      "https://arxiv.org/abs/2405.04356",
    ]);
  }

  // 23. Prompt modes.
  {
    const s = addSlide(p, "Reconstruction mode and edit mode", "Model", 23);
    addSimpleTable(s,
      ["Design choice", "SR reconstruction mode", "Prompt edit mode"],
      [
        ["Condition", "null or weak prompt", "strong counterfactual prompt"],
        ["Residual permission", "evidence confidence c", "separate edit-permission map"],
        ["Frequency support", "high-pass detail only", "full-band residual allowed"],
        ["Sensor consistency", "strict / back-projected", "soft LR consistency"],
        ["Required label", "Sentinel-like estimate", "synthetic_edit=true"],
      ], 92, 158, [250, 430, 430], 52, 17);
    addText(s, "Current experiment", 104, 534, 220, 26, 19, C.navy, { bold: true });
    addArrow(s, 326, 542, 72, 10, C.dark);
    addText(s, "prompt path disabled; only SR reconstruction is scored with PSNR / SSIM", 414, 526, 742, 52, 20, C.dark, { bold: true });
    addCallout(s, "Edit-mode outputs are synthetic and are excluded from reconstruction metrics.", 166, 608, 948, 36, C.orange, 18);
    setNotes(s, "Do not mix prompt-edit results with reconstruction metrics. Edit mode is planned as a separately labeled synthetic visualization capability.");
  }

  // 24. Training.
  {
    const s = addSlide(p, "Stage-wise training schedule", "Training", 24);
    const stages = [
      ["1", "Base", "deterministic RGB fidelity"],
      ["2", "Residual VAE", "encode x_S2 - x_base"],
      ["3", "Diffusion", "predict residual latent"],
      ["4", "Joint refit", "controlled correction"],
    ];
    for (let i = 0; i < 3; i += 1) addArrow(s, 318 + i * 290, 224, 54, 10, C.dark);
    stages.forEach((stage, i) => {
      addNode(s, 92 + i * 290, 176, 218, 104, `${stage[0]}. ${stage[1]}`, stage[2], i === 3 ? "orange" : "teal");
    });
    addText(s, "Module update schedule", 92, 324, 320, 26, 19, C.navy, { bold: true });
    addSimpleTable(s,
      ["Stage", "Base", "VAE", "Diffusion", "LR encoder", "Mapper + decoder"],
      [
        ["1. Base", "train", "-", "-", "-", "-"],
        ["2. VAE", "frozen", "train", "-", "train", "train"],
        ["3. Diffusion", "frozen", "frozen", "train", "condition", "frozen"],
        ["4. Joint", "frozen", "frozen", "frozen first", "train", "train"],
      ], 92, 360, [180, 140, 140, 170, 190, 270], 42, 16);
    addText(s, "Selection rule", 94, 586, 160, 24, 18, C.orange, { bold: true });
    addText(s, "best validation PSNR | early stopping | best + latest checkpoints only | final joint checkpoint stores the complete model state", 252, 580, 934, 44, 18, C.dark);
    addText(s, "Prompt/edit adapters are a later, separately evaluated stage.", 374, 638, 532, 22, 16, C.gray, { italic: true, align: "center" });
    setNotes(s, "Stage-wise checkpoints initialize the next stage. During inference the final joint checkpoint contains the state dictionaries for the complete model, so a separate model object per stage is not loaded.");
  }

  // 25. Losses.
  {
    const s = addSlide(p, "Training losses", "Training", 25);
    addSimpleTable(s,
      ["Loss", "Question it asks", "Current role"],
      [
        ["MSE", "Are pixel values correct?", "primary PSNR driver"],
        ["SSIM", "Are local structure and contrast correct?", "primary"],
        ["Multiscale MSE", "Does fidelity survive at several scales?", "supporting"],
        ["Radiometric", "Are channel mean and contrast stable?", "supporting"],
        ["Residual / base guard", "Does correction match what base missed?", "regression control"],
      ], 74, 162, [230, 620, 280], 66, 17);
    addCallout(s, "Loss weights are coefficients, not accuracy percentages. They must be interpreted with validation metrics.", 154, 566, 972, 44, C.orange, 19);
    setNotes(s, "The supervisor-directed objective is currently distortion fidelity. Adversarial, wavelet, and perceptual losses should return only after the model proves it can beat the base.");
  }

  // 26. Metrics.
  {
    const s = addSlide(p, "Evaluation metrics", "Evaluation", 26);
    addTwoColumnBullets(s,
      "Primary",
      ["L1 / MAE: absolute reflectance error", "PSNR: fidelity derived from MSE", "SSIM: local luminance, contrast, structure"],
      "Secondary diagnostics",
      ["ERGAS / SAM: radiometric and spectral error", "UIQI: luminance, contrast and correlation", "sCC: high-pass structural correlation", "Re-degradation L1: Landsat consistency"],
      { leftColor: C.teal, rightColor: C.navy });
    addCallout(s, "PSNR is not percentage accuracy, and a sharp image is not necessarily correct.", 198, 558, 884, 46, C.orange, 22);
    setNotes(s, "All full-reference metrics depend on pair quality and registration. A high PSNR value can still hide spatially concentrated errors, so per-patch maps remain necessary.");
  }

  // 27. Diagnostics.
  {
    const s = addSlide(p, "Diagnostic outputs", "Evaluation", 27);
    const items = [
      ["Base", "low-frequency anchor"], ["Residual", "signed added detail"], ["Confidence", "accepted support"],
      ["Error map", "spatial mismatch"], ["Fourier", "periodic artifacts"], ["Wavelet", "directional detail"],
    ];
    const positions = [[74,176],[442,176],[810,176],[74,396],[442,396],[810,396]];
    positions.forEach((pos, i) => {
      addRect(s, pos[0], pos[1], 320, 156, i === 1 ? C.paleOrange : i === 2 ? C.paleTeal : C.light, i === 1 ? C.orange : i === 2 ? C.teal : C.mid, 1.5);
      if (i === 4) {
        for (let k = 0; k < 9; k += 1) addRect(s, pos[0] + 30 + k * 28, pos[1] + 34 + Math.abs(4-k)*5, 4, 60 - Math.abs(4-k)*8, C.navy, C.navy, 0);
      } else if (i === 2) {
        for (let r = 0; r < 5; r += 1) for (let c = 0; c < 9; c += 1) addRect(s, pos[0] + 28 + c*29, pos[1] + 28 + r*19, 25, 15, (r+c)%3===0 ? C.teal : C.paleTeal, C.white, 0);
      } else {
        for (let k = 0; k < 7; k += 1) addRule(s, pos[0] + 26, pos[1] + 34 + k*12, 260, i === 1 ? C.orange : C.navy, i === 1 && k%2 ? 3 : 1);
      }
      addText(s, items[i][0], pos[0] + 18, pos[1] + 112, 130, 26, 18, C.dark, { bold: true });
      addText(s, items[i][1], pos[0] + 140, pos[1] + 112, 160, 26, 16, C.gray, { align: "right" });
    });
    setNotes(s, "A residual repeated across unrelated inputs indicates a texture shortcut. Lattice peaks indicate phase artifacts. Confidence should vary spatially rather than collapse to a uniform scalar.");
  }

  // 28. Preliminary results.
  {
    const s = addSlide(p, "Initial comparison of three model variants", "Results", 28, "Single split | 28 test patches | 4 stochastic samples | 20 diffusion steps");
    addSimpleTable(s,
      ["Experiment", "L1 down", "PSNR up", "SSIM up", "Edge F1 up"],
      [
        ["RGB standard", "0.01426", "33.44", "0.8580", "0.0712"],
        ["RGB fidelity", "0.01507", "33.29", "0.8510", "0.0340"],
        ["Multispectral fidelity", "0.01546", "32.97", "0.8538", "0.0904"],
      ], 92, 182, [340, 180, 180, 180, 180], 72, 19);
    addMetricBar(s, "Best PSNR", "RGB standard", 0.88, 126, 470, 1020, C.teal);
    addMetricBar(s, "Best edge F1", "Multispectral", 0.70, 126, 516, 1020, C.orange);
    addCallout(s, "This table compares complete models. Each model must also be compared with its own SwinIR base.", 142, 580, 998, 38, C.navy, 18);
    setNotes(s, "Report the negative result plainly. Adding fidelity weights or sending six bands through the whole network did not reliably improve PSNR/SSIM. The sample size is diagnostic, not conclusive.", ["Researcher-supplied preliminary evaluation table from the SR-3x experiment suite."]);
  }

  // 29. Failure causes.
  {
    const s = addSlide(p, "Analysis of the initial results", "Results", 29);
    addBullets(s, [
      "Residual Landsat-Sentinel radiometric mismatch remains in the pair.",
      "The old multispectral design allowed NIR/SWIR to shift base RGB.",
      "Diffusion can add texture that does not match the target pixels.",
      "Joint tuning can overfit the decoder to training-specific patterns.",
      "Registration and temporal change impose an upper bound on PSNR.",
      "A small test set creates unstable rankings.",
    ], 86, 142, 1110, 72, 20, C.orange);
    addCallout(s, "Next experiment: test calibration, multispectral guidance and residual scaling separately.", 126, 596, 1028, 40, C.teal, 19);
    setNotes(s, "Explain that a larger network can memorize mismatch more effectively. The improvement plan changes responsibility boundaries before changing capacity.");
  }

  // 30. New improvement.
  {
    const s = addSlide(p, "Revised RGB harmonization and multispectral guidance", "Experiment", 30);
    addArrow(s, 306, 260, 54, 14, C.gray);
    addArrow(s, 600, 260, 54, 14, C.gray);
    addArrow(s, 902, 260, 54, 14, C.gray);
    addNode(s, 80, 200, 216, 140, "Train patches only", "fit per-channel\nslope + offset", "gray");
    addNode(s, 370, 200, 220, 140, "Harmonized RGB", "Sentinel_30m = a x\nLandsat_30m + b", "orange");
    addNode(s, 664, 200, 228, 140, "Separated branches", "RGB -> base\n6 bands -> residual", "teal");
    addNode(s, 966, 200, 230, 140, "Conservative joint", "freeze diffusion\nselect residual alpha", "navy");
    addCallout(s, "Calibration is fitted on the training split only, and the fitted coefficients are saved.", 112, 454, 1038, 48, C.orange, 20);
    addCallout(s, "Residual scaling is selected on validation data; alpha = 0 returns the base model.", 112, 534, 1038, 48, C.teal, 20);
    setNotes(s, "Calibration is fitted at 30 m from training pairs only. Validation and test targets never influence the coefficients. NIR/SWIR can guide the residual pathway but cannot directly alter base RGB.", [
      "Project source: src/geodiff_gan/data/radiometry.py",
      "Project source: configs/landsat_sentinel_3x_multispectral_guided.yaml",
      "https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf",
    ]);
  }

  // 31. Experiment matrix.
  {
    const s = addSlide(p, "Controlled experiment variants", "Experiment", 31);
    addSimpleTable(s,
      ["Experiment", "RGB harmonization", "Base input", "Residual input"],
      [
        ["RGB fidelity", "No", "RGB", "RGB"],
        ["Multispectral fidelity", "No", "6 bands", "6 bands"],
        ["RGB harmonized", "Yes", "RGB", "RGB"],
        ["Multispectral guided", "Yes", "RGB", "6 bands"],
      ], 84, 160, [360, 280, 250, 250], 62, 18);
    addText(s, "Acceptance criteria", 104, 476, 260, 30, 24, C.navy, { bold: true });
    addText(s, "PSNR delta vs base > 0  |  SSIM delta >= 0  |  L1 improvement > 0", 104, 526, 1070, 34, 21, C.dark, { bold: true });
    addCallout(s, "Validation selects alpha from 0, 0.25, 0.5, 0.75 and 1.0. Alpha = 0 returns the base output.", 188, 586, 904, 38, C.orange, 18);
    setNotes(s, "Predefine the criteria before examining test results. The same validation rule must be applied to all models, and test targets must not select alpha.");
  }

  // 32. Novelty.
  {
    const s = addSlide(p, "Proposed research contribution", "Contribution", 32);
    addTwoColumnBullets(s,
      "Established components",
      ["SwinIR", "residual VAE", "conditional diffusion U-Net", "FiLM decoder and wavelet GAN"],
      "Candidate contribution",
      ["deterministic base and stochastic residual", "separate reconstruction and edit policies", "RGB base with multispectral residual guidance", "validation can select the base-only output"],
      { leftColor: C.gray, rightColor: C.teal });
    addCallout(s, "The contribution will be claimed only if controlled ablations show a repeatable improvement.", 150, 570, 980, 44, C.orange, 20);
    setNotes(s, "The inspiration is residual decomposition in HART, diffusion-to-GAN mapping and style modulation, and high-frequency wavelet discrimination. The satellite-specific claim is the evidence-controlled cross-sensor integration and its validated behavior.", [
      "https://arxiv.org/abs/2410.10812",
      "https://arxiv.org/abs/2405.04356",
      "https://arxiv.org/abs/2402.19215",
    ]);
  }

  // 33. Completed.
  {
    const s = addSlide(p, "Implementation status", "Progress", 33);
    addBullets(s, [
      "Synthetic 4x and real Landsat-to-Sentinel 3x data pipelines",
      "QA masking, pairing, spatial splits, quarantine, and cached NPZ manifests",
      "RGB and six-band small/medium GeoDiff-GAN variants",
      "Stage-wise training, resume, early stopping, best/latest checkpoints",
      "Debug tensors, Fourier/wavelet maps, uncertainty, and per-patch metrics",
      "Train-only calibration and dual-stream model path implemented",
    ], 82, 146, 1120, 73, 20, C.teal);
    addCallout(s, "Each experiment has its own config, manifest, calibration file and output directory.", 154, 596, 972, 40, C.navy, 19);
    setNotes(s, "Separate software completion from scientific validation. The implementation exists; the next experiments determine whether the mechanisms improve held-out fidelity.");
  }

  // 34. Limitations.
  {
    const s = addSlide(p, "Limitations of the reference pairs", "Limits", 34);
    addBullets(s, [
      "Sentinel is a paired reference observation, not perfect hidden ground truth.",
      "Misregistration can teach false edges and reduce otherwise reasonable PSNR.",
      "A global affine calibration cannot model nonlinear band-response differences.",
      "A three-day gap does not guarantee no land-cover or atmospheric change.",
      "Some generated details may be incorrect.",
      "Current geographic diversity is insufficient for broad deployment claims.",
    ], 88, 144, 1110, 73, 20, C.orange);
    addCallout(s, "The output is a Sentinel-like estimate from Landsat input; it is not an observed 10 m Landsat image.", 124, 594, 1032, 42, C.navy, 19);
    setNotes(s, "Keep raw products, masks, provenance, uncertainty, and reference imagery available to the analyst. Prompt edits require an explicit synthetic label.");
  }

  // 35. Remaining.
  {
    const s = addSlide(p, "Remaining experiments", "Next", 35);
    addBullets(s, [
      "Run harmonized RGB and multispectral-guided experiments.",
      "Compare base versus final per patch, per tile, and by scene type.",
      "Audit calibration coefficients and registration-shift distributions.",
      "Run spatial K-fold and complete-tile holdout evaluation.",
      "Add recent open-source baselines, seeds, and confidence intervals.",
      "Add perceptual, wavelet and GAN losses only after PSNR/SSIM improve.",
      "Keep QGIS as the current analysis tool; evaluate ArcGIS integration later.",
    ], 84, 128, 1120, 68, 19, C.teal);
    setNotes(s, "Prioritize data audit and base-relative evaluation. Model complexity is not the first remaining dependency.");
  }

  // 36. Outcomes.
  {
    const s = addSlide(p, "Possible outcomes of the study", "Next", 36);
    const outcomes = [
      ["1", "Full model wins", "fidelity and detail improve"],
      ["2", "Base wins reconstruction", "diffusion helps only perceptual/edit mode"],
      ["3", "Pair uncertainty dominates", "data quality must improve first"],
    ];
    outcomes.forEach((o, i) => {
      addRect(s, 92 + i * 390, 202, 340, 246, i === 0 ? C.paleTeal : i === 1 ? C.paleNavy : C.paleOrange, i === 0 ? C.teal : i === 1 ? C.navy : C.orange, 2);
      addText(s, o[0], 116 + i * 390, 222, 60, 54, 38, i === 0 ? C.teal : i === 1 ? C.navy : C.orange, { bold: true });
      addText(s, o[1], 116 + i * 390, 302, 292, 58, 24, C.dark, { bold: true });
      addText(s, o[2], 116 + i * 390, 374, 292, 50, 18, C.gray);
    });
    addCallout(s, "A negative result is still useful if the experiment is controlled and fully reported.", 222, 536, 836, 46, C.navy, 22);
    setNotes(s, "Make the decision rule explicit: held-out measurements and pair audits determine the conclusion, not preference for the more complex architecture.");
  }

  // 37. Conclusion.
  {
    const s = addSlide(p, "Current conclusion", "Conclusion", 37);
    addText(s, "01", 90, 174, 70, 44, 30, C.teal, { bold: true });
    addText(s, "SR is useful when matching HR data are unavailable, but the output remains an estimate.", 178, 170, 950, 64, 23, C.dark, { bold: true });
    addRule(s, 90, 252, 1060, C.mid, 1);
    addText(s, "02", 90, 292, 70, 44, 30, C.teal, { bold: true });
    addText(s, "Real Landsat-to-Sentinel SR is also a harmonization, registration, and temporal-matching problem.", 178, 288, 950, 64, 23, C.dark, { bold: true });
    addRule(s, 90, 370, 1060, C.mid, 1);
    addText(s, "03", 90, 410, 70, 44, 30, C.teal, { bold: true });
    addText(s, "The model uses an RGB base with multispectral residual guidance. Confidence gating and validation-based scaling limit unwanted corrections.", 178, 406, 950, 76, 23, C.dark, { bold: true });
    addCallout(s, "The next experiment must test whether the learned residual improves on SwinIR on held-out data.", 182, 560, 916, 50, C.orange, 21);
    setNotes(s, "Close by resolving the opening question. The central criterion is whether the residual improves held-out reconstruction without violating evidence or creating systematic artifacts.");
  }

  // 38. Discussion.
  {
    const s = addSlide(p, "Decisions required for the final study", "Discussion", 38);
    addText(s, "1", 96, 188, 52, 50, 34, C.orange, { bold: true });
    addText(s, "Calibration", 166, 188, 240, 34, 24, C.navy, { bold: true });
    addText(s, "Is train-only affine harmonization sufficient, or should nonlinear sensor-response calibration be tested?", 166, 234, 950, 56, 20, C.dark);
    addRule(s, 96, 312, 1050, C.mid, 1);
    addText(s, "2", 96, 340, 52, 50, 34, C.orange, { bold: true });
    addText(s, "Geographic holdout", 166, 340, 300, 34, 24, C.navy, { bold: true });
    addText(s, "Which complete city or tile best represents the intended deployment geography?", 166, 386, 950, 42, 20, C.dark);
    addRule(s, 96, 458, 1050, C.mid, 1);
    addText(s, "3", 96, 486, 52, 50, 34, C.orange, { bold: true });
    addText(s, "Operating points", 166, 486, 260, 34, 24, C.navy, { bold: true });
    addText(s, "Should reconstruction and perceptual/edit objectives be reported as separate operating modes?", 166, 532, 950, 42, 20, C.dark);
    addText(s, "Questions and critical feedback are welcome.", 166, 608, 950, 30, 20, C.teal, { bold: true });
    setNotes(s, "Invite the panel to challenge the data protocol and acceptance criteria. These decisions are more consequential than cosmetic architecture changes.");
  }

  // Backup 39. Equations.
  {
    const s = addSlide(p, "Appendix: reconstruction equations", "Appendix", 39);
    addText(s, "x_base = B(y_RGB)", 122, 174, 1020, 50, 30, C.navy, { bold: true, align: "center" });
    addText(s, "r_hat = R(y_RGB,NIR,SWIR,z)", 122, 258, 1020, 50, 30, C.teal, { bold: true, align: "center" });
    addText(s, "x_hat = clip(x_base + alpha x confidence x high_pass(r_hat), 0, 1)", 84, 342, 1112, 60, 27, C.orange, { bold: true, align: "center" });
    addRule(s, 220, 440, 840, C.mid, 2);
    addText(s, "PSNR = -10 log10(MSE)  for normalized reflectance", 170, 478, 940, 46, 25, C.dark, { align: "center" });
    addCallout(s, "Alpha is used only when validation performance improves over the base output.", 288, 570, 704, 40, C.teal, 20);
    setNotes(s, "Use y for Landsat, x for Sentinel, B for the deterministic base, and R for the residual generator. Explain each term before discussing the equation.");
  }

  // Backup 40. Tensors.
  {
    const s = addSlide(p, "Appendix: multispectral model tensor sizes", "Appendix", 40);
    addSimpleTable(s,
      ["Tensor", "Shape without batch", "Role"],
      [
        ["Landsat RGB", "3 x 128 x 128", "base input"],
        ["Landsat multispectral", "6 x 128 x 128", "residual evidence"],
        ["SwinIR base", "3 x 384 x 384", "deterministic anchor"],
        ["LR features", "24@128, 48@64, 96@32, 96@16", "multi-scale skips"],
        ["Residual latent", "4 x 48 x 48", "diffusion state"],
        ["Mapper outputs", "48 x 48 x 48 + 4 styles + 1 gate", "decoder controls"],
        ["Final RGB", "3 x 384 x 384", "Sentinel-like estimate"],
      ], 70, 138, [320, 440, 390], 62, 17);
    setNotes(s, "These values come from the current small guided configuration: base_embed_dim=32, lr_channels=24, mapper_channels=48, style_dim=96, and scale=3.", ["Project source: configs/landsat_sentinel_3x_multispectral_guided.yaml"]);
  }

  // Backup 41. Resize conv.
  {
    const s = addSlide(p, "Appendix: resize-convolution upsampling", "Appendix", 41);
    addTwoColumnBullets(s,
      "PixelShuffle risk",
      ["channels become spatial phases", "unequal phase learning can form lattice patterns", "requires Fourier inspection"],
      "Resize-convolution path",
      ["bilinear resize first", "ordinary 3 x 3 convolution second", "used in current base and residual decoder"],
      { leftColor: C.orange, rightColor: C.teal });
    addCallout(s, "Resize-convolution reduces one source of periodic artifacts but does not guarantee an artifact-free output.", 208, 558, 864, 44, C.navy, 21);
    setNotes(s, "Be precise: the current image-space paths use resize-convolution, but the latent diffusion U-Net retains an internal PixelShuffle upsampler that can be ablated separately.", ["Project source: src/geodiff_gan/models/base.py and generator.py"]);
  }

  // Backup 42. Reproducibility.
  {
    const s = addSlide(p, "Appendix: files saved for reproducibility", "Appendix", 42);
    addBullets(s, [
      "Git branch and commit",
      "resolved YAML configuration",
      "manifest, masks, and calibration snapshot",
      "random seed and environment versions",
      "stage initialization, best, and latest checkpoints",
      "validation selection decision and residual alpha",
      "aggregate metrics, per-patch metrics, outputs, and diagnostics",
    ], 92, 142, 1090, 70, 20, C.teal);
    setNotes(s, "This record prevents accidental comparison of different data, checkpoints, or inference settings. Best and latest checkpoints serve different audit purposes.");
  }

  // Backup 43. Honest result labels.
  {
    const s = addSlide(p, "Appendix: terminology used in this study", "Appendix", 43);
    addSimpleTable(s,
      ["Label", "Use when"],
      [
        ["Preliminary", "limited patches, one split, or one seed"],
        ["Diagnostic", "finding mechanisms and failures, not claiming generalization"],
        ["Reference target", "paired Sentinel observation, not hidden truth"],
        ["Generated estimate", "model output in reconstruction mode"],
        ["Candidate contribution", "mechanism still awaiting ablation"],
      ], 126, 168, [300, 730], 68, 19);
    addCallout(s, "Report PSNR, SSIM, MAE, ERGAS, SAM, UIQI and sCC by name instead of calling them accuracy.", 166, 570, 948, 44, C.orange, 20);
    setNotes(s, "These labels prevent overclaiming and make the progress seminar easier to defend under cross-questioning.");
  }

  return p;
}

async function main() {
  await fs.mkdir(OUTPUT_DIR, { recursive: true });
  await fs.mkdir(RENDER_DIR, { recursive: true });
  await fs.mkdir(ASSET_DIR, { recursive: true });
  const assets = {
    landsat: await readImageBlob(path.join(SOURCE_ASSET_DIR, "landsat_30m.webp")),
    inferred: await readImageBlob(path.join(SOURCE_ASSET_DIR, "infered_10m.webp")),
    sentinel: await readImageBlob(path.join(SOURCE_ASSET_DIR, "sentinel_10m.webp")),
  };
  const presentation = buildDeck(assets);

  const sourceNotes = [
    "GeoDiff-GAN SR-3x progress seminar source record",
    "",
    "Project sources:",
    "- src/geodiff_gan/models/system.py",
    "- src/geodiff_gan/models/base.py",
    "- src/geodiff_gan/models/generator.py",
    "- src/geodiff_gan/models/diffusion.py",
    "- src/geodiff_gan/models/vae.py",
    "- configs/landsat_sentinel_3x_multispectral_guided.yaml",
    "- learning/39_progress_seminar_presentation_content.md",
    "- learning/40_progress_seminar_cross_questions.md",
    "- outs/landsat_30m.webp",
    "- outs/infered_10m.webp",
    "- outs/sentinel_10m.webp",
    "",
    "External sources are listed in each slide's [Sources] speaker-notes block.",
    "No old 4x architecture infographic was used.",
  ].join("\n");
  await fs.writeFile(path.join(BUILD_ROOT, "source-notes.txt"), sourceNotes);

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(FINAL_PPTX);
  console.log(`Created ${FINAL_PPTX}`);
  console.log(`Slides: ${presentation.slides.items.length}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
