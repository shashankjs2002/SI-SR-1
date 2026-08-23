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
    teal: { fill: C.paleTeal, line: C.teal },
    navy: { fill: C.paleNavy, line: C.navy },
    orange: { fill: C.paleOrange, line: C.orange },
    gray: { fill: C.light, line: C.gray },
  };
  const style = styles[palette];
  const box = addRect(slide, x, y, w, h, style.fill, style.line, 1.5, name);
  const multilineTitle = title.includes("\n") || title.length > 17;
  const titleHeight = multilineTitle ? 46 : 28;
  const titleSize = multilineTitle ? 18 : 20;
  addText(slide, title, x + 12, y + 10, w - 24, titleHeight, titleSize, style.line, { bold: true, align: "center" });
  if (detail) {
    const detailTop = y + 18 + titleHeight;
    addText(slide, detail, x + 10, detailTop, w - 20, h - (detailTop - y) - 10, 15, C.gray, { align: "center" });
  }
  return box;
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
  addText(slide, "Evidence-Constrained\nDiffusion-Driven GAN", 72, 116, 680, 148, 52, C.navy, { bold: true });
  addText(slide, "3x cross-sensor satellite image super-resolution", 74, 280, 680, 42, 25, C.gray);
  addCallout(slide, "Landsat 8/9 at 30 m  ->  Sentinel-2 reference at 10 m", 74, 354, 680, 48, C.orange, 22);

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
    const s = addSlide(p, "One Landsat pixel cannot reveal nine unique Sentinel pixels", "Context", 2);
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
    addCallout(s, "The nine values are constrained by one measurement, but they are not uniquely observed by Landsat.", 156, 608, 968, 44, C.orange, 20);
    setNotes(s, "Both panels cover the same geographic footprint. The boxes explain sampling density, not physical display size. Expected answer: one coarse measurement cannot uniquely determine nine fine measurements; the inverse problem is ill-posed.", [
      "https://www.usgs.gov/landsat-missions/landsat-8",
      "Project assets: outs/landsat_30m.webp and outs/sentinel_10m.webp",
    ]);
  }

  // 3. Spatial resolution.
  {
    const s = addSlide(p, "Spatial resolution is the ground area represented by one sample", "Context", 3);
    addColumnHeader(s, "Landsat 8/9 OLI", 98, 176, 440, C.orange);
    addText(s, "30 m", 98, 236, 440, 80, 52, C.orange, { bold: true });
    addText(s, "RGB, NIR and SWIR bands used here", 98, 330, 440, 48, 21, C.gray);
    addColumnHeader(s, "Sentinel-2 MSI", 706, 176, 440, C.teal);
    addText(s, "10 m", 706, 236, 440, 80, 52, C.teal, { bold: true });
    addText(s, "RGB reference bands B4 / B3 / B2", 706, 330, 440, 48, 21, C.gray);
    addRule(s, 622, 176, 2, C.mid, 360);
    addCallout(s, "Smaller ground sampling distance increases spatial sampling; it does not automatically improve spectral, radiometric, or temporal resolution.", 130, 500, 1010, 66, C.navy, 21);
    setNotes(s, "Use this slide to separate spatial resolution from the other three resolution dimensions. The model changes the output sampling grid; it does not upgrade the sensor itself.", [
      "https://www.usgs.gov/landsat-missions/landsat-8",
      "https://documentation.dataspace.copernicus.eu/Data/SentinelMissions/Sentinel2.html",
    ]);
  }

  // 4. SR definition.
  {
    const s = addSlide(p, "Super-resolution estimates detail; interpolation enlarges the grid", "Context", 4);
    addArrow(s, 330, 292, 110, 22, C.gray);
    addArrow(s, 750, 292, 110, 22, C.gray);
    addNode(s, 100, 225, 220, 160, "LR measurement", "3 x 128 x 128\nactual Landsat", "orange");
    addNode(s, 452, 225, 286, 160, "Learned prior", "patterns from aligned\ntraining pairs", "navy");
    addNode(s, 870, 225, 290, 160, "HR estimate", "3 x 384 x 384\nSentinel-like RGB", "teal");
    addText(s, "Bicubic", 100, 170, 220, 32, 23, C.gray, { bold: true, align: "center" });
    addText(s, "Learned SR", 556, 170, 520, 32, 23, C.teal, { bold: true, align: "center" });
    addCallout(s, "Bicubic: larger array, no scene-specific inference.", 122, 450, 470, 48, C.gray, 19);
    addCallout(s, "Learned SR: estimates structure, then must be validated.", 680, 450, 480, 48, C.teal, 19);
    setNotes(s, "Emphasize that array enlargement is not equivalent to information recovery. The model output remains an estimate even when it is visually sharper.", ["https://arxiv.org/abs/2108.10257"]);
  }

  // 5. Why SR.
  {
    const s = addSlide(p, "SR matters when the matching high-resolution observation is unavailable", "Context", 5);
    addBullets(s, [
      "Historical continuity: Landsat provides a long archive for retrospective studies.",
      "Clouds, acquisition gaps, and timing can remove the desired Sentinel observation.",
      "Commercial very-high-resolution imagery may be restricted or expensive.",
      "A consistent lower-resolution record may exist when the higher-resolution record does not.",
    ], 90, 160, 1080, 82, 21, C.teal);
    addCallout(s, "If a valid cloud-free Sentinel-2 image exists for the required place and date, use the real observation.", 140, 542, 1000, 54, C.orange, 22);
    setNotes(s, "Ask the audience what should be preferred when a matching Sentinel image exists. The answer is the real measurement. SR addresses missing, cloudy, costly, or historical cases.", [
      "https://www.usgs.gov/landsat-missions/landsat-surface-reflectance",
      "https://documentation.dataspace.copernicus.eu/Data/SentinelMissions/Sentinel2.html",
    ]);
  }

  // 6. Applications.
  {
    const s = addSlide(p, "The useful role is analysis support, not unquestioned ground truth", "Context", 6);
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
    const s = addSlide(p, "The model can estimate structure, not unique hidden reality", "Context", 7);
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
    const s = addSlide(p, "The study tests real 3x cross-sensor reconstruction", "Problem", 8);
    addArrow(s, 350, 284, 90, 20, C.gray);
    addArrow(s, 810, 284, 90, 20, C.gray);
    addNode(s, 84, 214, 250, 178, "Input", "Landsat 8/9 L2\nRGB at 30 m\n3 x 128 x 128", "orange");
    addNode(s, 460, 214, 330, 178, "GeoDiff-GAN SR-3x", "evidence-constrained\nresidual generation\noptional NIR/SWIR guidance", "navy");
    addNode(s, 922, 214, 270, 178, "Reference", "Sentinel-2 L2A\nRGB at 10 m\n3 x 384 x 384", "teal");
    addCallout(s, "Primary objective: improve held-out PSNR and SSIM without making the full model worse than its own deterministic base.", 116, 505, 1048, 62, C.orange, 22);
    setNotes(s, "This is not synthetic downsampling. Landsat and Sentinel are different observations from different instruments, so sensor mismatch becomes part of the learning problem.", [
      "https://www.usgs.gov/landsat-missions/landsat-surface-reflectance",
      "https://documentation.dataspace.copernicus.eu/Data/SentinelMissions/Sentinel2.html",
    ]);
  }

  // 9. Evolution.
  {
    const s = addSlide(p, "The research moved from software validation to real sensor harmonization", "Problem", 9);
    addArrow(s, 368, 288, 90, 18, C.gray);
    addArrow(s, 792, 288, 90, 18, C.gray);
    addNode(s, 80, 205, 270, 188, "Phase 1", "Synthetic Sentinel\n40 m -> 10 m\n4x architecture test", "gray");
    addNode(s, 480, 205, 290, 188, "Phase 2", "Real Landsat 30 m\n-> Sentinel 10 m\n3x cross-sensor pairs", "orange");
    addNode(s, 904, 205, 292, 188, "Phase 3", "Train-only RGB harmonization\n+ multispectral residual\nguidance", "teal");
    addText(s, "Each phase answers a harder question; model size alone cannot solve data mismatch.", 156, 500, 970, 42, 24, C.navy, { bold: true, align: "center" });
    setNotes(s, "Explain that the current SR-3x branch was cut to address the supervisor-directed real Landsat-to-Sentinel problem. The new phase follows measured failure modes.");
  }

  // 10. Cross-sensor dimensions.
  {
    const s = addSlide(p, "A cross-sensor pixel difference has four possible causes", "Data", 10);
    const xs = [82, 382, 682, 982];
    const labels = [
      ["Spatial", "sampling, PSF, alignment"],
      ["Spectral", "different band responses"],
      ["Radiometric", "calibration, atmosphere, illumination"],
      ["Temporal", "real change between dates"],
    ];
    labels.forEach((item, index) => {
      addRect(s, xs[index], 198, 216, 230, index === 1 ? C.paleTeal : index === 2 ? C.paleOrange : C.paleNavy, index === 2 ? C.orange : index === 1 ? C.teal : C.navy, 2);
      addText(s, String(index + 1), xs[index] + 18, 216, 42, 42, 28, index === 2 ? C.orange : index === 1 ? C.teal : C.navy, { bold: true });
      addText(s, item[0], xs[index] + 18, 278, 180, 34, 24, C.dark, { bold: true });
      addText(s, item[1], xs[index] + 18, 330, 180, 74, 18, C.gray);
    });
    addCallout(s, "If the pair is wrong, a larger model learns the mismatch more efficiently; it does not solve the science.", 150, 512, 980, 58, C.orange, 22);
    setNotes(s, "Use roads and riverbanks as alignment examples. Explain that a mismatch may be sensor color response, haze, change, or displacement rather than missing spatial detail.", [
      "https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf",
      "https://www.usgs.gov/landsat-missions/landsat-8",
      "https://documentation.dataspace.copernicus.eu/Data/SentinelMissions/Sentinel2.html",
    ]);
  }

  // 11. Questions.
  {
    const s = addSlide(p, "Six questions determine whether the generative residual is useful", "Problem", 11);
    addBullets(s, [
      "Does the final model beat bicubic and the exact embedded SwinIR base?",
      "Does diffusion add measured detail or only plausible texture?",
      "Can evidence confidence suppress unsupported residuals?",
      "Does train-only radiometric harmonization reduce domain gap?",
      "Can NIR/SWIR guide the residual without shifting base RGB?",
      "Do gains survive spatial and complete-tile holdout tests?",
    ], 88, 142, 1110, 73, 20, C.teal);
    setNotes(s, "State the hypothesis: a conservative RGB base plus evidence-gated multispectral residual guidance should be safer than unrestricted full-image generation.");
  }

  // 12. Bands.
  {
    const s = addSlide(p, "RGB is reconstructed; NIR and SWIR are guidance channels", "Data", 12);
    addSimpleTable(s,
      ["Role", "Product", "Bands", "Resolution"],
      [
        ["LR RGB", "Landsat 8/9 C2 L2", "B4 / B3 / B2", "30 m"],
        ["Residual guidance", "Landsat 8/9 C2 L2", "B5 / B6 / B7", "30 m"],
        ["HR reference", "Sentinel-2 L2A", "B4 / B3 / B2", "10 m"],
        ["Quality", "Both products", "QA / SCL / no-data", "native"],
      ], 78, 170, [210, 350, 330, 190], 62, 18);
    addCallout(s, "The six-band model still predicts only three RGB output channels.", 190, 542, 900, 48, C.teal, 22);
    setNotes(s, "Surface reflectance is used instead of display RGB or raw digital numbers. Auxiliary bands provide material and moisture cues but are never presented as output colors.", [
      "https://www.usgs.gov/landsat-missions/landsat-8",
      "https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S2L2A.html",
    ]);
  }

  // 13. Pair creation.
  {
    const s = addSlide(p, "Pair creation is a geospatial measurement pipeline", "Data", 13);
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
    addCallout(s, "Every pair stores sensor product IDs, date gap, transforms, masks, and split provenance.", 150, 554, 980, 46, C.orange, 21);
    setNotes(s, "Walk through the processing sequence. Stress that a patch is not just two images; it is a paired measurement with provenance and validity support.", [
      "https://www.usgs.gov/landsat-missions/landsat-surface-reflectance",
      "https://documentation.dataspace.copernicus.eu/Data/SentinelMissions/Sentinel2.html",
    ]);
  }

  // 14. Equal footprint.
  {
    const s = addSlide(p, "Different array sizes cover the same 3.84 km footprint", "Data", 14);
    addText(s, "128 pixels x 30 m", 118, 198, 420, 48, 30, C.orange, { bold: true, align: "center" });
    addText(s, "= 3840 m", 118, 264, 420, 54, 38, C.navy, { bold: true, align: "center" });
    addText(s, "384 pixels x 10 m", 742, 198, 420, 48, 30, C.teal, { bold: true, align: "center" });
    addText(s, "= 3840 m", 742, 264, 420, 54, 38, C.navy, { bold: true, align: "center" });
    addRule(s, 632, 182, 2, C.mid, 200);
    addCallout(s, "Equal panel width in a plot does not mean equal native resolution. Bicubic 384 x 384 adds no new measurement.", 134, 448, 1012, 62, C.orange, 22);
    addText(s, "Audience question: why can the 128-pixel image look physically as large as the 384-pixel image?", 156, 548, 968, 42, 20, C.gray, { italic: true, align: "center" });
    setNotes(s, "The plotting library scales both arrays to the same display box. Native sample count and ground sampling distance do not change.");
  }

  // 15. QA.
  {
    const s = addSlide(p, "Masks remove invalid pixels; quarantine records rejected pairs", "Data", 15);
    addTwoColumnBullets(s,
      "Automatic rejection",
      ["cloud / cirrus / shadow / snow", "no-data and black borders", "radiometric saturation", "valid support below 95%"],
      "Manual pair audit",
      ["road and river displacement", "coastline misalignment", "large temporal change", "unexpected color response"],
      { leftColor: C.orange, rightColor: C.teal });
    addCallout(s, "Valid fraction is necessary, but it cannot prove subpixel registration.", 210, 558, 860, 44, C.navy, 22);
    setNotes(s, "Rejected samples are quarantined with a reason rather than silently deleted. This preserves an audit trail and supports later threshold tuning.", [
      "https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S2L2A.html",
    ]);
  }

  // 16. Splits.
  {
    const s = addSlide(p, "Spatial guard bands prevent overlap leakage", "Data", 16);
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
    const s = addSlide(p, "The final model must beat the exact base it contains", "Evaluation", 17);
    addText(s, "Evaluation ladder", 88, 160, 300, 34, 24, C.navy, { bold: true });
    const labels = ["Bicubic", "SwinIR base", "RGB GeoDiff-GAN", "Multispectral guided", "External SR methods"];
    labels.forEach((label, i) => {
      addRect(s, 104 + i * 214, 236, 182, 118, i === 1 ? C.paleOrange : i >= 2 ? C.paleTeal : C.light, i === 1 ? C.orange : i >= 2 ? C.teal : C.gray, 2);
      addText(s, label, 116 + i * 214, 272, 158, 48, 20, C.dark, { bold: true, align: "center" });
    });
    addCallout(s, "Fairness requires identical pairs, masks, splits, test indices, and metric implementations.", 152, 444, 976, 52, C.navy, 22);
    addText(s, "Safety baseline", 344, 376, 180, 26, 17, C.orange, { bold: true, align: "center" });
    setNotes(s, "SwinIR is both an internal branch and a baseline. A final output below the base in held-out PSNR/SSIM is a regression, even if it looks sharper.", ["https://arxiv.org/abs/2108.10257"]);
  }

  // 18. Current architecture overview.
  {
    const s = addSlide(p, "The current SR-3x model separates reconstruction from generated detail", "Model", 18);
    // Arrows first.
    addArrow(s, 240, 222, 50, 14, C.gray);
    addArrow(s, 488, 222, 52, 14, C.gray);
    addArrow(s, 1018, 280, 54, 14, C.gray);
    addArrow(s, 238, 466, 50, 14, C.gray);
    addArrow(s, 470, 466, 48, 14, C.gray);
    addArrow(s, 704, 466, 46, 14, C.gray);
    addArrow(s, 918, 466, 50, 14, C.gray);
    addDownArrow(s, 1115, 354, 18, 72, C.gray);
    addNode(s, 74, 170, 158, 118, "Landsat RGB", "3 x 128 x 128", "orange");
    addNode(s, 298, 170, 180, 118, "SwinIR base", "RGB only\nresize-conv 3x", "navy");
    addNode(s, 548, 170, 226, 118, "Conservative base", "3 x 384 x 384", "navy");
    addNode(s, 80, 414, 150, 118, "RGB +\nNIR / SWIR", "6 x 128 x 128", "orange");
    addNode(s, 296, 414, 166, 118, "LR encoder", "multi-scale evidence", "teal");
    addNode(s, 526, 414, 168, 118, "Latent\ndiffusion", "noise -> 4 x 48 x 48", "navy");
    addNode(s, 758, 414, 152, 118, "GeoMapper", "content + FiLM + gate", "teal");
    addNode(s, 976, 414, 178, 118, "Residual\ndecoder", "resize-conv -> detail", "orange");
    addNode(s, 1076, 214, 130, 142, "Add", "base + alpha x\ngated high-pass\nresidual", "teal");
    addText(s, "Training only: Sentinel - base -> residual VAE", 542, 338, 480, 30, 17, C.gray, { italic: true, align: "center" });
    addCallout(s, "Output: 3 x 384 x 384 Sentinel-like RGB estimate; validation may set residual scale alpha to zero.", 150, 574, 980, 44, C.orange, 20);
    setNotes(s, "Explain the two responsibility paths. The top path reconstructs measured low-frequency structure. The bottom path proposes high-frequency residual detail using all six Landsat bands. The residual is gated and can be rejected.", [
      "Project source: src/geodiff_gan/models/system.py on branch SR-3x",
      "https://arxiv.org/abs/2108.10257",
      "https://arxiv.org/abs/2410.10812",
      "https://arxiv.org/abs/2405.04356",
    ]);
  }

  // 19. Base detailed.
  {
    const s = addSlide(p, "SwinIR anchors color and structure before any stochastic detail", "Model", 19);
    const xs = [58, 260, 476, 704, 922, 1090];
    for (let i = 0; i < xs.length - 1; i += 1) addArrow(s, xs[i] + 154, 290, 42, 14, C.gray);
    addNode(s, xs[0], 232, 150, 130, "RGB input", "3 x 128 x 128", "orange");
    addNode(s, xs[1], 232, 160, 130, "Shallow conv", "3 x 3\nC = 32", "navy");
    addNode(s, xs[2], 232, 176, 130, "Window\nattention", "4 shifted blocks\n8 x 8, 4 heads", "navy");
    addNode(s, xs[3], 232, 164, 130, "Body + skip", "feature residual", "navy");
    addNode(s, xs[4], 232, 160, 130, "Resize-conv", "bilinear 3x\n+ 3 x 3 conv", "teal");
    addNode(s, xs[5], 232, 132, 130, "Base HR", "3 x 384 x 384", "teal");
    addText(s, "+ bicubic RGB", 910, 388, 190, 28, 18, C.gray, { align: "center" });
    addCallout(s, "The base is the fallback. Auxiliary NIR/SWIR channels cannot directly alter its RGB reconstruction.", 144, 494, 990, 56, C.orange, 22);
    addText(s, "Current code: image-space upsampling uses resize-convolution.", 232, 580, 816, 28, 18, C.teal, { bold: true, align: "center" });
    setNotes(s, "The SwinIR-style branch predicts an RGB residual over bicubic interpolation. In the guided model base_input_channels=3, so NIR/SWIR are excluded from this path.", [
      "Project source: src/geodiff_gan/models/base.py",
      "https://arxiv.org/abs/2108.10257",
    ]);
  }

  // 20. VAE.
  {
    const s = addSlide(p, "The residual VAE learns a compact language for what the base missed", "Model", 20);
    const xs = [68, 280, 500, 720, 940];
    for (let i = 0; i < xs.length - 1; i += 1) addArrow(s, xs[i] + 166, 276, 38, 14, C.gray);
    addNode(s, xs[0], 208, 160, 148, "Target\nresidual", "Sentinel - base\n3 x 384 x 384", "orange");
    addNode(s, xs[1], 208, 170, 148, "Encoder", "24@384 -> 48@192\n-> 96@96 -> 96@48", "navy");
    addNode(s, xs[2], 208, 170, 148, "Moments", "mean + log variance\n4 x 48 x 48 each", "navy");
    addNode(s, xs[3], 208, 170, 148, "Latent z", "reparameterized\n4 x 48 x 48", "teal");
    addNode(s, xs[4], 208, 190, 148, "VAE decoder", "residual reconstruction\nused for preparation", "teal");
    addCallout(s, "Training: the VAE encoder sees Sentinel - base; KL regularization organizes the latent space.", 124, 430, 1018, 50, C.orange, 20);
    addCallout(s, "Inference: the Sentinel target is unavailable; diffusion samples the residual latent instead.", 124, 516, 1018, 50, C.teal, 20);
    setNotes(s, "The VAE itself is not the super-resolution model. It is trained on target-minus-base residuals and provides the latent space in which diffusion operates.", ["Project source: src/geodiff_gan/models/vae.py"]);
  }

  // 21. Diffusion.
  {
    const s = addSlide(p, "Latent diffusion models multiple plausible residuals at 48 x 48", "Model", 21);
    addText(s, "Down path", 98, 170, 170, 28, 22, C.navy, { bold: true });
    addText(s, "Up path", 956, 170, 170, 28, 22, C.teal, { bold: true, align: "right" });
    const down = [
      [120, 230, "48 @ 48", "input latent + LR evidence"],
      [328, 270, "96 @ 24", "cross-attention"],
      [536, 310, "144 @ 12", "cross-attention"],
      [744, 350, "192 @ 6", "bottleneck"],
    ];
    down.forEach((node, i) => {
      if (i < down.length - 1) addArrow(s, node[0] + 154, node[1] + 54, 42, 14, C.gray);
      addNode(s, node[0], node[1], 150, 110, node[2], node[3], i < 2 ? "navy" : "teal");
    });
    addArrow(s, 918, 396, 48, 14, C.gray);
    addNode(s, 976, 338, 190, 126, "Velocity prediction", "4 x 48 x 48\nDDIM sampling", "orange");
    addText(s, "Conditions: timestep + mode + optional degradation/text + LR feature map", 168, 504, 944, 34, 21, C.dark, { bold: true, align: "center" });
    addCallout(s, "Current code detail: the latent U-Net still uses subpixel upsampling internally; only image-space base/decoder are resize-conv.", 126, 568, 1028, 48, C.orange, 18);
    setNotes(s, "Diffusion represents one-to-many uncertainty, but plausibility is not fidelity. Multiple samples support uncertainty estimation. Mention the current internal upsampler accurately as an implementation detail for future ablation.", [
      "Project source: src/geodiff_gan/models/diffusion.py",
      "https://arxiv.org/abs/2410.10812",
    ]);
  }

  // 22. Mapper decoder.
  {
    const s = addSlide(p, "GeoMapper decides what detail to decode and where to trust it", "Model", 22);
    addArrow(s, 260, 316, 60, 16, C.gray);
    addArrow(s, 600, 316, 60, 16, C.gray);
    addArrow(s, 948, 316, 56, 16, C.gray);
    addNode(s, 70, 246, 180, 150, "Inputs", "latent 4 x 48 x 48\nLR f64 48 x 64 x 64", "navy");
    addNode(s, 330, 220, 260, 202, "GeoMapper", "4 residual blocks\ncontent: 48 x 48 x 48\n4 FiLM styles: 96\nevidence: 1 x 48 x 48", "teal");
    addNode(s, 670, 220, 268, 202, "Residual decoder", "stages: 48 -> 96 -> 192 -> 384\nchannels: 48 / 36 / 24 / 24\nLR skip projections\nresize-convolution", "orange");
    addNode(s, 1014, 246, 190, 150, "Controlled residual", "high-pass\nx evidence confidence\nx validation alpha", "teal");
    addCallout(s, "Evidence confidence controls reconstruction detail. Edit permission is a separate policy used only for labeled synthetic edits.", 138, 520, 1004, 58, C.navy, 21);
    setNotes(s, "The decoder has two heads in code: detail residual and edit residual. SR mode high-pass filters the detail head and multiplies it by evidence confidence before adding it to the base.", [
      "Project source: src/geodiff_gan/models/generator.py",
      "Project source: src/geodiff_gan/models/system.py",
      "https://arxiv.org/abs/2405.04356",
    ]);
  }

  // 23. Prompt modes.
  {
    const s = addSlide(p, "Reconstruction and prompt editing are separate operating modes", "Model", 23);
    addTwoColumnBullets(s,
      "SR mode",
      ["Null or weak prompt", "Evidence-gated high-pass residual", "Report fidelity and consistency", "Label: reconstructed estimate"],
      "Edit mode",
      ["Strong counterfactual prompt", "Separate edit-permission gate", "Soft LR consistency", "Metadata: synthetic_edit=true"],
      { leftColor: C.teal, rightColor: C.orange });
    addCallout(s, "Prompt conditioning is disabled in the current PSNR/SSIM experiment.", 226, 562, 828, 44, C.navy, 22);
    setNotes(s, "Do not mix prompt-edit results with reconstruction metrics. Edit mode is planned as a separately labeled synthetic visualization capability.");
  }

  // 24. Training.
  {
    const s = addSlide(p, "A staged curriculum prevents all modules from drifting at once", "Training", 24);
    const stages = [
      ["1", "Base", "pixel + structure"],
      ["2", "VAE", "residual representation"],
      ["3", "Diffusion", "latent denoising"],
      ["4", "Joint", "controlled residual"],
      ["5", "Edit", "future adapters"],
    ];
    for (let i = 0; i < 4; i += 1) addArrow(s, 260 + i * 224, 280, 48, 14, C.gray);
    stages.forEach((stage, i) => {
      addNode(s, 80 + i * 224, 220, 170, 140, `${stage[0]}. ${stage[1]}`, stage[2], i === 3 ? "orange" : i === 4 ? "gray" : "teal");
    });
    addText(s, "Current fidelity-first joint policy", 88, 430, 450, 30, 24, C.navy, { bold: true });
    addBullets(s, ["maximum 8 epochs | LR <= 5e-6", "freeze diffusion | disable GAN/perceptual terms", "early stop on validation PSNR"], 98, 478, 1040, 50, 19, C.orange);
    setNotes(s, "Stage-wise checkpoints initialize the next stage. During inference the final joint checkpoint contains the state dictionaries for the complete model, so a separate model object per stage is not loaded.");
  }

  // 25. Losses.
  {
    const s = addSlide(p, "Each loss asks a different fidelity question", "Training", 25);
    addSimpleTable(s,
      ["Loss", "Question it asks", "Current role"],
      [
        ["MSE", "Are pixel values correct?", "primary PSNR driver"],
        ["SSIM", "Are local structure and contrast correct?", "primary"],
        ["Multiscale MSE", "Does fidelity survive at several scales?", "supporting"],
        ["Radiometric", "Are channel mean and contrast stable?", "supporting"],
        ["Residual / base guard", "Does correction match what base missed?", "regression control"],
      ], 74, 162, [230, 620, 280], 66, 17);
    addCallout(s, "A loss weight is not an accuracy percentage; inspect weighted contributions and validation metrics together.", 154, 566, 972, 44, C.orange, 19);
    setNotes(s, "The supervisor-directed objective is currently distortion fidelity. Adversarial, wavelet, and perceptual losses should return only after the model proves it can beat the base.");
  }

  // 26. Metrics.
  {
    const s = addSlide(p, "PSNR and SSIM are primary; other metrics diagnose failures", "Evaluation", 26);
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
    const s = addSlide(p, "Diagnostics expose mechanisms hidden by aggregate scores", "Evaluation", 27);
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
    const s = addSlide(p, "Sharper edges have not yet improved fidelity", "Evidence", 28, "Single split | 28 test patches | 4 stochastic samples | 20 diffusion steps");
    addSimpleTable(s,
      ["Experiment", "L1 down", "PSNR up", "SSIM up", "Edge F1 up"],
      [
        ["RGB standard", "0.01426", "33.44", "0.8580", "0.0712"],
        ["RGB fidelity", "0.01507", "33.29", "0.8510", "0.0340"],
        ["Multispectral fidelity", "0.01546", "32.97", "0.8538", "0.0904"],
      ], 92, 182, [340, 180, 180, 180, 180], 72, 19);
    addMetricBar(s, "Best PSNR", "RGB standard", 0.88, 126, 470, 1020, C.teal);
    addMetricBar(s, "Best edge F1", "Multispectral", 0.70, 126, 516, 1020, C.orange);
    addCallout(s, "These values rank three full models; they do not prove improvement over the exact embedded SwinIR base.", 142, 580, 998, 38, C.navy, 18);
    setNotes(s, "Report the negative result plainly. Adding fidelity weights or sending six bands through the whole network did not reliably improve PSNR/SSIM. The sample size is diagnostic, not conclusive.", ["Researcher-supplied preliminary evaluation table from the SR-3x experiment suite."]);
  }

  // 29. Failure causes.
  {
    const s = addSlide(p, "The underperformance points to data and responsibility mismatch, not insufficient model size", "Evidence", 29);
    addBullets(s, [
      "Residual Landsat-Sentinel radiometric mismatch remains in the pair.",
      "The old multispectral design allowed NIR/SWIR to shift base RGB.",
      "Diffusion may produce plausible but pixel-inaccurate residual texture.",
      "Joint tuning can overfit the decoder to training-specific patterns.",
      "Registration and temporal change impose an upper bound on PSNR.",
      "A small test set creates unstable rankings.",
    ], 86, 142, 1110, 72, 20, C.orange);
    addCallout(s, "The next step is controlled isolation of calibration, guidance, and residual acceptance - not a larger GAN weight.", 126, 596, 1028, 40, C.teal, 19);
    setNotes(s, "Explain that a larger network can memorize mismatch more effectively. The improvement plan changes responsibility boundaries before changing capacity.");
  }

  // 30. New improvement.
  {
    const s = addSlide(p, "Harmonized RGB separates base reconstruction from multispectral guidance", "Improvement", 30);
    addArrow(s, 306, 260, 54, 14, C.gray);
    addArrow(s, 600, 260, 54, 14, C.gray);
    addArrow(s, 902, 260, 54, 14, C.gray);
    addNode(s, 80, 200, 216, 140, "Train patches only", "fit per-channel\nslope + offset", "gray");
    addNode(s, 370, 200, 220, 140, "Harmonized RGB", "Sentinel_30m = a x\nLandsat_30m + b", "orange");
    addNode(s, 664, 200, 228, 140, "Dual responsibility", "RGB -> base\n6 bands -> residual", "teal");
    addNode(s, 966, 200, 230, 140, "Conservative joint", "freeze diffusion\nselect residual alpha", "navy");
    addCallout(s, "Data protection: fit calibration on train only and save the coefficients.", 112, 454, 1038, 48, C.orange, 20);
    addCallout(s, "Model protection: base guard prevents regression; validation may return alpha = 0.", 112, 534, 1038, 48, C.teal, 20);
    setNotes(s, "Calibration is fitted at 30 m from training pairs only. Validation and test targets never influence the coefficients. NIR/SWIR can guide the residual pathway but cannot directly alter base RGB.", [
      "Project source: src/geodiff_gan/data/radiometry.py",
      "Project source: configs/landsat_sentinel_3x_multispectral_guided.yaml",
      "https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf",
    ]);
  }

  // 31. Experiment matrix.
  {
    const s = addSlide(p, "Four controlled variants isolate calibration from multispectral guidance", "Improvement", 31);
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
    addCallout(s, "Validation selects alpha from 0, 0.25, 0.5, 0.75, 1.0; alpha=0 is the safe fallback.", 188, 586, 904, 38, C.orange, 18);
    setNotes(s, "Predefine the criteria before examining test results. The same validation rule must be applied to all models, and test targets must not select alpha.");
  }

  // 32. Novelty.
  {
    const s = addSlide(p, "Novelty depends on demonstrated integration, not a list of familiar blocks", "Contribution", 32);
    addTwoColumnBullets(s,
      "Established components",
      ["SwinIR", "residual VAE", "conditional diffusion U-Net", "FiLM decoder and wavelet GAN"],
      "Candidate contribution",
      ["deterministic-stochastic residual split", "separate evidence and edit policies", "RGB-base / multispectral-residual separation", "validation fallback to the base"],
      { leftColor: C.gray, rightColor: C.teal });
    addCallout(s, "A candidate contribution becomes defensible only after controlled ablations show a reproducible benefit.", 150, 570, 980, 44, C.orange, 20);
    setNotes(s, "The inspiration is residual decomposition in HART, diffusion-to-GAN mapping and style modulation, and high-frequency wavelet discrimination. The satellite-specific claim is the evidence-controlled cross-sensor integration and its validated behavior.", [
      "https://arxiv.org/abs/2410.10812",
      "https://arxiv.org/abs/2405.04356",
      "https://arxiv.org/abs/2402.19215",
    ]);
  }

  // 33. Completed.
  {
    const s = addSlide(p, "The software pipeline is complete enough for controlled experiments", "Progress", 33);
    addBullets(s, [
      "Synthetic 4x and real Landsat-to-Sentinel 3x data pipelines",
      "QA masking, pairing, spatial splits, quarantine, and cached NPZ manifests",
      "RGB and six-band small/medium GeoDiff-GAN variants",
      "Stage-wise training, resume, early stopping, best/latest checkpoints",
      "Debug tensors, Fourier/wavelet maps, uncertainty, and per-patch metrics",
      "Train-only calibration and dual-stream model path implemented",
    ], 82, 146, 1120, 73, 20, C.teal);
    addCallout(s, "Experiments are isolated by resolved config, manifest, calibration snapshot, and output directory.", 154, 596, 972, 40, C.navy, 19);
    setNotes(s, "Separate software completion from scientific validation. The implementation exists; the next experiments determine whether the mechanisms improve held-out fidelity.");
  }

  // 34. Limitations.
  {
    const s = addSlide(p, "The reference pair contains uncertainty that no model can remove", "Limits", 34);
    addBullets(s, [
      "Sentinel is a paired reference observation, not perfect hidden ground truth.",
      "Misregistration can teach false edges and reduce otherwise reasonable PSNR.",
      "Global affine calibration cannot remove nonlinear band-response differences.",
      "A three-day gap does not guarantee no land-cover or atmospheric change.",
      "Generated detail may be plausible but false.",
      "Current geographic diversity is insufficient for broad deployment claims.",
    ], 88, 144, 1110, 73, 20, C.orange);
    addCallout(s, "Describe the output as a Sentinel-like estimate conditioned on Landsat - never as an observed 10 m Landsat image.", 124, 594, 1032, 42, C.navy, 19);
    setNotes(s, "Keep raw products, masks, provenance, uncertainty, and reference imagery available to the analyst. Prompt edits require an explicit synthetic label.");
  }

  // 35. Remaining.
  {
    const s = addSlide(p, "Remaining work moves from diagnosis to defensible evidence", "Next", 35);
    addBullets(s, [
      "Run harmonized RGB and multispectral-guided experiments.",
      "Compare base versus final per patch, per tile, and by scene type.",
      "Audit calibration coefficients and registration-shift distributions.",
      "Run spatial K-fold and complete-tile holdout evaluation.",
      "Add recent open-source baselines, seeds, and confidence intervals.",
      "Reintroduce perceptual / wavelet / GAN objectives only after fidelity gains.",
      "Keep QGIS as the current analysis tool; evaluate ArcGIS integration later.",
    ], 84, 128, 1120, 68, 19, C.teal);
    setNotes(s, "Prioritize data audit and base-relative evaluation. Model complexity is not the first remaining dependency.");
  }

  // 36. Outcomes.
  {
    const s = addSlide(p, "Any of three outcomes can produce a scientifically useful thesis", "Next", 36);
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
    addCallout(s, "A controlled negative result is stronger than selecting only favorable images.", 222, 536, 836, 46, C.navy, 22);
    setNotes(s, "Make the decision rule explicit: held-out measurements and pair audits determine the conclusion, not preference for the more complex architecture.");
  }

  // 37. Conclusion.
  {
    const s = addSlide(p, "The contribution is an evidence-controlled cross-sensor framework", "Conclusion", 37);
    addText(s, "01", 90, 174, 70, 44, 30, C.teal, { bold: true });
    addText(s, "SR is useful when matching HR data are unavailable, but the output remains an estimate.", 178, 170, 950, 64, 23, C.dark, { bold: true });
    addRule(s, 90, 252, 1060, C.mid, 1);
    addText(s, "02", 90, 292, 70, 44, 30, C.teal, { bold: true });
    addText(s, "Real Landsat-to-Sentinel SR is also a harmonization, registration, and temporal-matching problem.", 178, 288, 950, 64, 23, C.dark, { bold: true });
    addRule(s, 90, 370, 1060, C.mid, 1);
    addText(s, "03", 90, 410, 70, 44, 30, C.teal, { bold: true });
    addText(s, "The proposed model confines stochastic detail with an RGB base, multispectral guidance, evidence gating, and validation fallback.", 178, 406, 950, 76, 23, C.dark, { bold: true });
    addCallout(s, "The next result must show that the learned residual adds measurable information beyond SwinIR.", 182, 560, 916, 50, C.orange, 21);
    setNotes(s, "Close by resolving the opening question. The central criterion is whether the residual improves held-out reconstruction without violating evidence or creating systematic artifacts.");
  }

  // 38. Discussion.
  {
    const s = addSlide(p, "Three decisions would strengthen the final study", "Discussion", 38);
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
    const s = addSlide(p, "Backup: core reconstruction equations", "Appendix", 39);
    addText(s, "x_base = B(y_RGB)", 122, 174, 1020, 50, 30, C.navy, { bold: true, align: "center" });
    addText(s, "r_hat = R(y_RGB,NIR,SWIR,z)", 122, 258, 1020, 50, 30, C.teal, { bold: true, align: "center" });
    addText(s, "x_hat = clip(x_base + alpha x confidence x high_pass(r_hat), 0, 1)", 84, 342, 1112, 60, 27, C.orange, { bold: true, align: "center" });
    addRule(s, 220, 440, 840, C.mid, 2);
    addText(s, "PSNR = -10 log10(MSE)  for normalized reflectance", 170, 478, 940, 46, 25, C.dark, { align: "center" });
    addCallout(s, "alpha is accepted only when validation improves over x_base.", 288, 570, 704, 40, C.teal, 20);
    setNotes(s, "Use y for Landsat, x for Sentinel, B for the deterministic base, and R for the residual generator. Explain each term before discussing the equation.");
  }

  // Backup 40. Tensors.
  {
    const s = addSlide(p, "Backup: small multispectral-guided tensor sizes", "Appendix", 40);
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
    const s = addSlide(p, "Backup: resize-convolution reduces image-space phase imbalance", "Appendix", 41);
    addTwoColumnBullets(s,
      "PixelShuffle risk",
      ["channels become spatial phases", "unequal phase learning can form lattice patterns", "requires Fourier inspection"],
      "Resize-convolution path",
      ["bilinear resize first", "ordinary 3 x 3 convolution second", "used in current base and residual decoder"],
      { leftColor: C.orange, rightColor: C.teal });
    addCallout(s, "It reduces one artifact mechanism; it does not guarantee artifact-free output.", 208, 558, 864, 44, C.navy, 21);
    setNotes(s, "Be precise: the current image-space paths use resize-convolution, but the latent diffusion U-Net retains an internal PixelShuffle upsampler that can be ablated separately.", ["Project source: src/geodiff_gan/models/base.py and generator.py"]);
  }

  // Backup 42. Reproducibility.
  {
    const s = addSlide(p, "Backup: every result must be reproducible from saved evidence", "Appendix", 42);
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
    const s = addSlide(p, "Backup: use labels that match the strength of the evidence", "Appendix", 43);
    addSimpleTable(s,
      ["Label", "Use when"],
      [
        ["Preliminary", "limited patches, one split, or one seed"],
        ["Diagnostic", "finding mechanisms and failures, not claiming generalization"],
        ["Reference target", "paired Sentinel observation, not hidden truth"],
        ["Generated estimate", "model output in reconstruction mode"],
        ["Candidate contribution", "mechanism still awaiting ablation"],
      ], 126, 168, [300, 730], 68, 19);
    addCallout(s, "Never replace PSNR, SSIM, MAE, ERGAS, SAM, UIQI, or sCC with the word 'accuracy'.", 166, 570, 948, 44, C.orange, 20);
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
