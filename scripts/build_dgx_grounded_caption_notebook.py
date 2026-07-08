from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "dgx" / "GeoDiff_GAN_DGX_Grounded_Multispectral_Captioning.ipynb"


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip() + "\n",
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip() + "\n",
    }


cells = [
    markdown(
        r"""
        # Grounded Sentinel-2 Captioning on DGX

        This notebook replaces RGB-only captioning with a conservative multispectral workflow:

        1. Load the prepared RGB HR patch.
        2. Read the matching B08, B11, and SCL windows from the original SAFE product.
        3. Compute NDVI, MNDWI, NDBI, connected water support, vegetation support, and cloud support.
        4. Show Qwen3-VL a labelled multispectral panel plus numerical evidence.
        5. Remove claims that contradict strong spectral/SCL evidence.
        6. Commit every completed caption to SQLite, allowing exact crash recovery.
        7. Export a backward-compatible JSONL file with `brief`, `descriptive`, and `analytical`.

        **Important limitation:** NDVI cannot be recovered from the RGB `.npz` alone. The original
        Sentinel-2 SAFE products must remain available because NDVI needs B08 and MNDWI/NDBI need
        B11. This workflow does not modify existing patches, manifests, checkpoints, or captions.
        """
    ),
    markdown("## 1. Paths and run settings"),
    code(
        r"""
        from pathlib import Path
        import os, sys

        # Use the Python 3.11 GeoDiff kernel created for the DGX workflow.
        HOME = Path.home()
        THESIS_ROOT = HOME / "geodiff_dgx"
        REPOSITORY_DIR = THESIS_ROOT / "geodiff-gan"
        DATASET_ROOT = THESIS_ROOT / "datasets" / "sentinel2-bharat"
        WORK_ROOT = THESIS_ROOT / "geodiff-output"

        # Prefer the final split manifest, then fall back to the raw prepared manifest.
        manifest_candidates = [
            WORK_ROOT / "manifest_dgx_80_10_10.jsonl",
            WORK_ROOT / "manifest_raw.jsonl",
            WORK_ROOT / "manifest.jsonl",
        ]
        MANIFEST = next((path for path in manifest_candidates if path.exists()), manifest_candidates[0])

        CAPTION_ROOT = WORK_ROOT / "captions_grounded"
        CAPTION_ROOT.mkdir(parents=True, exist_ok=True)
        CAPTION_DB = CAPTION_ROOT / "captions_grounded.sqlite"
        CAPTION_JSONL = CAPTION_ROOT / "captions_grounded_qwen3vl.jsonl"
        EVIDENCE_JSONL = CAPTION_ROOT / "spectral_evidence.jsonl"

        MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
        PREFERRED_CAPTION = "descriptive"
        DISPLAY_GAIN = 2.5
        MAX_NEW_TOKENS = 320
        CHECKPOINT_EVERY = 1            # SQLite commit interval; 1 is safest.
        EXPORT_EVERY = 25               # Atomic JSONL export interval.
        CAPTION_SPLIT = "all"           # "all", "train", "val", or "test".
        START_INDEX = 0
        END_INDEX = None                # Example: 100 for a trial run.
        RETRY_GENERATION = 2
        FORCE_RECATION_INDICES = []     # Example: [12, 91]. Existing rows are replaced.

        # A100 40 GB: BF16 gives better caption quality than 4-bit quantization and fits this model.
        USE_4BIT = False
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        print("Python:", sys.version)
        print("Repository:", REPOSITORY_DIR)
        print("Dataset:", DATASET_ROOT)
        print("Manifest:", MANIFEST)
        print("Caption DB:", CAPTION_DB)
        print("Caption JSONL:", CAPTION_JSONL)

        if sys.version_info < (3, 10):
            raise RuntimeError("Select the GeoDiff Python 3.11 kernel before continuing.")
        if not MANIFEST.exists():
            raise FileNotFoundError(f"No prepared manifest found: {manifest_candidates}")
        if not DATASET_ROOT.exists():
            raise FileNotFoundError(f"Original SAFE dataset is required for B08/B11: {DATASET_ROOT}")
        """
    ),
    markdown("## 2. Install caption dependencies without replacing CUDA PyTorch"),
    code(
        r"""
        import subprocess

        def run(command, cwd=None):
            command = [str(item) for item in command]
            print("+", " ".join(command), flush=True)
            subprocess.run(command, cwd=cwd, check=True)

        # Do not install or upgrade torch on the supervisor's DGX image.
        run([
            sys.executable, "-m", "pip", "install", "-q",
            "transformers>=4.57", "accelerate>=1.0", "qwen-vl-utils>=0.0.14",
            "rasterio>=1.3", "scipy>=1.10", "matplotlib>=3.7",
            "pandas>=2", "tqdm>=4.66", "Pillow>=10",
        ])
        if USE_4BIT:
            run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes>=0.46"])
        print("Dependencies installed. Restart the kernel only if imports in the next cell fail.")
        """
    ),
    markdown("## 3. Verify GPU, load manifest, and map records to canonical SAFE products"),
    code(
        r"""
        import json, re
        from collections import Counter
        from pathlib import Path

        import numpy as np
        import torch
        from tqdm.auto import tqdm

        print("Torch:", torch.__version__)
        print("CUDA:", torch.cuda.is_available())
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Select the DGX GPU kernel.")
        print("GPU:", torch.cuda.get_device_name(0))
        print("BF16 supported:", torch.cuda.is_bf16_supported())
        torch.backends.cuda.matmul.allow_tf32 = True

        def read_jsonl(path):
            values = []
            with Path(path).open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        values.append(json.loads(line))
            return values

        records = read_jsonl(MANIFEST)
        if CAPTION_SPLIT != "all":
            records = [record for record in records if record.get("split") == CAPTION_SPLIT]
        records = records[START_INDEX:END_INDEX]
        for index, record in enumerate(records, start=START_INDEX):
            record["_manifest_index"] = index

        SENTINEL_ID = re.compile(r"(S2[A-Z]_MSIL2A_[A-Z0-9_]+\.SAFE)", re.I)

        def is_safe_root(path):
            path = Path(path)
            return path.is_dir() and (path / "manifest.safe").is_file() and (path / "GRANULE").is_dir()

        def product_aliases(path):
            path = Path(path)
            names = {path.name.casefold(), path.stem.casefold()}
            match = SENTINEL_ID.search(path.name)
            if match:
                names.add(match.group(1).casefold())
                names.add(Path(match.group(1)).stem.casefold())
            parent = path.parent
            while parent != parent.parent:
                if parent.suffix.casefold() == ".safe":
                    names.add(parent.name.casefold())
                    names.add(parent.stem.casefold())
                    match = SENTINEL_ID.search(parent.name)
                    if match:
                        names.add(match.group(1).casefold())
                        names.add(Path(match.group(1)).stem.casefold())
                parent = parent.parent
            return names

        def record_aliases(record):
            value = str(record.get("source_product", ""))
            names = {Path(value).name.casefold(), Path(value).stem.casefold()}
            match = SENTINEL_ID.search(value)
            if match:
                names.add(match.group(1).casefold())
                names.add(Path(match.group(1)).stem.casefold())
            return {name for name in names if name}

        safe_roots = [path for path in DATASET_ROOT.rglob("*.SAFE") if is_safe_root(path)]
        safe_map = {}
        for safe in safe_roots:
            for alias in product_aliases(safe):
                safe_map.setdefault(alias, safe)

        def safe_for_record(record):
            matches = {safe_map[alias] for alias in record_aliases(record) if alias in safe_map}
            if len(matches) == 1:
                return next(iter(matches))
            if not matches:
                tile = record.get("tile_id", "")
                tile_matches = [path for path in safe_roots if f"_T{tile}_" in path.name]
                if len(tile_matches) == 1:
                    return tile_matches[0]
                raise KeyError(
                    f"No unique SAFE match for source_product={record.get('source_product')!r}, "
                    f"tile={tile!r}; tile matches={len(tile_matches)}"
                )
            raise KeyError(f"Ambiguous SAFE mapping for {record.get('source_product')}: {matches}")

        missing = []
        for record in records:
            try:
                record["_safe"] = str(safe_for_record(record))
            except Exception as error:
                missing.append((record["_manifest_index"], record.get("source_product"), str(error)))

        print("Caption records:", len(records))
        print("Splits:", Counter(record.get("split") for record in records))
        print("Canonical SAFE roots:", len(safe_roots))
        print("Unmatched records:", len(missing))
        if missing:
            print("First unmatched records:", missing[:10])
            raise RuntimeError("Fix SAFE mapping before captioning; no RGB-only fallback is used.")
        """
    ),
    markdown("## 4. Multispectral evidence extraction and labelled visual panel"),
    code(
        r"""
        import io
        from functools import lru_cache

        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import rasterio
        from PIL import Image, ImageDraw, ImageFont
        from rasterio.enums import Resampling
        from rasterio.windows import Window, bounds, from_bounds
        from scipy import ndimage

        REFLECTANCE_SCALE = 10000.0
        EPSILON = 1e-6
        WATER_WORDS = re.compile(
            r"\b(water|river|stream|canal|lake|pond|reservoir|wetland|coast|coastal|shoreline)\b",
            re.I,
        )
        CLOUD_WORDS = re.compile(r"\b(cloud|clouds|cloudy|haze|hazy)\b", re.I)
        SNOW_WORDS = re.compile(r"\b(snow|ice|glacier)\b", re.I)
        DENSE_VEGETATION_WORDS = re.compile(
            r"\b(dense vegetation|dense forest|forest canopy|woodland|lush vegetation)\b",
            re.I,
        )

        def find_band(product, pattern):
            matches = sorted(Path(product).rglob(pattern))
            if not matches:
                raise FileNotFoundError(f"Missing {pattern} below {product}")
            return matches[0]

        @lru_cache(maxsize=32)
        def band_paths(product_text):
            product = Path(product_text)
            return {
                "red": find_band(product, "*_B04_10m.jp2"),
                "green": find_band(product, "*_B03_10m.jp2"),
                "blue": find_band(product, "*_B02_10m.jp2"),
                "nir": find_band(product, "*_B08_10m.jp2"),
                "swir": find_band(product, "*_B11_20m.jp2"),
                "scl": find_band(product, "*_SCL_20m.jp2"),
            }

        def resolve_patch_path(record):
            patch = Path(record["patch"])
            candidates = [
                patch,
                WORK_ROOT / patch,
                MANIFEST.parent / patch,
                THESIS_ROOT / patch,
            ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate
            raise FileNotFoundError(f"Patch not found; tried: {candidates}")

        def normalized_difference(a, b):
            return np.clip((a - b) / (a + b + EPSILON), -1, 1)

        def largest_component_fraction(mask):
            labels, count = ndimage.label(mask)
            if count == 0:
                return 0.0
            sizes = np.bincount(labels.ravel())[1:]
            return float(sizes.max() / mask.size) if sizes.size else 0.0

        def robust_fraction(mask, valid):
            denominator = max(int(valid.sum()), 1)
            return float((mask & valid).sum() / denominator)

        def extract_evidence(record):
            patch_path = resolve_patch_path(record)
            with np.load(patch_path) as data:
                hr = np.asarray(data["hr"], dtype=np.float32)
                valid_npz = (
                    np.asarray(data["valid_mask"]).astype(bool)
                    if "valid_mask" in data
                    else np.ones(hr.shape[-2:], dtype=bool)
                )
            if hr.shape[0] == 3:
                rgb = np.moveaxis(hr, 0, -1)
            else:
                rgb = hr
            height, width = rgb.shape[:2]
            paths = band_paths(record["_safe"])

            with (
                rasterio.open(paths["red"]) as red_ds,
                rasterio.open(paths["nir"]) as nir_ds,
                rasterio.open(paths["swir"]) as swir_ds,
                rasterio.open(paths["scl"]) as scl_ds,
            ):
                window = Window(int(record["col"]), int(record["row"]), width, height)
                red = red_ds.read(1, window=window).astype(np.float32) / REFLECTANCE_SCALE
                nir = nir_ds.read(1, window=window).astype(np.float32) / REFLECTANCE_SCALE
                geo_bounds = bounds(window, red_ds.transform)
                swir_window = from_bounds(*geo_bounds, transform=swir_ds.transform)
                scl_window = from_bounds(*geo_bounds, transform=scl_ds.transform)
                swir = swir_ds.read(
                    1, window=swir_window, out_shape=(height, width),
                    resampling=Resampling.bilinear, boundless=True, fill_value=0,
                ).astype(np.float32) / REFLECTANCE_SCALE
                scl = scl_ds.read(
                    1, window=scl_window, out_shape=(height, width),
                    resampling=Resampling.nearest, boundless=True, fill_value=0,
                )

            green = rgb[..., 1]
            valid = valid_npz & np.isfinite(red) & np.isfinite(nir) & np.isfinite(swir)
            valid &= (red > 0) & (nir > 0) & (swir > 0)

            ndvi = normalized_difference(nir, red)
            ndwi = normalized_difference(green, nir)
            mndwi = normalized_difference(green, swir)
            ndbi = normalized_difference(swir, nir)

            # Conservative support masks. SCL class 6 is direct water evidence.
            scl_water = scl == 6
            spectral_water = (mndwi > 0.12) & (ndvi < 0.15) & (nir < 0.18)
            water_mask = valid & (scl_water | spectral_water)
            vegetation_mask = valid & (ndvi > 0.35)
            dense_vegetation_mask = valid & (ndvi > 0.55)
            built_candidate_mask = valid & (ndbi > 0.08) & (ndvi < 0.25) & (mndwi < 0.0)
            cloud_mask = np.isin(scl, [8, 9, 10])
            snow_mask = scl == 11

            water_fraction = robust_fraction(water_mask, valid)
            water_largest = largest_component_fraction(water_mask)
            scl_water_fraction = robust_fraction(scl_water, valid)
            vegetation_fraction = robust_fraction(vegetation_mask, valid)
            dense_vegetation_fraction = robust_fraction(dense_vegetation_mask, valid)
            cloud_fraction = robust_fraction(cloud_mask, np.ones_like(valid))
            snow_fraction = robust_fraction(snow_mask, np.ones_like(valid))

            water_supported = bool(
                scl_water_fraction >= 0.003
                or (water_fraction >= 0.006 and water_largest >= 0.002)
            )
            dense_vegetation_supported = bool(dense_vegetation_fraction >= 0.03)
            cloud_supported = bool(cloud_fraction >= 0.005)
            snow_supported = bool(snow_fraction >= 0.003)

            valid_indices = valid
            def stats(array):
                values = array[valid_indices]
                if values.size == 0:
                    return {"mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
                return {
                    "mean": float(values.mean()),
                    "p10": float(np.quantile(values, 0.10)),
                    "p50": float(np.quantile(values, 0.50)),
                    "p90": float(np.quantile(values, 0.90)),
                }

            evidence = {
                "valid_fraction": float(valid.mean()),
                "ndvi": stats(ndvi),
                "ndwi": stats(ndwi),
                "mndwi": stats(mndwi),
                "ndbi": stats(ndbi),
                "vegetation_fraction": vegetation_fraction,
                "dense_vegetation_fraction": dense_vegetation_fraction,
                "built_candidate_fraction": robust_fraction(built_candidate_mask, valid),
                "water_fraction": water_fraction,
                "water_largest_component_fraction": water_largest,
                "scl_water_fraction": scl_water_fraction,
                "cloud_fraction": cloud_fraction,
                "snow_fraction": snow_fraction,
                "water_supported": water_supported,
                "dense_vegetation_supported": dense_vegetation_supported,
                "cloud_supported": cloud_supported,
                "snow_supported": snow_supported,
                "water_rule": (
                    "Water terms are allowed."
                    if water_supported
                    else "Water, river, lake, reservoir, canal, wetland, coast, and shoreline terms are forbidden."
                ),
            }
            arrays = {
                "rgb": np.clip(rgb, 0, 1),
                "false_color": np.clip(np.stack([nir, red, green], axis=-1), 0, 1),
                "ndvi": ndvi,
                "mndwi": mndwi,
                "ndbi": ndbi,
                "support": np.stack(
                    [water_mask, vegetation_mask, built_candidate_mask], axis=-1
                ).astype(np.float32),
            }
            return evidence, arrays

        def uint8_rgb(array, gain=1.0):
            return (np.clip(array * gain, 0, 1) * 255).round().astype(np.uint8)

        def colorize_index(array, cmap_name):
            normalized = np.clip((array + 1.0) / 2.0, 0, 1)
            return (mpl.colormaps[cmap_name](normalized)[..., :3] * 255).astype(np.uint8)

        def add_label(image, label):
            image = image.copy()
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, image.width, 34), fill=(245, 247, 250))
            draw.text((10, 8), label, fill=(15, 30, 50))
            return image

        def build_panel(arrays, tile_size=384):
            tiles = [
                add_label(Image.fromarray(uint8_rgb(arrays["rgb"], DISPLAY_GAIN)), "True color RGB"),
                add_label(Image.fromarray(uint8_rgb(arrays["false_color"], DISPLAY_GAIN)), "False color: NIR / red / green"),
                add_label(Image.fromarray(colorize_index(arrays["ndvi"], "RdYlGn")), "NDVI: green = vegetation"),
                add_label(Image.fromarray(colorize_index(arrays["mndwi"], "BrBG")), "MNDWI: positive may support water"),
                add_label(Image.fromarray(colorize_index(arrays["ndbi"], "PuOr")), "NDBI: positive is built/bare candidate"),
                add_label(Image.fromarray(uint8_rgb(arrays["support"])), "Support mask: R water, G vegetation, B built candidate"),
            ]
            tiles = [tile.resize((tile_size, tile_size), Image.Resampling.LANCZOS) for tile in tiles]
            panel = Image.new("RGB", (tile_size * 3, tile_size * 2), "white")
            for index, tile in enumerate(tiles):
                panel.paste(tile, ((index % 3) * tile_size, (index // 3) * tile_size))
            return panel

        # Test exact geospatial alignment before loading the VLM.
        TEST_INDEX = 0
        test_record = records[TEST_INDEX]
        test_evidence, test_arrays = extract_evidence(test_record)
        test_panel = build_panel(test_arrays)
        display(test_panel)
        print(json.dumps(test_evidence, indent=2))
        """
    ),
    markdown("## 5. Prompt, strict parsing, and evidence-based claim validation"),
    code(
        r"""
        ALLOWED_DENSITY = {"sparse", "moderate", "dense", "mixed", "unclear"}
        ALLOWED_TERRAIN = {"flat", "hilly", "mountainous", "riverine", "arid", "mixed", "unclear"}
        ALLOWED_TEXTURE = {"smooth", "fine-grained", "coarse", "grid-like", "linear", "mixed", "unclear"}

        def evidence_text(evidence):
            return json.dumps(evidence, ensure_ascii=True, indent=2, sort_keys=True)

        def caption_prompt(evidence, correction=None):
            correction_text = f"\nPrevious-output correction required:\n{correction}\n" if correction else ""
            schema = {
                "brief": "maximum 18 words",
                "descriptive": "one or two neutral evidence-grounded sentences",
                "analytical": {
                    "land_cover": ["conservative visible classes"],
                    "visible_objects": ["only structures visibly resolved in RGB"],
                    "object_density": "sparse | moderate | dense | mixed | unclear",
                    "terrain": "flat | hilly | mountainous | riverine | arid | mixed | unclear",
                    "texture": "smooth | fine-grained | coarse | grid-like | linear | mixed | unclear",
                    "spatial_layout": "short visible spatial arrangement",
                    "water_presence": "absent | possible | present",
                    "vegetation_level": "very low | low | moderate | high | mixed",
                    "uncertainty": "low | medium | high",
                },
            }
            rules = [
                "Do not infer a city, country, coordinates, ownership, people, date, event, crop species, or building use.",
                "Do not call dark pixels water unless water_supported is true.",
                "If water_supported is false, do not use any water-related term anywhere.",
                "NDBI is not proof of buildings; use RGB geometry before saying built-up area.",
                "NDVI supports vegetation amount but does not prove forest or agriculture by itself.",
                "Describe roads, blocks, fields, ridges, quarries, or settlements only when their geometry is visible.",
                'Prefer "possible" and "unclear" over invention.',
                "Never mention the panel, indices, bands, masks, prompt, or numerical values in the caption.",
            ]
            return "\n".join([
                "You are producing a conservative caption for one Sentinel-2 L2A patch.",
                "",
                "The attached six-panel image contains true-color RGB, false color, NDVI, MNDWI, NDBI, and a support mask.",
                "Numerical evidence was computed from co-registered B02, B03, B04, B08, B11 and SCL.",
                "Treat explicit support booleans and the water rule as constraints.",
                "Spectral indices are supporting evidence, not proof of exact object identity.",
                "",
                "NUMERICAL EVIDENCE:",
                evidence_text(evidence),
                "",
                "Return one strict JSON object only:",
                json.dumps(schema, indent=2),
                "",
                "Rules:",
                *[f"- {rule}" for rule in rules],
                correction_text,
            ])

        def parse_json_object(text):
            stripped = text.strip()
            if stripped.startswith("```"):
                stripped = stripped.strip("`").removeprefix("json").strip()
            start, end = stripped.find("{"), stripped.rfind("}")
            if start < 0 or end <= start:
                raise ValueError("No JSON object in model response")
            value = json.loads(stripped[start : end + 1])
            if not isinstance(value, dict):
                raise ValueError("Caption response is not an object")
            return value

        def clean_string(value, fallback="unclear"):
            return value.strip() if isinstance(value, str) and value.strip() else fallback

        def clean_list(value):
            if not isinstance(value, list):
                return []
            output = []
            for item in value:
                if isinstance(item, str) and item.strip():
                    item = item.strip()
                    if item.casefold() not in {existing.casefold() for existing in output}:
                        output.append(item)
            return output[:10]

        def contains(pattern, payload):
            return bool(pattern.search(json.dumps(payload, ensure_ascii=True)))

        def validate_payload(payload, evidence):
            issues = []
            analytical = payload.get("analytical")
            if not isinstance(analytical, dict):
                analytical = {}
                issues.append("analytical object was missing")

            normalized = {
                "brief": clean_string(payload.get("brief"), ""),
                "descriptive": clean_string(payload.get("descriptive"), ""),
                "analytical": {
                    "land_cover": clean_list(analytical.get("land_cover")),
                    "visible_objects": clean_list(analytical.get("visible_objects")),
                    "object_density": clean_string(analytical.get("object_density")),
                    "terrain": clean_string(analytical.get("terrain")),
                    "texture": clean_string(analytical.get("texture")),
                    "spatial_layout": clean_string(analytical.get("spatial_layout")),
                    "water_presence": clean_string(analytical.get("water_presence"), "absent"),
                    "vegetation_level": clean_string(analytical.get("vegetation_level")),
                    "uncertainty": clean_string(analytical.get("uncertainty"), "high"),
                },
            }

            if not evidence["water_supported"] and contains(WATER_WORDS, normalized):
                issues.append("water-related claim contradicts water_supported=false")
            if not evidence["cloud_supported"] and contains(CLOUD_WORDS, normalized):
                issues.append("cloud/haze claim contradicts SCL cloud support")
            if not evidence["snow_supported"] and contains(SNOW_WORDS, normalized):
                issues.append("snow/ice claim contradicts SCL snow support")
            if not evidence["dense_vegetation_supported"] and contains(DENSE_VEGETATION_WORDS, normalized):
                issues.append("dense forest/vegetation claim contradicts NDVI support")
            if len(normalized["brief"].split()) > 18:
                issues.append("brief exceeds 18 words")
            if not normalized["descriptive"]:
                issues.append("descriptive caption is empty")
            if not normalized["brief"]:
                issues.append("brief caption is empty")
            return normalized, issues

        def analytical_to_text(value):
            if not isinstance(value, dict):
                return ""
            parts = []
            for key in ("land_cover", "visible_objects"):
                item = value.get(key)
                if isinstance(item, list) and item:
                    parts.append(f"{key.replace('_', ' ')}: " + ", ".join(item))
            for key in (
                "object_density", "terrain", "texture", "spatial_layout",
                "water_presence", "vegetation_level", "uncertainty",
            ):
                item = value.get(key)
                if isinstance(item, str) and item:
                    parts.append(f"{key.replace('_', ' ')}: {item}")
            return "; ".join(parts)

        def safe_fallback(payload, evidence):
            # Remove unsupported claims if retries still violate hard evidence.
            analytical = payload["analytical"]
            patterns = []
            if not evidence["water_supported"]:
                patterns.append(WATER_WORDS)
                analytical["water_presence"] = "absent"
            if not evidence["cloud_supported"]:
                patterns.append(CLOUD_WORDS)
            if not evidence["snow_supported"]:
                patterns.append(SNOW_WORDS)
            if not evidence["dense_vegetation_supported"]:
                patterns.append(DENSE_VEGETATION_WORDS)

            def allowed(text):
                return not any(pattern.search(text) for pattern in patterns)

            analytical["land_cover"] = [item for item in analytical["land_cover"] if allowed(item)]
            analytical["visible_objects"] = [item for item in analytical["visible_objects"] if allowed(item)]
            if not allowed(payload["brief"]):
                payload["brief"] = ", ".join(analytical["land_cover"][:3]) or "mixed land surface"
            if not allowed(payload["descriptive"]):
                cover = ", ".join(analytical["land_cover"][:4]) or "mixed land surface"
                layout = analytical.get("spatial_layout", "unclear spatial arrangement")
                payload["descriptive"] = f"The patch shows {cover}, with {layout}."
            return payload

        def top_level_caption(payload):
            if PREFERRED_CAPTION == "brief":
                return payload["brief"]
            if PREFERRED_CAPTION == "analytical":
                return analytical_to_text(payload["analytical"])
            return payload["descriptive"]
        """
    ),
    markdown("## 6. Load Qwen3-VL on the A100"),
    code(
        r"""
        from transformers import AutoProcessor
        import transformers

        def resolve_model_class():
            for name in (
                "Qwen3VLForConditionalGeneration",
                "AutoModelForImageTextToText",
                "AutoModelForMultimodalLM",
                "AutoModelForVision2Seq",
            ):
                model_class = getattr(transformers, name, None)
                if model_class is not None:
                    print("Using model class:", name)
                    return model_class
            raise RuntimeError("Installed transformers does not expose a Qwen3-VL model class.")

        model_kwargs = {
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
        }
        if USE_4BIT:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        model = resolve_model_class().from_pretrained(MODEL_ID, **model_kwargs).eval()
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model_device = next(model.parameters()).device
        print("Model device:", model_device)
        print("GPU allocated GiB:", round(torch.cuda.memory_allocated() / 2**30, 2))

        def generate_response(panel, prompt):
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": panel},
                    {"type": "text", "text": prompt},
                ],
            }]
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model_device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    repetition_penalty=1.05,
                )
            trimmed = generated[:, inputs["input_ids"].shape[-1] :]
            return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
        """
    ),
    markdown("## 7. Transactional checkpoint database and atomic JSONL export"),
    code(
        r"""
        import os, sqlite3, time

        connection = sqlite3.connect(CAPTION_DB)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute('''
            CREATE TABLE IF NOT EXISTS captions (
                patch TEXT PRIMARY KEY,
                manifest_index INTEGER NOT NULL,
                tile_id TEXT,
                split_name TEXT,
                status TEXT NOT NULL,
                payload_json TEXT,
                evidence_json TEXT,
                raw_response TEXT,
                error TEXT,
                source_model TEXT,
                updated_at REAL NOT NULL
            )
        ''')
        connection.commit()

        def export_jsonl():
            temporary = CAPTION_JSONL.with_suffix(CAPTION_JSONL.suffix + ".tmp")
            evidence_temporary = EVIDENCE_JSONL.with_suffix(EVIDENCE_JSONL.suffix + ".tmp")
            rows = connection.execute(
                "SELECT patch, manifest_index, tile_id, split_name, payload_json, evidence_json "
                "FROM captions WHERE status='complete' ORDER BY manifest_index"
            ).fetchall()
            with temporary.open("w", encoding="utf-8") as captions_handle, evidence_temporary.open(
                "w", encoding="utf-8"
            ) as evidence_handle:
                for patch, index, tile, split_name, payload_text, evidence_text_value in rows:
                    payload = json.loads(payload_text)
                    value = {
                        "patch": patch,
                        "manifest_index": index,
                        "tile_id": tile,
                        "split": split_name,
                        "caption": top_level_caption(payload),
                        "captions": payload,
                        "preferred_caption": PREFERRED_CAPTION,
                        "source_model": MODEL_ID,
                        "grounding": "B02_B03_B04_B08_B11_SCL",
                        "spectral_evidence": json.loads(evidence_text_value),
                    }
                    captions_handle.write(json.dumps(value, ensure_ascii=True) + "\n")
                    evidence_handle.write(json.dumps({
                        "patch": patch,
                        "manifest_index": index,
                        "evidence": value["spectral_evidence"],
                    }, ensure_ascii=True) + "\n")
                captions_handle.flush()
                evidence_handle.flush()
                os.fsync(captions_handle.fileno())
                os.fsync(evidence_handle.fileno())
            os.replace(temporary, CAPTION_JSONL)
            os.replace(evidence_temporary, EVIDENCE_JSONL)
            return len(rows)

        force_patches = {
            record["patch"]
            for record in records
            if record["_manifest_index"] in set(FORCE_RECATION_INDICES)
        }
        if force_patches:
            connection.executemany("DELETE FROM captions WHERE patch=?", [(patch,) for patch in force_patches])
            connection.commit()
            print("Deleted for re-captioning:", len(force_patches))

        completed = {
            row[0]
            for row in connection.execute(
                "SELECT patch FROM captions WHERE status='complete'"
            ).fetchall()
        }
        print("Already completed:", len(completed))
        print("Pending in selected range:", sum(record["patch"] not in completed for record in records))
        print("Exported rows:", export_jsonl())
        """
    ),
    markdown("## 8. Generate grounded captions and resume automatically"),
    code(
        r"""
        pending = [record for record in records if record["patch"] not in completed]
        succeeded = 0
        failed = 0

        for position, record in enumerate(tqdm(pending, desc="grounded captions"), start=1):
            patch = record["patch"]
            try:
                evidence, arrays = extract_evidence(record)
                panel = build_panel(arrays)
                payload = None
                raw_response = ""
                issues = []

                for attempt in range(RETRY_GENERATION + 1):
                    correction = None
                    if issues:
                        correction = (
                            "The previous response violated these constraints: "
                            + "; ".join(issues)
                            + ". Regenerate the entire JSON without those unsupported claims."
                        )
                    raw_response = generate_response(panel, caption_prompt(evidence, correction))
                    parsed = parse_json_object(raw_response)
                    payload, issues = validate_payload(parsed, evidence)
                    if not issues:
                        break

                if issues:
                    payload = safe_fallback(payload, evidence)
                    payload, remaining = validate_payload(payload, evidence)
                    issues = remaining
                if issues:
                    raise ValueError("Caption still violates constraints: " + "; ".join(issues))

                connection.execute(
                    '''
                    INSERT OR REPLACE INTO captions
                    (patch, manifest_index, tile_id, split_name, status, payload_json,
                     evidence_json, raw_response, error, source_model, updated_at)
                    VALUES (?, ?, ?, ?, 'complete', ?, ?, ?, NULL, ?, ?)
                    ''',
                    (
                        patch,
                        record["_manifest_index"],
                        record.get("tile_id", ""),
                        record.get("split", ""),
                        json.dumps(payload, ensure_ascii=True),
                        json.dumps(evidence, ensure_ascii=True),
                        raw_response,
                        MODEL_ID,
                        time.time(),
                    ),
                )
                succeeded += 1
            except Exception as error:
                connection.execute(
                    '''
                    INSERT OR REPLACE INTO captions
                    (patch, manifest_index, tile_id, split_name, status, payload_json,
                     evidence_json, raw_response, error, source_model, updated_at)
                    VALUES (?, ?, ?, ?, 'failed', NULL, NULL, NULL, ?, ?, ?)
                    ''',
                    (
                        patch,
                        record["_manifest_index"],
                        record.get("tile_id", ""),
                        record.get("split", ""),
                        f"{type(error).__name__}: {error}",
                        MODEL_ID,
                        time.time(),
                    ),
                )
                failed += 1
                print(f"\nFAILED index={record['_manifest_index']} patch={patch}: {error}", flush=True)

            if position % CHECKPOINT_EVERY == 0:
                connection.commit()
            if position % EXPORT_EVERY == 0:
                connection.commit()
                exported = export_jsonl()
                print(
                    f"\ncheckpoint selected={position}/{len(pending)} "
                    f"new_ok={succeeded} new_failed={failed} exported={exported}",
                    flush=True,
                )

        connection.commit()
        exported = export_jsonl()
        print("New completed:", succeeded)
        print("New failed:", failed)
        print("Total exported:", exported)
        print("SQLite checkpoint:", CAPTION_DB)
        print("Training caption file:", CAPTION_JSONL)
        """
    ),
    markdown("## 9. Visualize the caption and spectral evidence for any manifest index"),
    code(
        r"""
        import textwrap
        import pandas as pd

        def caption_row_for_index(index):
            row = connection.execute(
                "SELECT patch, status, payload_json, evidence_json, error "
                "FROM captions WHERE manifest_index=?",
                (int(index),),
            ).fetchone()
            return row

        def record_for_index(index):
            for record in records:
                if record["_manifest_index"] == int(index):
                    return record
            raise IndexError(
                f"Index {index} is outside the selected range "
                f"{records[0]['_manifest_index']}..{records[-1]['_manifest_index']}"
            )

        def show_caption(index):
            record = record_for_index(index)
            evidence, arrays = extract_evidence(record)
            panel = build_panel(arrays)
            row = caption_row_for_index(index)

            display(panel)
            print(f"manifest_index: {index}")
            print(f"patch: {record['patch']}")
            print(f"tile: {record.get('tile_id')}  split: {record.get('split')}")
            print("\nSPECTRAL EVIDENCE")
            print(json.dumps(evidence, indent=2))
            if row is None:
                print("\nCAPTION: not generated")
                return
            _, status, payload_text, _, error = row
            print("\nSTATUS:", status)
            if error:
                print("ERROR:", error)
            if payload_text:
                payload = json.loads(payload_text)
                print("\nBRIEF")
                print(payload["brief"])
                print("\nDESCRIPTIVE")
                print(payload["descriptive"])
                print("\nANALYTICAL")
                print(json.dumps(payload["analytical"], indent=2))

        INDEX_TO_VISUALIZE = 0
        show_caption(INDEX_TO_VISUALIZE)
        """
    ),
    markdown("## 10. Audit all completed captions and list suspicious cases"),
    code(
        r"""
        audit_rows = []
        for patch, index, payload_text, evidence_text_value in connection.execute(
            "SELECT patch, manifest_index, payload_json, evidence_json "
            "FROM captions WHERE status='complete' ORDER BY manifest_index"
        ):
            payload = json.loads(payload_text)
            evidence = json.loads(evidence_text_value)
            _, issues = validate_payload(payload, evidence)
            audit_rows.append({
                "manifest_index": index,
                "patch": patch,
                "water_supported": evidence["water_supported"],
                "water_fraction": evidence["water_fraction"],
                "vegetation_fraction": evidence["vegetation_fraction"],
                "cloud_fraction": evidence["cloud_fraction"],
                "issues": "; ".join(issues),
                "caption": payload["descriptive"],
            })

        audit = pd.DataFrame(audit_rows)
        print("Completed captions:", len(audit))
        if len(audit):
            print("Water-supported patches:", int(audit["water_supported"].sum()))
            print("Captions with validation issues:", int(audit["issues"].ne("").sum()))
            display(audit.sort_values(
                ["issues", "water_fraction"], ascending=[False, False]
            ).head(30))

        failed_rows = pd.read_sql_query(
            "SELECT manifest_index, patch, error, updated_at "
            "FROM captions WHERE status='failed' ORDER BY manifest_index",
            connection,
        )
        print("Failed rows:", len(failed_rows))
        if len(failed_rows):
            display(failed_rows.head(50))

        # To repair selected captions:
        # 1. Put their indices in FORCE_RECATION_INDICES in cell 1.
        # 2. Rerun cells 7 and 8. Only those rows are regenerated and replaced.
        """
    ),
    markdown("## 11. Use the new captions in GeoDiff-GAN training"),
    code(
        r"""
        print("Use these settings in the training configuration:")
        print(f"runtime_config['data']['captions'] = {str(CAPTION_JSONL)!r}")
        print("runtime_config['data']['caption_field'] = 'caption'")
        print("runtime_config['data']['caption_sampling'] = 'random'")
        print("runtime_config['data']['random_caption_fields'] = ['brief', 'descriptive', 'analytical']")

        # Keep the old caption file until this audit is satisfactory.
        # CAPTION_JSONL is already backward-compatible with SentinelPatchDataset.
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "GeoDiff-GAN Python 3.11",
            "language": "python",
            "name": "geodiff-py311",
        },
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
print(OUTPUT)
