#!/usr/bin/env python3
"""Render reproducible ImageNet detail comparisons from verified lossless grids."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import flip_evaluator as flip
import matplotlib
matplotlib.use("Agg")
from matplotlib import colormaps
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


CHECKPOINT_SHA = "dd28db9306d526bdc9fbf8016403e106c4f317fd8f5c04792f0953e53310fdab"
GRID_SHA = "ce88a26b6dba52c0d07649aea91342830babedbf57175003cf534ea53c730928"
REFERENCE_SHA = "e5dd9c8167134559cf6d10dfc397493a67307ac20cb8fd1fcbcd7696b7a60d8f"
MODELS = [
    dict(key="vqgan16", method="VQGAN (public release)", shape="16×16×1", rfid=4.90, note="*"),
    dict(key="rq4", method="RQ-VAE", shape="8×8×4", rfid=4.73, note=""),
    dict(key="laser2", method="LASER", shape="8×8×2", rfid=7.79, note=""),
    dict(key="laser4", method="LASER", shape="8×8×4", rfid=4.21, note=""),
]
CLASSES = ["tench", "goldfish", "great white shark", "tiger shark", "hammerhead", "electric ray", "stingray", "cock"]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
COLORS = ["#007eaf", "#b35b00"]
LEFT, GAP, ZOOM, CROP = 24, 14, 192, 64
FULL_X = [LEFT, LEFT + 256 + GAP]
GROUP_X = [LEFT + 2 * (256 + GAP), LEFT + 2 * (256 + GAP) + 3 * (ZOOM + GAP) + 18]
WIDTH, HEIGHT = GROUP_X[1] + 3 * (ZOOM + GAP) + 16, 1530


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def label(draw, pos, value, size=17, color="#202020", bold=False):
    draw.text(pos, value, font=ImageFont.truetype(BOLD if bold else FONT, size), fill=color)


def choose_regions(original):
    """Select both crops using only source gradients, with deterministic ties."""
    gray = np.asarray(original, dtype=np.float32).mean(2) / 255.0

    def score(x, y):
        a = gray[y:y+CROP, x:x+CROP]
        return float(np.abs(np.diff(a, axis=0)).mean() + np.abs(np.diff(a, axis=1)).mean())

    candidates = [(score(x, y), -y, -x) for y in range(48, 145, 16) for x in range(48, 145, 16)]
    _, ny, nx = max(candidates)
    first = (-nx, -ny, -nx+CROP, -ny+CROP)
    candidates = []
    for y in range(16, 177, 16):
        for x in range(16, 177, 16):
            overlap = max(0, min(x+CROP, first[2])-max(x, first[0])) * max(0, min(y+CROP, first[3])-max(y, first[1]))
            if overlap <= 512:
                candidates.append((score(x, y), -y, -x))
    _, ny, nx = max(candidates)
    return [first, (-nx, -ny, -nx+CROP, -ny+CROP)]


def boxed(image, boxes):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    for k, (x0, y0, x1, y1) in enumerate(boxes):
        draw.rectangle((x0, y0, x1-1, y1-1), outline=COLORS[k], width=2)
        draw.rectangle((x0, y0-19, x0+18, y0), fill=COLORS[k])
        label(draw, (x0+2, y0-20), chr(65+k), 15, "white")
    return result


def zoom(image, box):
    return image.crop(tuple(box)).resize((ZOOM, ZOOM), Image.Resampling.NEAREST)


def color_map(values):
    # No per-image normalization: the magma input is the actual [0, 1] error.
    return Image.fromarray(np.rint(colormaps["magma"](values)[..., :3] * 255).astype(np.uint8))


def render_page(entries, title, subtitle, output):
    canvas = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(canvas)
    label(draw, (LEFT, 16), title, 27, bold=True)
    label(draw, (LEFT, 57), subtitle, 17)
    for row, entry in enumerate(entries):
        y = 108 + row * 322
        reference, recon, error, boxes = (entry[k] for k in ("reference", "recon", "error", "boxes"))
        label(draw, (LEFT, y), entry["title"], 18, bold=True)
        for x, name, image in zip(FULL_X, ["Reference", "Reconstruction"], [reference, recon]):
            label(draw, (x, y+28), name, 17)
            canvas.paste(boxed(image, boxes), (x, y+54))
        for k, group_x in enumerate(GROUP_X):
            label(draw, (group_x, y), f"{chr(65+k)} — {'central detail' if k == 0 else 'source texture'}", 18, COLORS[k])
            for j, (name, image) in enumerate(zip(["Reference 3×", "Reconstruction 3×", "FLIP error 3×"], [reference, recon, error])):
                x = group_x + j * (ZOOM + GAP)
                label(draw, (x, y+31), name, 15)
                canvas.paste(zoom(image, boxes[k]), (x, y+62))
                draw.rectangle((x-1, y+61, x+ZOOM, y+62+ZOOM), outline=COLORS[k], width=2)
            label(draw, (group_x, y+269), f"Crop (x={boxes[k][0]}, y={boxes[k][1]}, 64×64)", 14)
    label(draw, (LEFT, 1410), "64×64 crops enlarged 3× by nearest neighbor. Source-selected regions are identical for every model; no enhancement.", 17)
    label(draw, (LEFT, 1440), "rFID values are supplied table results, not estimates from these eight examples. * VQGAN release README reports 4.98 (table: 4.90).", 16)
    label(draw, (LEFT, 1470), "NVIDIA LDR-FLIP, 67.02 pixels/degree, fixed 0–1 scale:", 16)
    bar_x, bar_y, bar_w = 530, 1471, 230
    gradient = np.tile(np.linspace(0, 1, bar_w, dtype=np.float32), (14, 1))
    canvas.paste(color_map(gradient), (bar_x, bar_y))
    label(draw, (bar_x-17, bar_y-2), "0", 14)
    label(draw, (bar_x+bar_w+6, bar_y-2), "1", 14)
    label(draw, (805, 1470), "Dark: lower perceptual error. Bright: higher error. Computed at native 256×256 before cropping.", 15)
    canvas.save(output)
    return canvas


def write_pdf(images, path):
    # Matplotlib embeds these raster panels losslessly; Pillow PDF uses JPEG.
    with PdfPages(path) as pdf:
        for image in images:
            fig = plt.figure(figsize=(image.width/160, image.height/160))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(np.asarray(image), interpolation="none")
            ax.axis("off")
            pdf.savefig(fig, dpi=160)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.source / "figures"
    assert digest(source / "reference-images.png") == REFERENCE_SHA
    assert digest(source / "reconstruction-grid.png") == GRID_SHA
    manifest = json.loads((source / "manifest.json").read_text())
    assert manifest["metrics"]["laser4"]["checkpoint_sha256"] == CHECKPOINT_SHA
    reference_strip = Image.open(source / "reference-images.png").convert("RGB")
    reconstruction_grid = Image.open(source / "reconstruction-grid.png").convert("RGB")
    assert reference_strip.size == (2048, 256) and reconstruction_grid.size == (2048, 1024)
    references = [reference_strip.crop((i*256, 0, (i+1)*256, 256)) for i in range(8)]
    regions = [choose_regions(image) for image in references]
    selection = {
        "selection": "Same eight unfiltered examples as the prior comparison: first lexicographic validation image from each of the first eight synsets. A: maximal source-gradient 64×64 crop on a central 16-pixel grid. B: maximal source-gradient crop on a wider grid with overlap <=512 pixels with A. Ties favor top then left. No reconstruction is used for selection.",
        "crop_size": CROP, "zoom_factor": 3, "resampling": "nearest", "images": [
            dict(index=i, image_id=Path(name).stem, synset=Path(name).parent.name, class_name=CLASSES[i], boxes=regions[i])
            for i, name in enumerate(manifest["image_paths"])
        ],
    }
    (args.output / "fixed-regions.json").write_text(json.dumps(selection, indent=2) + "\n")
    raw = args.output / "pixels"
    raw.mkdir(exist_ok=True)
    entries, metrics = {}, []
    for i, ref in enumerate(references):
        ref.save(raw / f"example-{i+1:02d}-reference.png")
        for m, model in enumerate(MODELS):
            rec = reconstruction_grid.crop((i*256, m*256, (i+1)*256, (m+1)*256))
            error, mean_error, parameters = flip.evaluate(
                np.asarray(ref, dtype=np.float32)/255,
                np.asarray(rec, dtype=np.float32)/255, "LDR", applyMagma=False,
            )
            error = np.asarray(error).squeeze(-1)
            assert np.isfinite(error).all() and error.min() >= 0 and error.max() <= 1
            heatmap = color_map(error)
            stem = f"example-{i+1:02d}-{model['key']}"
            rec.save(raw / f"{stem}.png")
            heatmap.save(raw / f"{stem}-flip.png")
            np.save(raw / f"{stem}-flip.npy", error)
            for k, box in enumerate(regions[i]):
                for name, image in [("reference", ref), ("reconstruction", rec), ("flip", heatmap)]:
                    crop = image.crop(tuple(box))
                    crop.save(raw / f"{stem}-{chr(65+k)}-{name}-64.png")
                    expanded = zoom(image, box)
                    # Every enlarged pixel must be a direct repeat of its source.
                    assert np.array_equal(np.asarray(expanded), np.repeat(np.repeat(np.asarray(crop), 3, 0), 3, 1))
            entry = dict(reference=ref, recon=rec, error=heatmap, boxes=regions[i],
                         title=f"{model['method']}  {model['shape']}  |  rFID {model['rfid']:.2f}{model['note']}")
            entries[i, m] = entry
            metrics.append(dict(example=i+1, image_id=selection["images"][i]["image_id"], model=model["key"],
                                flip_mean=float(mean_error), ppd=float(parameters["ppd"]),
                                crop_a_flip=float(error[regions[i][0][1]:regions[i][0][3], regions[i][0][0]:regions[i][0][2]].mean()),
                                crop_b_flip=float(error[regions[i][1][1]:regions[i][1][3], regions[i][1][0]:regions[i][1][2]].mean())))
        print(f"Computed matched FLIP maps for example {i+1}/8", flush=True)
    pages = []
    for i in range(8):
        image_id = selection["images"][i]["image_id"]
        pages.append(render_page(
            [entries[i, m] for m in range(4)],
            f"ImageNet 256×256 Stage 1 — {CLASSES[i]} — example {i+1}/8",
            f"{image_id}  |  Four available models, same input and crops  |  All dictionaries / codebooks: 16,384",
            args.output / f"comparison-example-{i+1:02d}.png",
        ))
    write_pdf(pages, args.output / "imagenet-stage1-model-comparison.pdf")
    laser_pages = []
    for page in range(2):
        rows = []
        for i in range(page*4, page*4+4):
            row = dict(entries[i, 3])
            row["title"] = f"Example {i+1} — {CLASSES[i]}  |  {selection['images'][i]['image_id'][-8:]}"
            rows.append(row)
        laser_pages.append(render_page(
            rows, f"ImageNet LASER rFID 4.21 — reconstruction details — examples {page*4+1}–{page*4+4}",
            "Frozen 8×8×4 tokenizer  |  Dictionary: 16,384  |  Same lossless pixels as the archived stage-1 comparison",
            args.output / f"laser-rfid421-zoom-page-{page+1}.png",
        ))
    write_pdf(laser_pages, args.output / "laser-rfid421-reconstruction-zooms.pdf")
    overview = Image.new("RGB", (2370, 1450), "white")
    draw = ImageDraw.Draw(overview)
    label(draw, (24, 18), "ImageNet 256×256 Stage 1 — matched reconstruction overview", 29, bold=True)
    for i in range(8):
        label(draw, (294+i*258, 66), CLASSES[i], 14)
    for row in range(5):
        y = 94+row*260
        label(draw, (24, y+84), "Reference" if row == 0 else MODELS[row-1]["method"], 18, bold=True)
        if row:
            model = MODELS[row-1]
            label(draw, (24, y+118), f"{model['shape']}   K=16,384", 17)
            label(draw, (24, y+148), f"rFID {model['rfid']:.2f}{model['note']}", 18)
        for i in range(8):
            overview.paste(boxed(references[i] if row == 0 else entries[i, row-1]["recon"], regions[i]), (294+i*258, y))
    label(draw, (24, 1404), "rFID: supplied full-dataset results. * VQGAN public-release README: 4.98; supplied table: 4.90. Five other table rows lack exact checkpoints.", 17)
    overview.save(args.output / "matched-reconstruction-overview.png")
    write_pdf([overview], args.output / "matched-reconstruction-overview.pdf")
    with (args.output / "flip-metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    result = dict(
        source_artifact="helloimlixin-rutgers/laser/imagenet-stage1-table-reconstruction-figures:v2",
        source_run="helloimlixin-rutgers/laser/a4mxuypy", reference_gallery="helloimlixin-rutgers/laser/ffhq-k4-continue150-zooms-20260923",
        checkpoint_sha256=CHECKPOINT_SHA, source_grid_sha256=GRID_SHA, source_reference_sha256=REFERENCE_SHA,
        models=MODELS, checkpoint_provenance=manifest["metrics"], unavailable_exact_rows=manifest["unavailable_exact_rows"],
        vqgan_rfid_discrepancy=manifest["provenance"]["vqgan16"], examples=8, regions_per_image=2,
        source_pixel_representation="archived lossless PNG of clipped, rounded uint8 sRGB reconstructions; no fresh inference",
        image_selection=selection["selection"], flip=dict(implementation="NVIDIA flip-evaluator 1.7, CPU", dynamic_range="LDR", ppd=metrics[0]["ppd"], scale=[0,1], color_map="matplotlib magma", computed_before_crop=True),
        pixel_integrity="All 192×192 zooms equal 3× nearest-neighbor repeats of the saved 64×64 source crops; overlays appear only on full images.",
        training_unchanged=True, cpu_only=True, new_dataset_rfid_measured=False,
    )
    (args.output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(output=str(args.output), comparison_pages=8, laser_pages=2)), flush=True)


if __name__ == "__main__":
    main()
