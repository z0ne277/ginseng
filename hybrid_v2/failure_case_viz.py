"""
Failure case visualization: parse batch retrieval results and generate topology op grids
for query / relevant / top-1 retrieved.
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

try:
    import cv2  # type: ignore
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


def _load_font(size: int = 16) -> ImageFont.FreeTypeFont:
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",  # Microsoft YaHei
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]
    for p in candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def pil_load_l(path: str, size: int = 224) -> Image.Image:
    img = Image.open(path).convert("L")
    if size is not None:
        img = img.resize((size, size), resample=Image.BILINEAR)
    return img


def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn = float(x.min())
    mx = float(x.max())
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def _preprocess_tensor(image_path: str, size: int = 224) -> torch.Tensor:
    img = Image.open(image_path).convert("L").convert("RGB")
    if size is not None:
        img = img.resize((size, size), resample=Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    tensor = (tensor - 0.5) / 0.5
    return tensor


def _load_model(checkpoint_dir: Optional[str] = None, model_path: Optional[str] = None):
    from model import MoCoV3HybridTopo

    cfg = None
    if checkpoint_dir:
        ckpt = Path(checkpoint_dir)
        cfg_path = ckpt / "config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if model_path is None:
            candidate = ckpt / "best_model.pth"
            if not candidate.exists():
                candidate = ckpt / "last_model.pth"
            model_path = str(candidate)

    if model_path is None:
        return None, None, None

    if cfg is None:
        cfg = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MoCoV3HybridTopo(
        feature_dim=int(cfg.get("feature_dim", 256)),
        topo_dim=int(cfg.get("topo_dim", 128)),
        K=int(cfg.get("K", 4096)),
        m=float(cfg.get("m", 0.999)),
        T=float(cfg.get("T", 0.07)),
        topo_weight=float(cfg.get("topo_weight", 0.35)),
        use_legacy_branch=bool(cfg.get("use_legacy_branch", True)),
        use_skeleton_branch=bool(cfg.get("use_skeleton_branch", False)),
        use_edge_branch=bool(cfg.get("use_edge_branch", False)),
        use_frequency_branch=bool(cfg.get("use_frequency_branch", False)),
        device=device,
    )
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, device, cfg


def _compute_branch_weights(
    image_path: str,
    model,
    device,
    size: int,
) -> Optional[Dict]:
    img_t = _preprocess_tensor(image_path, size=size).to(device)
    with torch.no_grad():
        feat = model.encoder_q(img_t)
        _, branch_outputs = model.topo_extractor_q(feat)

    extractor = model.topo_extractor_q
    branch_names = list(getattr(extractor, "branch_names", []))
    if not branch_names:
        return None

    branch_features = [branch_outputs[n] for n in branch_names]
    raw_concat = torch.cat(branch_features, dim=1)
    raw_concat = extractor.branch_dropout(raw_concat)
    context = extractor.context_projector(raw_concat)
    if extractor.use_adaptive_weights:
        weights = extractor.adaptive_weight_net(context)
    else:
        weights = torch.softmax(extractor.static_branch_weights, dim=0).unsqueeze(0)

    weights_raw = weights[0].detach().cpu().numpy()

    if len(branch_names) == 1:
        weights_used = np.array([1.0], dtype=np.float32)
    else:
        weights_used = weights_raw.copy()
        if getattr(extractor, "preserve_legacy_residual", False) and ("legacy" in branch_names):
            legacy_idx = branch_names.index("legacy")
            weights_used[legacy_idx] = 0.0
            s = float(weights_used.sum())
            if s > 1e-6:
                weights_used = weights_used / s
        if float(weights_used.sum()) < 1e-6:
            weights_used = np.full_like(weights_used, 1.0 / len(weights_used))

    return {
        "branches": branch_names,
        "weights": weights_used.tolist(),
        "raw_weights": weights_raw.tolist(),
    }


def fft_low_high(gray01: np.ndarray, keep_ratio: float = 0.10) -> Tuple[np.ndarray, np.ndarray]:
    h, w = gray01.shape
    f = np.fft.fft2(gray01)
    fshift = np.fft.fftshift(f)

    cy, cx = h // 2, w // 2
    ry, rx = int(h * keep_ratio), int(w * keep_ratio)

    mask_low = np.zeros((h, w), dtype=np.float32)
    mask_low[cy - ry : cy + ry + 1, cx - rx : cx + rx + 1] = 1.0

    low = fshift * mask_low
    high = fshift * (1.0 - mask_low)

    low_ishift = np.fft.ifftshift(low)
    high_ishift = np.fft.ifftshift(high)

    low_img = np.abs(np.fft.ifft2(low_ishift))
    high_img = np.abs(np.fft.ifft2(high_ishift))

    return normalize01(low_img), normalize01(high_img)


def _pad(img: np.ndarray, pad: int = 1) -> np.ndarray:
    return np.pad(img, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)


def binary_erosion(img: np.ndarray, iterations: int = 1) -> np.ndarray:
    if HAS_CV2:
        kernel = np.ones((3, 3), dtype=np.uint8)
        out = cv2.erode((img * 255).astype(np.uint8), kernel, iterations=iterations)
        return (out > 0).astype(np.uint8)
    out = img.copy()
    kernel = np.ones((3, 3), dtype=np.uint8)
    for _ in range(iterations):
        padded = _pad(out, 1)
        h, w = out.shape
        new = np.zeros_like(out)
        for i in range(h):
            for j in range(w):
                region = padded[i : i + 3, j : j + 3]
                new[i, j] = 1 if np.all(region * kernel) else 0
        out = new
    return out


def binary_dilation(img: np.ndarray, iterations: int = 1) -> np.ndarray:
    if HAS_CV2:
        kernel = np.ones((3, 3), dtype=np.uint8)
        out = cv2.dilate((img * 255).astype(np.uint8), kernel, iterations=iterations)
        return (out > 0).astype(np.uint8)
    out = img.copy()
    kernel = np.ones((3, 3), dtype=np.uint8)
    for _ in range(iterations):
        padded = _pad(out, 1)
        h, w = out.shape
        new = np.zeros_like(out)
        for i in range(h):
            for j in range(w):
                region = padded[i : i + 3, j : j + 3]
                new[i, j] = 1 if np.any(region * kernel) else 0
        out = new
    return out


def zhang_suen_thinning(img: np.ndarray, max_iter: int = 50) -> np.ndarray:
    if HAS_CV2:
        skel = np.zeros_like(img, dtype=np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        img_u8 = (img * 255).astype(np.uint8)
        done = False
        while not done:
            eroded = cv2.erode(img_u8, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img_u8, temp)
            skel = cv2.bitwise_or(skel, temp)
            img_u8 = eroded.copy()
            done = cv2.countNonZero(img_u8) == 0
        return (skel > 0).astype(np.uint8)
    out = img.copy().astype(np.uint8)
    h, w = out.shape
    if h < 3 or w < 3:
        return out

    def neighbors(x, y):
        return [
            out[x - 1, y],
            out[x - 1, y + 1],
            out[x, y + 1],
            out[x + 1, y + 1],
            out[x + 1, y],
            out[x + 1, y - 1],
            out[x, y - 1],
            out[x - 1, y - 1],
        ]

    def transitions(n):
        return sum((n[i] == 0 and n[(i + 1) % 8] == 1) for i in range(8))

    for _ in range(max_iter):
        changed = False
        to_remove = []
        for x in range(1, h - 1):
            for y in range(1, w - 1):
                if out[x, y] != 1:
                    continue
                n = neighbors(x, y)
                c = sum(n)
                if c < 2 or c > 6:
                    continue
                t = transitions(n)
                if t != 1:
                    continue
                if n[0] * n[2] * n[4] != 0:
                    continue
                if n[2] * n[4] * n[6] != 0:
                    continue
                to_remove.append((x, y))
        if to_remove:
            for x, y in to_remove:
                out[x, y] = 0
            changed = True

        to_remove = []
        for x in range(1, h - 1):
            for y in range(1, w - 1):
                if out[x, y] != 1:
                    continue
                n = neighbors(x, y)
                c = sum(n)
                if c < 2 or c > 6:
                    continue
                t = transitions(n)
                if t != 1:
                    continue
                if n[0] * n[2] * n[6] != 0:
                    continue
                if n[0] * n[4] * n[6] != 0:
                    continue
                to_remove.append((x, y))
        if to_remove:
            for x, y in to_remove:
                out[x, y] = 0
            changed = True

        if not changed:
            break

    return out


def sobel_edge(img01: np.ndarray) -> np.ndarray:
    if HAS_CV2:
        sobelx = cv2.Sobel(img01.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img01.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
        return normalize01(sobel)
    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    padded = np.pad(img01, ((1, 1), (1, 1)), mode="edge")
    h, w = img01.shape
    out = np.zeros_like(img01, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            region = padded[i : i + 3, j : j + 3]
            sx = np.sum(region * gx)
            sy = np.sum(region * gy)
            out[i, j] = np.sqrt(sx * sx + sy * sy)
    return normalize01(out)


def save_topology_ops(image_path: str, out_path: Path, size: int = 128) -> None:
    font = _load_font(16)
    binary_l = pil_load_l(image_path, size=size)
    gray01 = np.array(binary_l, dtype=np.float32) / 255.0
    mask = (gray01 >= 0.5).astype(np.uint8)

    low, high = fft_low_high(gray01, keep_ratio=0.10)

    erosion1 = binary_erosion(mask, 1)
    erosion2 = binary_erosion(mask, 2)
    erosion3 = binary_erosion(mask, 3)

    dilation1 = binary_dilation(mask, 1)
    dilation2 = binary_dilation(mask, 2)
    dilation3 = binary_dilation(mask, 3)

    skeleton = zhang_suen_thinning(mask, max_iter=30)
    skeleton2 = zhang_suen_thinning(binary_erosion(mask, 1), max_iter=30)
    skeleton3 = zhang_suen_thinning(binary_erosion(mask, 2), max_iter=30)

    morph_grad = np.clip(dilation1 - erosion1, 0, 1)
    sobel = sobel_edge(gray01)

    items = [
        ("二值图", gray01),
        ("低频(FFT)", low),
        ("高频(FFT)", high),
        ("腐蚀x1", erosion1),
        ("腐蚀x2", erosion2),
        ("腐蚀x3", erosion3),
        ("膨胀x1", dilation1),
        ("膨胀x2", dilation2),
        ("膨胀x3", dilation3),
        ("形态梯度", morph_grad),
        ("骨架迭代1", skeleton),
        ("骨架迭代2", skeleton2),
        ("骨架迭代3", skeleton3),
        ("Sobel边缘", sobel),
    ]

    cols = 4
    rows = int(np.ceil(len(items) / cols))
    cell = size
    pad = 6
    title_h = 24
    w = cols * cell + (cols + 1) * pad
    h = rows * (cell + title_h) + (rows + 1) * pad + 30
    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    for idx, (title, img) in enumerate(items):
        r, c = divmod(idx, cols)
        x = pad + c * (cell + pad)
        y = pad + r * (cell + title_h + pad)
        draw.text((x, y), title, font=font, fill=(255, 255, 255))
        arr = (img * 255).astype(np.uint8)
        im = Image.fromarray(arr, mode="L").convert("RGB")
        canvas.paste(im, (x, y + title_h))

    legend = "图例：二值图/腐蚀x/膨胀x/骨架迭代x/形态梯度/Sobel边缘/FFT低频/FFT高频"
    draw.text((pad, h - 24), legend, font=font, fill=(200, 200, 200))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def parse_results_file(path: Path) -> Dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    q_match = re.search(r"Query Image:\s*(.+)", text)
    query_image = q_match.group(1).strip() if q_match else ""

    pattern = "[0-9]+\\.[0-9]+"

    def _find_float(label: str) -> Optional[float]:
        for line in text.splitlines():
            if label.upper() in line.upper():
                m = re.search(pattern, line)
                if m:
                    return float(m.group(0))
        return None

    metrics = {
        "mrr": _find_float("MRR"),
        "map": _find_float("MAP"),
        "recall@1": _find_float("RECALL@1"),
        "recall@5": _find_float("RECALL@5"),
        "recall@10": _find_float("RECALL@10"),
    }

    rel = []
    if "RELEVANT DOCUMENTS" in text:
        tail = text.split("RELEVANT DOCUMENTS", 1)[1]
        for line in tail.splitlines():
            line = line.strip()
            if not line:
                continue
            m = re.match(r"\d+\.\s+(.+)", line)
            if m:
                rel.append(m.group(1).strip())

    retrieved = []
    for line in text.splitlines():
        line = line.strip()
        if re.match(r"^\d+\s+\d+\.\d+", line):
            parts = line.split()
            rank = int(parts[0])
            score = float(parts[1])
            is_rel = "RELEVANT" in line and "NOT" not in line
            cols = re.split(r"\s{2,}", line)
            path_col = cols[-1].strip() if cols else ""
            retrieved.append({"rank": rank, "score": score, "relevant": is_rel, "path": path_col})

    found = [r for r in retrieved if r["relevant"]]
    return {
        "query_image": query_image,
        "relevant": rel,
        "retrieved": retrieved,
        "found": found,
        "metrics": metrics,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract failure cases and visualize topology ops")
    ap.add_argument("--results_root", type=str, required=True, help="batch_results_* directory")
    ap.add_argument("--output_dir", type=str, required=True, help="output directory")
    ap.add_argument("--max_cases", type=int, default=20)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--metric", type=str, default="mrr", choices=["mrr", "map", "recall@1", "recall@5", "recall@10"])
    ap.add_argument("--checkpoint_dir", type=str, default=None)
    ap.add_argument("--model_path", type=str, default=None)
    args = ap.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = None
    device = None
    if args.checkpoint_dir or args.model_path:
        model, device, _ = _load_model(args.checkpoint_dir, args.model_path)

    result_files = sorted(results_root.glob("**/*_results.txt"))
    items = []
    for rf in result_files:
        data = parse_results_file(rf)
        if not data["query_image"]:
            continue
        metric_val = data["metrics"].get(args.metric)
        if metric_val is None:
            continue
        total_rel = len(data["relevant"])
        found_rel = len(data["found"])
        if total_rel == 0:
            continue
        items.append({
            "file": str(rf),
            "query_image": data["query_image"],
            "relevant": data["relevant"],
            "retrieved": data["retrieved"],
            "found_rel": found_rel,
            "total_rel": total_rel,
            "metric": metric_val,
        })

    items.sort(key=lambda x: (x["metric"], x["found_rel"], x["total_rel"]))
    items = items[: int(args.max_cases)]

    summary_csv = out_dir / "failure_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_image", "metric", "total_rel", "found_rel", "top1_path", "top1_rel", "results_file"])
        for item in items:
            retrieved = item["retrieved"]
            top1 = retrieved[0]["path"] if retrieved else ""
            top1_rel = retrieved[0]["relevant"] if retrieved else False
            w.writerow([item["query_image"], item["metric"], item["total_rel"], item["found_rel"], top1, top1_rel, item["file"]])

    for idx, item in enumerate(items, 1):
        case_dir = out_dir / f"case_{idx:03d}"
        case_dir.mkdir(parents=True, exist_ok=True)

        query = item["query_image"]
        rel_list = item["relevant"]
        retrieved = item["retrieved"]
        top1 = retrieved[0]["path"] if retrieved else ""

        with (case_dir / "case.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "query": query,
                    "relevant": rel_list,
                    "top1": top1,
                    "metric": item["metric"],
                    "found_rel": item["found_rel"],
                    "total_rel": item["total_rel"],
                    "results_file": item["file"],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        if query and Path(query).exists():
            save_topology_ops(query, case_dir / "query_topology_ops.png", size=args.size)
            if model is not None:
                bw = _compute_branch_weights(query, model, device, size=args.size)
                if bw:
                    with (case_dir / "query_branch_weights.json").open("w", encoding="utf-8") as f:
                        json.dump(bw, f, ensure_ascii=False, indent=2)

        if rel_list:
            rel0 = rel_list[0]
            if Path(rel0).exists():
                save_topology_ops(rel0, case_dir / "relevant_topology_ops.png", size=args.size)
                if model is not None:
                    bw = _compute_branch_weights(rel0, model, device, size=args.size)
                    if bw:
                        with (case_dir / "relevant_branch_weights.json").open("w", encoding="utf-8") as f:
                            json.dump(bw, f, ensure_ascii=False, indent=2)

        if top1 and Path(top1).exists():
            save_topology_ops(top1, case_dir / "top1_topology_ops.png", size=args.size)
            if model is not None:
                bw = _compute_branch_weights(top1, model, device, size=args.size)
                if bw:
                    with (case_dir / "top1_branch_weights.json").open("w", encoding="utf-8") as f:
                        json.dump(bw, f, ensure_ascii=False, indent=2)

    print(f"Cases: {len(items)}")
    print(f"Summary: {summary_csv}")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
