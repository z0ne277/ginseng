\
\
\
\
\
\
\


from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
from matplotlib import font_manager
import matplotlib.pyplot as plt

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

from torchvision import transforms

from model import MoCoV3HybridTopo


DEFAULT_MEAN = [0.5, 0.5, 0.5]
DEFAULT_STD = [0.5, 0.5, 0.5]


def _setup_chinese_fonts() -> None:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name]
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


_setup_chinese_fonts()


@dataclass
class Sample:
    dataset: str
    binary_path: str


def _read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _parse_named_paths(items: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for s in items:
        if "=" not in s:
            raise ValueError(f"Expected NAME=PATH, got: {s}")
        name, path = s.split("=", 1)
        out[name.strip()] = path.strip()
    return out


def _read_csv_rows(csv_path: str) -> List[Dict]:
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        return df.to_dict(orient="records")
    except Exception:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)


def load_samples_from_csv(dataset_name: str, csv_path: str) -> List[Sample]:
    rows = _read_csv_rows(csv_path)
    samples: List[Sample] = []
    for r in rows:
        if "image" not in r:
            raise ValueError(f"CSV missing 'image' column: {csv_path}")
        samples.append(Sample(dataset=dataset_name, binary_path=str(r["image"])))
    return samples


def pil_load_l(path: str, size: int = 224) -> Image.Image:
    img = Image.open(path).convert("L")
    if size is not None:
        img = img.resize((size, size), resample=Image.BILINEAR)
    return img


def to_mask_from_binary_l(binary_l: Image.Image, thresh: int = 128) -> np.ndarray:
    arr = np.array(binary_l, dtype=np.uint8)
    mask = (arr >= thresh).astype(np.uint8) * 255
    return mask


def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn = float(x.min())
    mx = float(x.max())
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


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


def morph_ops(mask_u8: np.ndarray) -> Dict[str, np.ndarray]:
    if not HAS_CV2:
        raise RuntimeError("OpenCV (cv2) is required for morphology visualization.")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    erosion = {f"erosion_{i}": cv2.erode(mask_u8, kernel, iterations=i) for i in (1, 2, 3)}
    dilation = {f"dilation_{i}": cv2.dilate(mask_u8, kernel, iterations=i) for i in (1, 2, 3)}

    grad = cv2.morphologyEx(mask_u8, cv2.MORPH_GRADIENT, kernel)

    skel = np.zeros_like(mask_u8)
    img = mask_u8.copy()
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skeleton_snaps: Dict[str, np.ndarray] = {}
    for it in range(1, 4):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        skeleton_snaps[f"skeleton_{it}"] = skel.copy()

    sobelx = cv2.Sobel(mask_u8.astype(np.float32) / 255.0, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(mask_u8.astype(np.float32) / 255.0, cv2.CV_32F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
    sobel = normalize01(sobel)

    out: Dict[str, np.ndarray] = {}
    out.update({k: v.astype(np.float32) / 255.0 for k, v in erosion.items()})
    out.update({k: v.astype(np.float32) / 255.0 for k, v in dilation.items()})
    out["morph_grad"] = grad.astype(np.float32) / 255.0
    out.update({k: v.astype(np.float32) / 255.0 for k, v in skeleton_snaps.items()})
    out["sobel_edge"] = sobel
    return out


def denorm_to_rgb01(img_t: torch.Tensor) -> np.ndarray:
    x = img_t.detach().float().cpu()
    if x.ndim == 4:
        x = x[0]
    x = x.permute(1, 2, 0).numpy()
    x = x * np.array(DEFAULT_STD, dtype=np.float32) + np.array(DEFAULT_MEAN, dtype=np.float32)
    return np.clip(x, 0.0, 1.0)


def overlay_heatmap(base_rgb01: np.ndarray, heat01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat01 = np.clip(heat01.astype(np.float32), 0.0, 1.0)
    cmap = plt.get_cmap("jet")
    heat_rgb = cmap(heat01)[..., :3].astype(np.float32)
    out = (1.0 - alpha) * base_rgb01.astype(np.float32) + alpha * heat_rgb
    return np.clip(out, 0.0, 1.0)


def _feat_to_map(feat: torch.Tensor, out_hw: Tuple[int, int] = (224, 224)) -> np.ndarray:
    if feat.ndim == 4:
        feat = feat.mean(dim=1, keepdim=False)
    elif feat.ndim == 3:
        feat = feat.mean(dim=0, keepdim=False)
    if feat.ndim == 3 and feat.size(0) == 1:
        feat = feat[0]
    if out_hw is not None and feat.ndim == 2:
        feat = feat.unsqueeze(0).unsqueeze(0)
        feat = F.interpolate(feat, size=out_hw, mode="bilinear", align_corners=False)
        feat = feat[0, 0]
    arr = feat.detach().float().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return normalize01(arr)


class GradCAMBackbone:
    def __init__(self, model: MoCoV3HybridTopo, target_module: torch.nn.Module):
        self.model = model
        self.target_module = target_module
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self._h_fwd = self.target_module.register_forward_hook(self._save_activation)
        try:
            self._h_bwd = self.target_module.register_full_backward_hook(self._save_gradient)
        except Exception:
            self._h_bwd = self.target_module.register_backward_hook(self._save_gradient)

    def close(self) -> None:
        self._h_fwd.remove()
        self._h_bwd.remove()

    def _save_activation(self, _module, _inp, out):
        self.activations = out

    def _save_gradient(self, _module, _grad_inp, grad_out):
        if isinstance(grad_out, (tuple, list)):
            self.gradients = grad_out[0]
        else:
            self.gradients = grad_out

    def compute_from_loss(self, loss: torch.Tensor, out_hw: Tuple[int, int]) -> Optional[np.ndarray]:
        self.model.zero_grad(set_to_none=True)
        loss.backward()

        if self.activations is None or self.gradients is None:
            return None

        acts = self.activations
        grads = self.gradients
        if acts.ndim != 4 or grads.ndim != 4:
            return None

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=out_hw, mode="bilinear", align_corners=False)

        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max().clamp_min(1e-8))
        return cam.detach().cpu().numpy()


class GinsengFeatureVisualizer:
    def __init__(self, model: MoCoV3HybridTopo, device: torch.device, save_dir: Path):
        self.model = model.to(device)
        self.device = device
        self.save_dir = save_dir
        _safe_mkdir(self.save_dir)

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD),
        ])

    @staticmethod
    def load_from_checkpoint_dir(checkpoint_dir: str) -> Tuple[Dict, str]:
        ckpt = Path(checkpoint_dir)
        cfg_path = ckpt / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing config.json in {checkpoint_dir}")
        cfg = _read_json(cfg_path)
        model_path = str(ckpt / "best_model.pth")
        if not Path(model_path).exists():
            model_path = str(ckpt / "last_model.pth")
        return cfg, model_path

    @staticmethod
    def build_model(cfg: Dict, device: torch.device) -> MoCoV3HybridTopo:
        return MoCoV3HybridTopo(
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

    def load_weights(self, model_path: str) -> None:
        state = torch.load(model_path, map_location=self.device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def _tensorize_inputs(self, binary_l: Image.Image) -> torch.Tensor:
        binary_rgb = binary_l.convert("RGB")
        img_t = self.preprocess(binary_rgb).unsqueeze(0)
        return img_t.to(self.device)

    def _get_gradcam_target(self) -> torch.nn.Module:
        enc = self.model.encoder_q
        if hasattr(enc, "layer4"):
            return enc.layer4
        if isinstance(enc, torch.nn.Sequential):
            for m in reversed(list(enc.modules())):
                if isinstance(m, torch.nn.Conv2d):
                    return m
            for m in reversed(list(enc.children())):
                return m
        return enc

    def _extract_backbone_feature(self, img_t: torch.Tensor, layer: str = "layer4") -> torch.Tensor:
        enc = self.model.encoder_q
        if hasattr(enc, layer):
            x = img_t
            x = enc.conv1(x)
            x = enc.bn1(x)
            x = enc.relu(x)
            x = enc.maxpool(x)
            x = enc.layer1(x)
            if layer == "layer1":
                return x
            x = enc.layer2(x)
            if layer == "layer2":
                return x
            x = enc.layer3(x)
            if layer == "layer3":
                return x
            x = enc.layer4(x)
            return x

        if isinstance(enc, torch.nn.Sequential):
            idx_to_name = {
                4: "layer1",
                5: "layer2",
                6: "layer3",
                7: "layer4",
            }
            x = img_t
            for idx, m in enumerate(enc.children()):
                x = m(x)
                if idx_to_name.get(idx) == layer:
                    return x
            return x

        return enc(img_t)

    def visualize_topology_ops(self, binary_path: str, out_dir: Path, *, thresh: int = 128) -> None:
        _safe_mkdir(out_dir)
        binary_l = pil_load_l(binary_path, size=224)
        gray01 = np.array(binary_l, dtype=np.float32) / 255.0
        mask_u8 = (np.array(binary_l, dtype=np.uint8) >= thresh).astype(np.uint8) * 255

        low, high = fft_low_high(gray01, keep_ratio=0.10)
        ops: Dict[str, np.ndarray] = {}
        if HAS_CV2:
            ops = morph_ops(mask_u8)

        name_map = {
            "binary": "二值图",
            "low_freq": "低频(FFT)",
            "high_freq": "高频(FFT)",
            "erosion_1": "腐蚀x1",
            "erosion_2": "腐蚀x2",
            "erosion_3": "腐蚀x3",
            "dilation_1": "膨胀x1",
            "dilation_2": "膨胀x2",
            "dilation_3": "膨胀x3",
            "skeleton_1": "骨架迭代1",
            "skeleton_2": "骨架迭代2",
            "skeleton_3": "骨架迭代3",
            "morph_grad": "形态梯度",
            "sobel_edge": "Sobel边缘",
        }

        items = [
            ("binary", gray01),
            ("low_freq", low),
            ("high_freq", high),
        ]
        for k, v in ops.items():
            items.append((k, v))

        n = len(items)
        cols = 4
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = np.array(axes).reshape(rows, cols)

        for i, (name, img01) in enumerate(items):
            r, c = divmod(i, cols)
            axes[r, c].imshow(img01, cmap="gray")
            axes[r, c].set_title(name_map.get(name, name))
            axes[r, c].axis("off")

        for i in range(n, rows * cols):
            r, c = divmod(i, cols)
            axes[r, c].axis("off")

        legend_lines = [
            "图例：二值图/腐蚀x/膨胀x/骨架迭代x/形态梯度/Sobel边缘",
            "FFT=频域低/高频",
        ]
        fig.subplots_adjust(bottom=0.12)
        fig.text(0.01, 0.01, "\n".join(legend_lines), fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / "topology_ops.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def visualize_branch_feature_maps(self, binary_path: str, out_dir: Path, *, feature_layer: str = "layer4") -> None:
        _safe_mkdir(out_dir)
        binary_l = pil_load_l(binary_path, size=224)
        img_t = self._tensorize_inputs(binary_l)

        with torch.no_grad():
            feat = self._extract_backbone_feature(img_t, layer=feature_layer)

        extractor = self.model.topo_extractor_q
        branches = getattr(extractor, "branches", {})
        if not branches:
            return

        items: List[Tuple[str, np.ndarray]] = []
        base_gray = np.array(binary_l, dtype=np.float32) / 255.0
        base_rgb = np.stack([base_gray, base_gray, base_gray], axis=-1)


        legacy = branches["legacy"] if ("legacy" in branches) else None
        if legacy is not None and hasattr(legacy, "erosion_layers"):
            for i, erosion in enumerate(legacy.erosion_layers, start=1):
                eroded = erosion(feat)
                items.append((f"legacy_erosion_{i}", _feat_to_map(eroded)))


        sk_branch = branches["skeleton"] if ("skeleton" in branches) else None
        if sk_branch is not None and hasattr(sk_branch, "erosion_ops"):
            use_improved = bool(getattr(sk_branch, "use_relu_residual_norm", False))
            for i, (erosion, dilation) in enumerate(
                zip(sk_branch.erosion_ops, sk_branch.dilation_ops), start=1
            ):
                eroded = erosion(feat)
                opened = dilation(eroded)
                if use_improved:
                    skeleton = F.relu(feat - opened)
                    skeleton = skeleton + 0.1 * feat
                    denom = skeleton.abs().mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
                    skeleton = skeleton / denom
                else:
                    skeleton = feat - opened
                items.append((f"skeleton_{i}", _feat_to_map(skeleton)))


        edge = branches["edge"] if ("edge" in branches) else None
        if edge is not None and hasattr(edge, "sobel_x"):
            edge_x = F.conv2d(feat, edge.sobel_x, padding=1, groups=edge.input_channels)
            edge_y = F.conv2d(feat, edge.sobel_y, padding=1, groups=edge.input_channels)
            sobel = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-8)
            lap = torch.abs(F.conv2d(feat, edge.laplacian, padding=1, groups=edge.input_channels))
            dilated = edge.dilation(feat)
            eroded = edge.erosion(feat)
            morph_grad = dilated - eroded
            items.append(("edge_sobel", _feat_to_map(sobel)))
            items.append(("edge_laplacian", _feat_to_map(lap)))
            items.append(("edge_morph_grad", _feat_to_map(morph_grad)))


        freq = branches["frequency"] if ("frequency" in branches) else None
        if freq is not None:
            blur = F.avg_pool2d(feat, kernel_size=5, stride=1, padding=2)
            high = feat - blur
            band = F.avg_pool2d(feat, kernel_size=3, stride=2, padding=1)
            band = F.interpolate(band, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            items.append(("freq_low", _feat_to_map(blur)))
            items.append(("freq_high", _feat_to_map(high)))
            items.append(("freq_band", _feat_to_map(band)))

        if not items:
            return

        name_map = {
            "legacy_erosion_1": "legacy-腐蚀1",
            "legacy_erosion_2": "legacy-腐蚀2",
            "legacy_erosion_3": "legacy-腐蚀3",
            "legacy_erosion_4": "legacy-腐蚀4",
            "skeleton_1": "骨架1",
            "skeleton_2": "骨架2",
            "skeleton_3": "骨架3",
            "skeleton_4": "骨架4",
            "edge_sobel": "边缘-Sobel",
            "edge_laplacian": "边缘-Laplacian",
            "edge_morph_grad": "边缘-形态梯度",
            "freq_low": "频域-低频",
            "freq_high": "频域-高频",
            "freq_band": "频域-带通",
        }

        n = len(items)
        cols = 4
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = np.array(axes).reshape(rows, cols)
        for i, (name, img01) in enumerate(items):
            r, c = divmod(i, cols)
            overlay = overlay_heatmap(base_rgb, img01, alpha=0.45)
            axes[r, c].imshow(overlay)
            axes[r, c].set_title(name_map.get(name, name))
            axes[r, c].axis("off")
        for i in range(n, rows * cols):
            r, c = divmod(i, cols)
            axes[r, c].axis("off")

        fig.tight_layout()
        fig.savefig(out_dir / "branch_feature_maps.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def visualize_branch_weights(self, binary_path: str, out_dir: Path) -> None:
        _safe_mkdir(out_dir)
        binary_l = pil_load_l(binary_path, size=224)
        img_t = self._tensorize_inputs(binary_l)

        with torch.no_grad():
            feat = self.model.encoder_q(img_t)
            topo, branch_outputs = self.model.topo_extractor_q(feat)

        extractor = self.model.topo_extractor_q
        branch_names = list(getattr(extractor, "branch_names", []))
        if not branch_names:
            return


        branch_features = [branch_outputs[n] for n in branch_names]
        raw_concat = torch.cat(branch_features, dim=1)
        raw_concat = extractor.branch_dropout(raw_concat)
        context = extractor.context_projector(raw_concat)
        if extractor.use_adaptive_weights:
            weights = extractor.adaptive_weight_net(context)
        else:
            weights = F.softmax(extractor.static_branch_weights, dim=0).unsqueeze(0).expand(img_t.size(0), -1)

        weights_raw = weights.detach().clone()

        if len(branch_names) == 1:
            weights_used = torch.ones_like(weights_raw)
        else:
            weights_used = weights_raw
            if extractor.preserve_legacy_residual and ("legacy" in branch_names):
                legacy_idx = branch_names.index("legacy")
                weights_used = weights_used.clone()
                weights_used[:, legacy_idx] = 0.0
                denom = weights_used.sum(dim=1, keepdim=True).clamp_min(1e-6)
                weights_used = weights_used / denom
            if float(weights_used.sum().item()) < 1e-6:
                weights_used = torch.full_like(weights_used, 1.0 / weights_used.size(1))

        weights_np = weights_used[0].detach().cpu().numpy()
        weights_raw_np = weights_raw[0].detach().cpu().numpy()

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.bar(branch_names, weights_np)
        ax.set_ylim(0, 1)
        ax.set_title("拓扑支路权重")
        ax.set_ylabel("Weight")
        fig.tight_layout()
        fig.savefig(out_dir / "branch_weights.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


        with (out_dir / "branch_weights.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "branches": branch_names,
                    "weights": weights_np.tolist(),
                    "raw_weights": weights_raw_np.tolist(),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )


        topo_vec = topo[0].detach().cpu().numpy().tolist()
        with (out_dir / "topo_feature.json").open("w", encoding="utf-8") as f:
            json.dump({"topo_feature": topo_vec}, f, ensure_ascii=False)

    def visualize_gradcam(self, binary_path: str, out_dir: Path) -> None:
        _safe_mkdir(out_dir)
        binary_l = pil_load_l(binary_path, size=224)
        img_t = self._tensorize_inputs(binary_l)

        cam_target = self._get_gradcam_target()
        cam = GradCAMBackbone(self.model, cam_target)

        feat = self.model.encoder_q(img_t)
        visual = self.model.avgpool(feat).flatten(1)
        visual = self.model.visual_mlp_q(visual)
        loss = visual.norm(dim=1).mean()
        cam_map = cam.compute_from_loss(loss, out_hw=(224, 224))
        cam.close()

        if cam_map is None:
            return

        base = denorm_to_rgb01(img_t)
        overlay = overlay_heatmap(base, cam_map, alpha=0.45)
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(overlay)
        ax.set_title("GradCAM (visual)")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / "gradcam_visual.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ginseng visualization (v2)")
    p.add_argument("--checkpoint_dir", type=str, required=True, help="Directory with config.json and best_model.pth")
    p.add_argument("--output_dir", type=str, default="visualizations_v2", help="Output directory")
    p.add_argument("--csv", action="append", default=[], help="Dataset CSV in NAME=PATH format (repeatable)")
    p.add_argument("--num_samples", type=int, default=16, help="Samples per dataset")
    p.add_argument("--seed", type=int, default=42, help="Sampling seed")
    p.add_argument("--gradcam", action="store_true", help="Export Grad-CAM overlays")
    p.add_argument("--feature-layer", type=str, default="layer4", choices=["layer1", "layer2", "layer3", "layer4"], help="Backbone layer for branch feature maps")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    random.seed(args.seed)

    cfg, model_path = GinsengFeatureVisualizer.load_from_checkpoint_dir(args.checkpoint_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GinsengFeatureVisualizer.build_model(cfg, device)
    out_root = Path(args.output_dir) / Path(args.checkpoint_dir).name
    _safe_mkdir(out_root)

    viz = GinsengFeatureVisualizer(model=model, device=device, save_dir=out_root)
    viz.load_weights(model_path)

    csvs = _parse_named_paths(args.csv)
    if not csvs:
        raise ValueError("Provide at least one --csv NAME=PATH")

    for name, csv_path in csvs.items():
        ds_dir = out_root / name
        _safe_mkdir(ds_dir)

        samples = load_samples_from_csv(dataset_name=name, csv_path=csv_path)
        if not samples:
            continue

        rng = random.Random(args.seed)
        rng.shuffle(samples)
        pick = samples[: int(args.num_samples)]

        for idx, s in enumerate(pick):
            sample_dir = ds_dir / f"{idx:04d}_{Path(s.binary_path).stem}"
            _safe_mkdir(sample_dir)
            viz.visualize_topology_ops(s.binary_path, sample_dir)
            viz.visualize_branch_feature_maps(s.binary_path, sample_dir, feature_layer=args.feature_layer)
            viz.visualize_branch_weights(s.binary_path, sample_dir)
            if args.gradcam:
                viz.visualize_gradcam(s.binary_path, sample_dir)

    with (out_root / "README.txt").open("w", encoding="utf-8") as f:
        f.write("Outputs:\n")
        f.write("- <dataset>/<sample>/topology_ops.png: image-space topology ops\n")
        f.write("- <dataset>/<sample>/branch_weights.png: branch fusion weights\n")
        f.write("- <dataset>/<sample>/branch_weights.json: raw weights\n")
        f.write("- <dataset>/<sample>/topo_feature.json: fused topology feature\n")
        f.write("- <dataset>/<sample>/branch_feature_maps.png: branch feature maps\n")
        f.write("- <dataset>/<sample>/gradcam_visual.png: Grad-CAM overlay (if --gradcam)\n")


if __name__ == "__main__":
    main()
