from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import load_config
from model import ImprovedMoCoV3WithTopoSideline
from preprocess_utils import build_tensor_transform, load_grayscale_rgb, resize_with_mode

DISPLAY_CANVAS_WIDTH = 300
DISPLAY_CANVAS_HEIGHT = 200


@dataclass
class GradCAMCase:
    group: str
    query_image: str = ""
    reference_image: str = ""
    candidate_references: Tuple[str, ...] = ()
    tag: str = ""
    ssi: Optional[float] = None
    map_score: Optional[float] = None


class GradCAMHook:
    def __init__(self, model: torch.nn.Module, target_module: torch.nn.Module):
        self.model = model
        self.target_module = target_module
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._forward_handle = self.target_module.register_forward_hook(self._save_activation)
        try:
            self._backward_handle = self.target_module.register_full_backward_hook(self._save_gradient)
        except Exception:
            self._backward_handle = self.target_module.register_backward_hook(self._save_gradient)

    def close(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()

    def _save_activation(self, _module, _inputs, output):
        self.activations = output

    def _save_gradient(self, _module, _grad_input, grad_output):
        if isinstance(grad_output, (tuple, list)):
            self.gradients = grad_output[0]
        else:
            self.gradients = grad_output

    def compute(self, score: torch.Tensor, out_hw: Tuple[int, int]) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hook did not capture activations or gradients.")

        acts = self.activations
        grads = self.gradients
        if acts.ndim != 4 or grads.ndim != 4:
            raise RuntimeError("Grad-CAM expects 4D feature maps.")

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=out_hw, mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam / cam.max().clamp_min(1e-8)
        return cam.detach().cpu().numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grad-CAM visualization for single_topo.")
    parser.add_argument("--config", type=str, default=None, help="Optional config JSON path.")
    parser.add_argument("--override", action="append", default=[], help="Override key=value.")
    parser.add_argument(
        "--cases-json",
        type=str,
        default=str(Path(__file__).with_name("configs").joinpath("gradcam_cases.json")),
        help="Path to fixed Grad-CAM case definitions.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).with_name("visualizations_gradcam")),
        help="Directory for per-case outputs and summary figures.",
    )
    parser.add_argument(
        "--view-mode",
        type=str,
        default="contain224",
        help="Single inference view used for visualization, e.g. contain224 or stretch224.",
    )
    parser.add_argument(
        "--score-type",
        type=str,
        default="visual",
        choices=["fused", "visual", "topo"],
        help="Similarity target used for Grad-CAM backpropagation.",
    )
    parser.add_argument(
        "--paper-prefix",
        type=str,
        default="gradcam_single_topo_visual",
        help="Summary figure filename prefix.",
    )
    parser.add_argument(
        "--reference-mode",
        type=str,
        default="hard_positive",
        choices=["fixed", "easy_positive", "hard_positive"],
        help="How to choose the same-group reference used to drive Grad-CAM.",
    )
    return parser.parse_args()


def parse_view_mode(view_mode: str) -> Dict[str, int | str]:
    token = str(view_mode).strip().lower()
    match = re.fullmatch(r"(stretch|contain)(\d+)", token)
    if not match:
        raise ValueError(f"Unsupported view mode: {view_mode}")
    return {"name": token, "mode": match.group(1), "size": int(match.group(2))}


def to_rgb01(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)


def overlay_heatmap(base_rgb01: np.ndarray, heat01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    cmap = plt.get_cmap("jet")
    heat_rgb = cmap(np.clip(heat01, 0.0, 1.0))[..., :3].astype(np.float32)
    mixed = (1.0 - alpha) * base_rgb01.astype(np.float32) + alpha * heat_rgb
    return np.clip(mixed, 0.0, 1.0)


def build_foreground_mask_rgb01(rgb01: np.ndarray, threshold: float = 1e-3) -> np.ndarray:
    return rgb01.max(axis=2) > threshold


def crop_rgb01_to_mask(rgb01: np.ndarray, mask: np.ndarray, margin_ratio: float = 0.04) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return rgb01
    margin = max(4, int(max(rgb01.shape[:2]) * margin_ratio))
    left = max(0, int(xs.min()) - margin)
    top = max(0, int(ys.min()) - margin)
    right = min(rgb01.shape[1], int(xs.max()) + margin + 1)
    bottom = min(rgb01.shape[0], int(ys.max()) + margin + 1)
    return rgb01[top:bottom, left:right]


def crop_rgb01_pair_to_mask(
    first_rgb01: np.ndarray,
    second_rgb01: np.ndarray,
    mask: np.ndarray,
    margin_ratio: float = 0.04,
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop two aligned images with one shared foreground bounding box."""
    if first_rgb01.shape != second_rgb01.shape:
        raise ValueError(
            "Paired display images must have identical shapes before cropping: "
            f"{first_rgb01.shape} != {second_rgb01.shape}"
        )
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return first_rgb01, second_rgb01
    margin = max(4, int(max(first_rgb01.shape[:2]) * margin_ratio))
    left = max(0, int(xs.min()) - margin)
    top = max(0, int(ys.min()) - margin)
    right = min(first_rgb01.shape[1], int(xs.max()) + margin + 1)
    bottom = min(first_rgb01.shape[0], int(ys.max()) + margin + 1)
    return (
        first_rgb01[top:bottom, left:right],
        second_rgb01[top:bottom, left:right],
    )


def contain_rgb01(
    rgb01: np.ndarray,
    width: int = DISPLAY_CANVAS_WIDTH,
    height: int = DISPLAY_CANVAS_HEIGHT,
    background_rgb: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    pil = Image.fromarray(np.clip(rgb01 * 255.0, 0.0, 255.0).astype(np.uint8))
    pil.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), background_rgb)
    offset_x = (width - pil.width) // 2
    offset_y = (height - pil.height) // 2
    canvas.paste(pil, (offset_x, offset_y))
    return to_rgb01(canvas)


def estimate_alignment(mask: np.ndarray) -> Tuple[float, bool]:
    ys, xs = np.nonzero(mask)
    if len(xs) < 32:
        return 0.0, False

    coords = np.column_stack([xs, ys]).astype(np.float32)
    coords -= coords.mean(axis=0, keepdims=True)
    cov = np.cov(coords, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major_axis = eigvecs[:, int(np.argmax(eigvals))]
    angle = float(np.degrees(np.arctan2(float(major_axis[1]), float(major_axis[0]))))
    if angle > 90:
        angle -= 180
    if angle < -90:
        angle += 180

    column_mass = mask.sum(axis=0).astype(np.float32)
    if float(column_mass.sum()) <= 0:
        return angle, False
    center_x = float(np.dot(np.arange(column_mass.size, dtype=np.float32), column_mass) / column_mass.sum())
    should_flip = center_x > (mask.shape[1] / 2.0)
    return angle, should_flip


def apply_alignment_to_rgb01(
    rgb01: np.ndarray,
    angle: float,
    should_flip: bool,
    crop_mask: Optional[np.ndarray] = None,
    fill_rgb: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    pil = Image.fromarray(np.clip(rgb01 * 255.0, 0.0, 255.0).astype(np.uint8))
    aligned = pil.rotate(-angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=fill_rgb)
    if should_flip:
        aligned = aligned.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    aligned_rgb01 = to_rgb01(aligned)
    if crop_mask is not None:
        aligned_rgb01 = crop_rgb01_to_mask(aligned_rgb01, crop_mask)
    return aligned_rgb01


def standardize_query_display(query_rgb01: np.ndarray, overlay_rgb01: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = build_foreground_mask_rgb01(query_rgb01)
    angle, should_flip = estimate_alignment(mask)
    aligned_query = apply_alignment_to_rgb01(query_rgb01, angle, should_flip)
    aligned_overlay = apply_alignment_to_rgb01(overlay_rgb01, angle, should_flip)
    aligned_mask = build_foreground_mask_rgb01(aligned_query)
    aligned_query, aligned_overlay = crop_rgb01_pair_to_mask(
        aligned_query,
        aligned_overlay,
        aligned_mask,
    )
    if aligned_query.shape[0] > aligned_query.shape[1]:
        aligned_query = np.rot90(aligned_query, k=1)
        aligned_overlay = np.rot90(aligned_overlay, k=1)
        aligned_mask = build_foreground_mask_rgb01(aligned_query)
        column_mass = aligned_mask.sum(axis=0).astype(np.float32)
        if float(column_mass.sum()) > 0:
            center_x = float(np.dot(np.arange(column_mass.size, dtype=np.float32), column_mass) / column_mass.sum())
            if center_x > (aligned_query.shape[1] / 2.0):
                aligned_query = np.fliplr(aligned_query).copy()
                aligned_overlay = np.fliplr(aligned_overlay).copy()
        aligned_mask = build_foreground_mask_rgb01(aligned_query)
        aligned_query, aligned_overlay = crop_rgb01_pair_to_mask(
            aligned_query,
            aligned_overlay,
            aligned_mask,
        )
    return contain_rgb01(aligned_query), contain_rgb01(aligned_overlay)


def standardize_single_display(rgb01: np.ndarray) -> np.ndarray:
    mask = build_foreground_mask_rgb01(rgb01)
    angle, should_flip = estimate_alignment(mask)
    aligned = apply_alignment_to_rgb01(rgb01, angle, should_flip)
    aligned_mask = build_foreground_mask_rgb01(aligned)
    aligned = crop_rgb01_to_mask(aligned, aligned_mask)
    if aligned.shape[0] > aligned.shape[1]:
        aligned = np.rot90(aligned, k=1)
        aligned_mask = build_foreground_mask_rgb01(aligned)
        column_mass = aligned_mask.sum(axis=0).astype(np.float32)
        if float(column_mass.sum()) > 0:
            center_x = float(np.dot(np.arange(column_mass.size, dtype=np.float32), column_mass) / column_mass.sum())
            if center_x > (aligned.shape[1] / 2.0):
                aligned = np.fliplr(aligned).copy()
        aligned = crop_rgb01_to_mask(aligned, build_foreground_mask_rgb01(aligned))
    return contain_rgb01(aligned)


def load_cases(path: str) -> List[GradCAMCase]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    raw_cases = payload.get("cases", payload)
    cases: List[GradCAMCase] = []
    for item in raw_cases:
        cases.append(
            GradCAMCase(
                group=str(item["group"]),
                query_image=str(item.get("query_image", "")),
                reference_image=str(item.get("reference_image", "")),
                candidate_references=tuple(str(path) for path in item.get("candidate_references", [])),
                tag=str(item.get("tag", "")),
                ssi=float(item["ssi"]) if item.get("ssi") is not None else None,
                map_score=float(item["map_score"]) if item.get("map_score") is not None else None,
            )
        )
    return cases


def parse_case_from_results(results_root: Path, group: str) -> Tuple[str, Tuple[str, ...]]:
    result_file = results_root / group / f"{group}_results.txt"
    if not result_file.exists():
        raise FileNotFoundError(f"Missing batch retrieval result file: {result_file}")

    text = result_file.read_text(encoding="utf-8", errors="replace")
    query_image = ""
    ranked_paths: List[str] = []
    relevant_paths: List[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("query_image:"):
            query_image = line.split(":", 1)[1].strip()
            continue
        if line.startswith("relevant\t"):
            relevant_paths.append(line.split("\t", 1)[1].strip())
            continue
        match = re.match(r"^\d+\t[-+0-9.eE]+\t(.+)$", line)
        if match:
            ranked_paths.append(match.group(1).strip())

    if not query_image:
        raise RuntimeError(f"Failed to parse query_image from {result_file}")
    if not relevant_paths:
        raise RuntimeError(f"Failed to parse relevant_paths from {result_file}")

    relevant_set = {Path(path).as_posix().lower(): path for path in relevant_paths}
    ranked_relevant_paths: List[str] = []
    for path in ranked_paths:
        normalized = Path(path).as_posix().lower()
        if normalized in relevant_set:
            ranked_relevant_paths.append(relevant_set[normalized])
    if not ranked_relevant_paths:
        ranked_relevant_paths = relevant_paths.copy()

    deduplicated = tuple(dict.fromkeys(ranked_relevant_paths))
    return query_image, deduplicated


def resolve_case_paths(case: GradCAMCase, results_root: Path) -> GradCAMCase:
    query_image = case.query_image
    reference_image = case.reference_image
    candidate_references = tuple(path for path in case.candidate_references if Path(path).exists())

    needs_resolution = (not query_image) or (not Path(query_image).exists()) or (
        reference_image and not Path(reference_image).exists()
    ) or (not reference_image and not candidate_references)
    if not needs_resolution:
        return case

    query_image, parsed_candidates = parse_case_from_results(results_root, case.group)
    if reference_image and Path(reference_image).exists():
        candidate_references = tuple(dict.fromkeys((reference_image,) + parsed_candidates))
    else:
        candidate_references = parsed_candidates
        reference_image = parsed_candidates[0] if parsed_candidates else ""
    return GradCAMCase(
        group=case.group,
        query_image=query_image,
        reference_image=reference_image,
        candidate_references=candidate_references,
        tag=case.tag,
        ssi=case.ssi,
        map_score=case.map_score,
    )


def resolve_model_path(cfg: Dict[str, object]) -> Path:
    model_path = cfg.get("model_path")
    if not model_path:
        raise ValueError("Missing model_path in merged config.")
    return Path(str(model_path)).resolve()


def build_model(cfg: Dict[str, object], device: torch.device) -> ImprovedMoCoV3WithTopoSideline:
    return ImprovedMoCoV3WithTopoSideline(
        feature_dim=int(cfg.get("feature_dim", 256)),
        topo_dim=int(cfg.get("topo_dim", 128)),
        K=int(cfg.get("K", 4096)),
        m=float(cfg.get("m", 0.999)),
        T=float(cfg.get("T", 0.07)),
        topo_weight=float(cfg.get("topo_weight", 0.35)),
        num_erosion_levels=int(cfg.get("num_erosion_levels", 4)),
        erosion_kernel_size=int(cfg.get("erosion_kernel_size", 3)),
        topology_operator=str(cfg.get("topology_operator", "min")),
        topology_negative_source=str(
            cfg.get("topology_negative_source", "queue")
        ),
        use_cbam=bool(cfg.get("use_cbam", True)),
        backbone_name=str(cfg.get("backbone_name", "resnet50")),
        pretrained_backbone=False,
        device=device,
    )


def clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        cleaned[new_key] = value
    return cleaned


def load_weights(model: torch.nn.Module, model_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Unexpected checkpoint format: {type(checkpoint)}")
    model.load_state_dict(clean_state_dict(checkpoint), strict=True)
    model.eval()


def build_input_tensor(
    image_path: str,
    view_spec: Dict[str, int | str],
    tensor_transform,
    pad_value: int,
    device: torch.device,
) -> Tuple[torch.Tensor, Image.Image]:
    pil = load_grayscale_rgb(resolve_existing_image_path(image_path))
    processed = resize_with_mode(pil, view_spec, pad_value=pad_value)
    tensor = tensor_transform(processed).unsqueeze(0).to(device)
    return tensor, processed


def get_target_module(model: ImprovedMoCoV3WithTopoSideline) -> torch.nn.Module:
    encoder = model.encoder_q
    if isinstance(encoder, torch.nn.Sequential) and len(encoder) > 0:
        return encoder[-1]
    return encoder


def resolve_existing_image_path(image_path: str) -> str:
    path = Path(image_path)
    if path.exists():
        return str(path)

    raw = str(path)
    if "library_binary" in raw:
        alt = Path(raw.replace("library_binary", "library_query_datasets"))
        if alt.exists():
            return str(alt)

    search_roots = [
        Path(__file__).resolve().parents[1] / "data" / "library_binary",
        Path(__file__).resolve().parents[1] / "data" / "gallery",
    ]
    for root in search_roots:
        candidate = root / path.name
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(f"Image path not found: {image_path}")


def forward_embeddings(
    model: ImprovedMoCoV3WithTopoSideline,
    img_tensor: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    feat = model.encoder_q(img_tensor)
    feat = model.cbam_q(feat)
    visual = model.avgpool(feat).flatten(1)
    visual = F.normalize(model.visual_mlp_q(visual), dim=1)
    topo, _ = model.topo_extractor_q(feat)
    fused = F.normalize(torch.cat([visual, topo], dim=1), dim=1)
    return {"visual": visual, "topo": topo, "fused": fused}


def select_reference_payload(
    model: ImprovedMoCoV3WithTopoSideline,
    case: GradCAMCase,
    query_embed_static: Dict[str, torch.Tensor],
    tensor_transform,
    view_spec: Dict[str, int | str],
    pad_value: int,
    device: torch.device,
    reference_mode: str,
) -> Dict[str, object]:
    candidate_paths = tuple(dict.fromkeys(case.candidate_references or ((case.reference_image,) if case.reference_image else ())))
    if not candidate_paths:
        raise RuntimeError(f"No same-group positive candidates found for group {case.group}.")

    payloads: List[Dict[str, object]] = []
    with torch.no_grad():
        for candidate_path in candidate_paths:
            ref_tensor, ref_img = build_input_tensor(candidate_path, view_spec, tensor_transform, pad_value, device)
            ref_embed = forward_embeddings(model, ref_tensor)
            payloads.append(
                {
                    "path": candidate_path,
                    "image": ref_img,
                    "embed": ref_embed,
                    "similarities": {
                        key: float((query_embed_static[key] * ref_embed[key]).sum(dim=1).item())
                        for key in ("fused", "visual", "topo")
                    },
                }
            )

    if reference_mode == "easy_positive":
        return max(payloads, key=lambda item: item["similarities"]["fused"])
    if reference_mode == "hard_positive":
        return min(payloads, key=lambda item: item["similarities"]["fused"])
    return payloads[0]


def compute_case_visualization(
    model: ImprovedMoCoV3WithTopoSideline,
    case: GradCAMCase,
    tensor_transform,
    view_spec: Dict[str, int | str],
    pad_value: int,
    device: torch.device,
    score_type: str,
    reference_mode: str,
) -> Dict[str, object]:
    query_tensor, query_img = build_input_tensor(case.query_image, view_spec, tensor_transform, pad_value, device)
    with torch.no_grad():
        query_embed_static = forward_embeddings(model, query_tensor)
    selected_reference = select_reference_payload(
        model=model,
        case=case,
        query_embed_static=query_embed_static,
        tensor_transform=tensor_transform,
        view_spec=view_spec,
        pad_value=pad_value,
        device=device,
        reference_mode=reference_mode,
    )
    ref_img = selected_reference["image"]
    ref_embed = selected_reference["embed"]

    cam = GradCAMHook(model, get_target_module(model))
    query_embed = forward_embeddings(model, query_tensor)

    fused_score = (query_embed["fused"] * ref_embed["fused"]).sum(dim=1)
    visual_score = (query_embed["visual"] * ref_embed["visual"]).sum(dim=1)
    topo_score = (query_embed["topo"] * ref_embed["topo"]).sum(dim=1)
    score_lookup = {"fused": fused_score, "visual": visual_score, "topo": topo_score}
    cam_map = cam.compute(score_lookup[score_type].mean(), out_hw=(query_tensor.shape[-2], query_tensor.shape[-1]))
    cam.close()

    query_rgb = to_rgb01(query_img)
    ref_rgb = to_rgb01(ref_img)
    foreground_mask = build_foreground_mask_rgb01(query_rgb).astype(np.float32)
    if float(foreground_mask.max()) > 0:
        cam_map = cam_map * foreground_mask
        cam_peak = float(cam_map.max())
        if cam_peak > 1e-8:
            cam_map = cam_map / cam_peak
    overlay = overlay_heatmap(query_rgb, cam_map, alpha=0.45)
    overlay[foreground_mask <= 0] = 0.0
    query_display_rgb, overlay_display_rgb = standardize_query_display(query_rgb, overlay)
    reference_display_rgb = standardize_single_display(ref_rgb)

    return {
        "group": case.group,
        "tag": case.tag,
        "ssi": case.ssi,
        "map_score": case.map_score,
        "query_path": case.query_image,
        "reference_path": str(selected_reference["path"]),
        "query_rgb": query_display_rgb,
        "reference_rgb": reference_display_rgb,
        "overlay": overlay_display_rgb,
        "cam_map": cam_map,
        "reference_mode": reference_mode,
        "reference_scores": selected_reference["similarities"],
        "scores": {
            "fused": float(fused_score.item()),
            "visual": float(visual_score.item()),
            "topo": float(topo_score.item()),
        },
    }


def save_case_artifacts(output_dir: Path, payload: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    query_rgb = payload["query_rgb"]
    reference_rgb = payload["reference_rgb"]
    overlay = payload["overlay"]
    cam_map = payload["cam_map"]

    plt.imsave(output_dir / "query.png", query_rgb)
    plt.imsave(output_dir / "reference.png", reference_rgb)
    plt.imsave(output_dir / "gradcam_overlay.png", overlay)
    plt.imsave(output_dir / "gradcam_heatmap.png", cam_map, cmap="jet")

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    axes[0].imshow(query_rgb)
    axes[0].set_title("Query")
    axes[1].imshow(reference_rgb)
    axes[1].set_title("Positive")
    axes[2].imshow(overlay)
    axes[2].set_title("Grad-CAM")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "case_triplet.png", dpi=180, bbox_inches="tight")
    fig.savefig(output_dir / "case_triplet.pdf", bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "group": payload["group"],
        "tag": payload["tag"],
        "ssi": payload["ssi"],
        "map_score": payload["map_score"],
        "query_path": payload["query_path"],
        "reference_path": payload["reference_path"],
        "reference_mode": payload["reference_mode"],
        "reference_scores": payload["reference_scores"],
        "scores": payload["scores"],
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


def build_summary_figure(results: List[Dict[str, object]], output_prefix: Path) -> None:
    cols = len(results)
    fig, axes = plt.subplots(
        3,
        cols + 1,
        figsize=(3.0 * cols + 0.9, 6.4),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [0.42] + [1.0] * cols},
    )
    if cols == 1:
        axes = np.array(axes).reshape(3, 2)

    row_labels = ["(a)", "(b)", "(c)"]
    for row in range(3):
        axes[row, 0].axis("off")
        axes[row, 0].text(
            0.5,
            0.5,
            row_labels[row],
            ha="center",
            va="center",
            fontsize=10,
        )

    for col, item in enumerate(results):
        title_lines = [f"Group {item['group']}"]
        if item.get("tag"):
            title_lines.append(str(item["tag"]))
        if item.get("ssi") is not None:
            title_lines.append(f"SSI={float(item['ssi']):.3f}")

        panel_col = col + 1
        axes[0, panel_col].imshow(item["query_rgb"])
        axes[1, panel_col].imshow(item["reference_rgb"])
        axes[2, panel_col].imshow(item["overlay"])
        axes[0, panel_col].set_title("\n".join(title_lines), fontsize=10)

        for row in range(3):
            axes[row, panel_col].axis("off")

    fig.savefig(output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_config("batch_retrieval", config_path=args.config, kv_overrides=args.override)
    device = torch.device("cuda" if bool(cfg.get("use_gpu", True)) and torch.cuda.is_available() else "cpu")
    tensor_transform = build_tensor_transform(cfg)
    view_spec = parse_view_mode(args.view_mode)
    pad_value = int(cfg.get("tta_pad_value", 0))
    results_root = Path(str(cfg.get("output_folder", ""))).resolve()

    model = build_model(cfg, device)
    load_weights(model, resolve_model_path(cfg), device)

    cases = [resolve_case_paths(case, results_root) for case in load_cases(args.cases_json)]
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, object]] = []
    for case in cases:
        result = compute_case_visualization(
            model=model,
            case=case,
            tensor_transform=tensor_transform,
            view_spec=view_spec,
            pad_value=pad_value,
            device=device,
            score_type=args.score_type,
            reference_mode=args.reference_mode,
        )
        case_dir = output_root / f"group_{case.group}"
        save_case_artifacts(case_dir, result)
        results.append(result)

    build_summary_figure(results, output_root / args.paper_prefix)

    with (output_root / "README.txt").open("w", encoding="utf-8") as handle:
        handle.write("Outputs:\n")
        handle.write("- group_xxx/query.png: aligned query view used in the paper figure.\n")
        handle.write("- group_xxx/reference.png: aligned same-group positive selected for Grad-CAM scoring.\n")
        handle.write("- group_xxx/gradcam_overlay.png: Grad-CAM overlay on the aligned query image.\n")
        handle.write("- group_xxx/case_triplet.pdf: three-panel summary for one case.\n")
        handle.write(f"- {args.paper_prefix}.pdf/.png: manuscript-ready multi-case summary.\n")


if __name__ == "__main__":
    main()
