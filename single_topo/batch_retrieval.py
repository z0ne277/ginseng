import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from config import load_config
from model import ImprovedMoCoV3WithTopoSideline
from preprocess_utils import (
    build_tensor_transform,
    build_tta_specs,
    build_tta_weights,
    load_grayscale_rgb,
    resize_with_mode,
)


config = {}
tensor_transform = None
tta_specs = []
tta_weights = []


def parse_args():
    parser = argparse.ArgumentParser(description="Batch retrieval for 2025-11-8 model")
    parser.add_argument("--config", type=str, default=None, help="Optional config JSON path")
    parser.add_argument("--override", action="append", default=[], help="Override key=value")
    return parser.parse_args()


def load_runtime_config():
    args = parse_args()
    cfg = load_config("batch_retrieval", config_path=args.config, kv_overrides=args.override)
    query_groups_json = cfg.get("query_groups_json")
    if query_groups_json:
        qpath = Path(query_groups_json)
        if not qpath.is_absolute():
            qpath = (Path(__file__).parent / qpath).resolve()
        with qpath.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        cfg["query_groups"] = payload.get("query_groups", [])
    else:
        cfg["query_groups"] = cfg.get("query_groups", [])
    return cfg


def normalize_path(path):
    return str(Path(path).resolve()).replace("\\", "/").lower()


def clean_query_groups(query_groups_raw):
    cleaned_groups = []
    skipped_groups = []

    for idx, group in enumerate(query_groups_raw):
        group_name = group.get("name", f"group_{idx}")
        query_img = group.get("query_image")
        if not query_img or not Path(query_img).exists():
            skipped_groups.append(
                {
                    "index": idx,
                    "name": group_name,
                    "reason": f"Query image not found: {query_img}",
                }
            )
            continue

        same_ginsengs = group.get("same_ginsengs", [])
        valid_ginsengs = [
            path
            for path in same_ginsengs
            if Path(path).exists() and normalize_path(path) != normalize_path(query_img)
        ]
        if not valid_ginsengs:
            skipped_groups.append(
                {
                    "index": idx,
                    "name": group_name,
                    "reason": "No valid relevant images found",
                }
            )
            continue

        cleaned_groups.append(
            {
                "name": group_name,
                "query_image": query_img,
                "same_ginsengs": valid_ginsengs,
            }
        )

    return cleaned_groups, skipped_groups


def get_metric_k_values(cfg):
    raw_values = cfg.get("metric_k_values", [1, 5, 10, 20])
    values = sorted({int(v) for v in raw_values if int(v) > 0})
    if not values:
        raise ValueError("metric_k_values must contain at least one positive integer")
    return tuple(values)


def extract_single_embedding(model, img_tensor):
    feature_type = str(config.get("feature_type", "both")).lower()
    if feature_type == "visual":
        feat = model.extract_features(
            img_tensor,
            use_query_encoder=True,
            feature_type="visual",
        )
        return F.normalize(feat, dim=1)

    if feature_type == "topo":
        feat = model.extract_features(
            img_tensor,
            use_query_encoder=True,
            feature_type="topo",
        )
        return F.normalize(feat, dim=1)

    visual_feat = model.extract_features(
        img_tensor,
        use_query_encoder=True,
        feature_type="visual",
    )
    topo_feat = model.extract_features(
        img_tensor,
        use_query_encoder=True,
        feature_type="topo",
    )
    fused_feat = torch.cat([visual_feat, topo_feat], dim=1)
    return F.normalize(fused_feat, dim=1)


def get_query_feature(model, img_path, device):
    img = load_grayscale_rgb(img_path)
    pad_value = int(config.get("tta_pad_value", 0))
    view_features = []

    with torch.no_grad():
        for spec in tta_specs:
            processed_img = resize_with_mode(img, spec, pad_value=pad_value)
            img_tensor = tensor_transform(processed_img).unsqueeze(0).to(device)
            view_features.append(extract_single_embedding(model, img_tensor).squeeze(0))

    weight_tensor = torch.tensor(
        tta_weights,
        dtype=view_features[0].dtype,
        device=view_features[0].device,
    ).view(-1, 1)
    query_feat = (torch.stack(view_features, dim=0) * weight_tensor).sum(dim=0, keepdim=True)
    query_feat = F.normalize(query_feat, dim=1)
    return query_feat.squeeze(0)


def tta_weights_match(saved_weights, current_weights, tolerance=1e-6):
    if not isinstance(saved_weights, (list, tuple)):
        return False
    try:
        parsed = [float(weight) for weight in saved_weights]
    except (TypeError, ValueError):
        return False
    if len(parsed) != len(current_weights):
        return False
    return all(abs(left - right) <= tolerance for left, right in zip(parsed, current_weights))


def retrieve_topk(query_img_path, model, feats_tensor, feats_paths, device, k=10):
    query_feat = get_query_feature(model, query_img_path, device)
    feats_tensor_norm = F.normalize(feats_tensor, dim=1)
    query_feat_norm = F.normalize(query_feat.unsqueeze(0), dim=1).squeeze(0)
    sims = torch.mv(feats_tensor_norm, query_feat_norm)
    topk_vals, topk_indices = torch.topk(sims, min(k + 5, len(sims)))

    query_norm = normalize_path(query_img_path)
    results = []
    for score, index in zip(topk_vals.tolist(), topk_indices.tolist()):
        path = feats_paths[index]
        if normalize_path(path) == query_norm:
            continue
        results.append(
            {
                "path": path,
                "score": float(score),
                "rank": len(results) + 1,
            }
        )
        if len(results) >= k:
            break
    return results


class RetrievalEvaluator:
    def __init__(self, results, relevant_paths, query_image_path=None, k_values=(1, 5, 10, 20)):
        self.results = results
        self.relevant_paths = {normalize_path(path) for path in relevant_paths}
        if query_image_path is not None:
            self.relevant_paths.discard(normalize_path(query_image_path))
        self.k_values = k_values
        self.metrics = self._calculate_metrics()

    def _is_relevant(self, result_path):
        return normalize_path(result_path) in self.relevant_paths

    def _calculate_mrr(self):
        for result in self.results:
            if self._is_relevant(result["path"]):
                return 1.0 / result["rank"]
        return 0.0

    def _calculate_map(self):
        if not self.relevant_paths:
            return 0.0
        sum_precisions = 0.0
        num_relevant_found = 0
        for result in self.results:
            if self._is_relevant(result["path"]):
                num_relevant_found += 1
                sum_precisions += num_relevant_found / result["rank"]
        return sum_precisions / len(self.relevant_paths)

    def _calculate_recall_at_k(self, k):
        if not self.relevant_paths:
            return 0.0
        relevant_in_topk = sum(1 for result in self.results[:k] if self._is_relevant(result["path"]))
        return relevant_in_topk / len(self.relevant_paths)

    def _calculate_metrics(self):
        metrics = {"mrr": self._calculate_mrr(), "map": self._calculate_map()}
        for k in self.k_values:
            metrics[f"recall@{k}"] = self._calculate_recall_at_k(k)
        return metrics

    def get_relevant_ranks(self):
        return sorted(result["rank"] for result in self.results if self._is_relevant(result["path"]))

    def get_metrics_dict(self):
        return self.metrics


class BatchEvaluator:
    def __init__(self, metric_k_values):
        self.results = []
        self.metric_k_values = tuple(metric_k_values)
        self.metric_keys = ["mrr", "map"] + [f"recall@{k}" for k in self.metric_k_values]

    def add_result(self, query_name, evaluator):
        self.results.append(
            {
                "query_name": query_name,
                "metrics": evaluator.get_metrics_dict(),
                "num_relevant": len(evaluator.relevant_paths),
                "found": len(evaluator.get_relevant_ranks()),
            }
        )

    def get_average_metrics(self):
        if not self.results:
            return {}
        avg = {key: 0.0 for key in self.metric_keys}
        for result in self.results:
            for key in avg:
                avg[key] += result["metrics"][key]
        count = len(self.results)
        return {key: value / count for key, value in avg.items()}

    def save_summary(self, output_folder):
        output_dir = Path(output_folder)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "batch_summary.txt"
        averages = self.get_average_metrics()
        metric_columns = [("mrr", "mrr"), ("map", "map")] + [
            (f"recall@{k}", f"r@{k}") for k in self.metric_k_values
        ]
        with summary_path.open("w", encoding="utf-8") as handle:
            handle.write("=" * 100 + "\n")
            handle.write("single_topo batch retrieval summary\n")
            handle.write("=" * 100 + "\n")
            handle.write(f"total_queries: {len(self.results)}\n\n")
            header = f"{'query':<24} " + " ".join(f"{label:<10}" for _, label in metric_columns) + f" {'found':<10}\n"
            handle.write(header)
            handle.write("-" * 100 + "\n")
            for result in self.results:
                metrics = result["metrics"]
                found = f"{result['found']}/{result['num_relevant']}"
                metric_text = " ".join(f"{metrics[key]:<10.4f}" for key, _ in metric_columns)
                handle.write(f"{result['query_name']:<24} {metric_text} {found:<10}\n")
            if averages:
                handle.write("-" * 100 + "\n")
                metric_text = " ".join(f"{averages[key]:<10.4f}" for key, _ in metric_columns)
                handle.write(f"{'AVERAGE':<24} {metric_text}\n")
            handle.write("=" * 100 + "\n")
        return summary_path


def load_model(device):
    model = ImprovedMoCoV3WithTopoSideline(
        feature_dim=config["feature_dim"],
        topo_dim=config["topo_dim"],
        K=config["K"],
        m=config["m"],
        T=config["T"],
        topo_weight=config["topo_weight"],
        num_erosion_levels=config["num_erosion_levels"],
        erosion_kernel_size=config.get("erosion_kernel_size", 3),
        topology_operator=config.get("topology_operator", "min"),
        topology_negative_source=config.get("topology_negative_source", "queue"),
        use_cbam=config.get("use_cbam", True),
        backbone_name=config.get("backbone_name", "resnet50"),
        pretrained_backbone=False,
        device=device,
    )
    checkpoint = torch.load(config["model_path"], map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def save_results_to_file(
    query_group_name,
    query_img_path,
    results,
    same_ginseng_paths,
    metrics,
    metric_k_values,
    output_folder,
):
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{query_group_name}_results.txt"
    with result_path.open("w", encoding="utf-8") as handle:
        handle.write("=" * 120 + "\n")
        handle.write("single_topo retrieval results\n")
        handle.write("=" * 120 + "\n")
        handle.write(f"query_group: {query_group_name}\n")
        handle.write(f"query_image: {query_img_path}\n")
        handle.write(f"num_relevant: {len(same_ginseng_paths)}\n")
        handle.write(f"mrr: {metrics['mrr']:.4f}\n")
        handle.write(f"map: {metrics['map']:.4f}\n")
        for k in metric_k_values:
            handle.write(f"recall@{k}: {metrics[f'recall@{k}']:.4f}\n")
        handle.write("-" * 120 + "\n")
        for result in results:
            handle.write(f"{result['rank']}\t{result['score']:.6f}\t{result['path']}\n")
        handle.write("-" * 120 + "\n")
        for path in same_ginseng_paths:
            handle.write(f"relevant\t{path}\n")


def main():
    global config, tensor_transform, tta_specs, tta_weights
    config = load_runtime_config()
    tensor_transform = build_tensor_transform(config)
    tta_specs = build_tta_specs(config)
    tta_weights = build_tta_weights(config, tta_specs)
    metric_k_values = get_metric_k_values(config)

    use_gpu = bool(config.get("use_gpu", True)) and torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    print(f"Device: {device}")
    print(f"TTA modes: {[spec['name'] for spec in tta_specs]}")
    print(f"TTA weights: {[round(weight, 4) for weight in tta_weights]}")
    print(f"Metric k values: {list(metric_k_values)}")

    cleaned_groups, skipped_groups = clean_query_groups(config["query_groups"])
    print(f"Valid query groups: {len(cleaned_groups)}")
    print(f"Skipped query groups: {len(skipped_groups)}")
    if not cleaned_groups:
        raise RuntimeError("No valid query groups found")

    model = load_model(device)
    data = torch.load(config["feature_path"], map_location=device)
    if not isinstance(data, dict) or "features" not in data or "paths" not in data:
        raise RuntimeError("feature_path must contain a dict with features and paths")

    feats_tensor = data["features"].to(device)
    feats_paths = data["paths"]
    saved_tta_modes = data.get("tta_modes")
    current_tta_modes = [spec["name"] for spec in tta_specs]
    if saved_tta_modes and list(saved_tta_modes) != current_tta_modes:
        print(f"Warning: feature file TTA modes {saved_tta_modes} != current query TTA modes {current_tta_modes}")
    saved_tta_weights = data.get("tta_weights")
    if saved_tta_weights and not tta_weights_match(saved_tta_weights, tta_weights):
        print(f"Warning: feature file TTA weights {saved_tta_weights} != current query TTA weights {tta_weights}")

    batch_evaluator = BatchEvaluator(metric_k_values)
    retrieve_k = max(int(config.get("top_k", 10)), max(metric_k_values))
    for index, group in enumerate(cleaned_groups, start=1):
        print(f"[{index}/{len(cleaned_groups)}] {group['name']}")
        results = retrieve_topk(
            group["query_image"],
            model,
            feats_tensor,
            feats_paths,
            device,
            k=retrieve_k,
        )
        evaluator = RetrievalEvaluator(
            results=results,
            relevant_paths=group["same_ginsengs"],
            query_image_path=group["query_image"],
            k_values=metric_k_values,
        )
        batch_evaluator.add_result(group["name"], evaluator)
        save_results_to_file(
            group["name"],
            group["query_image"],
            results,
            group["same_ginsengs"],
            evaluator.get_metrics_dict(),
            metric_k_values,
            Path(config["output_folder"]) / group["name"],
        )

    summary_path = batch_evaluator.save_summary(config["output_folder"])
    averages = batch_evaluator.get_average_metrics()
    if averages:
        print(f"Average mrr: {averages['mrr']:.4f}")
        print(f"Average map: {averages['map']:.4f}")
        print(f"Average recall@10: {averages['recall@10']:.4f}")
        print(f"Average recall@5: {averages['recall@5']:.4f}")
        print(f"Average recall@1: {averages['recall@1']:.4f}")
        if "recall@20" in averages:
            print(f"Average recall@20: {averages['recall@20']:.4f}")
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
