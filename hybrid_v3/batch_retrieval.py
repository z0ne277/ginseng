"""
改进多支路拓扑网络的批量检索和评估脚本

功能:
1. 批量处理多个查询组
2. 对每个查询计算评估指标: MRR, MAP, Recall@K
3. 生成整体评估报告
4. 可视化每个查询的检索结果

基于: hybrid/model.py (MoCoV3HybridTopo)
"""

import os
import sys
import argparse
import shutil
import torch
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib
from pathlib import Path
from PIL import Image
from torchvision import transforms
import json

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

from model import MoCoV3HybridTopo
from config import load_config

# ============================================================
# 配置参数
# ============================================================
# ??????
# Global runtime config (filled in main)
config = {}
image_preprocess = None

def parse_args():
    parser = argparse.ArgumentParser(description='Batch retrieval and evaluation')
    parser.add_argument('--config', type=str, default=None, help='Optional config JSON path')
    parser.add_argument('--override', action='append', default=[], help='Override key=value (repeatable)')
    return parser.parse_args()

def load_runtime_config():
    args = parse_args()
    cfg = load_config('batch_retrieval', config_path=args.config, kv_overrides=args.override)
    if 'use_topology' not in cfg:
        cfg['use_topology'] = True
    query_groups_json = cfg.get('query_groups_json', None)
    if query_groups_json:
        qpath = Path(query_groups_json)
        if not qpath.is_absolute():
            qpath = (Path(__file__).parent / qpath).resolve()
        with qpath.open('r', encoding='utf-8') as f:
            qcfg = json.load(f)
        cfg['query_groups'] = qcfg.get('query_groups', [])
    if 'query_groups' not in cfg:
        cfg['query_groups'] = []
    return cfg

def build_preprocess(cfg):
    preprocess_cfg = cfg.get('image_preprocess', {})
    return transforms.Compose([
        transforms.Resize((int(preprocess_cfg.get('resize', 224)), int(preprocess_cfg.get('resize', 224)))),
        transforms.ToTensor(),
        transforms.Normalize(mean=preprocess_cfg.get('mean', [0.5, 0.5, 0.5]), std=preprocess_cfg.get('std', [0.5, 0.5, 0.5]))
    ])

# ============================================================
# 工具函数
# ============================================================
def normalize_path(path):
    """规范化路径用于比较"""
    try:
        abs_path = os.path.abspath(path)
        return os.path.normpath(abs_path).lower()
    except Exception as e:
        print(f"⚠️  Warning: Failed to normalize path {path}: {e}")
        return os.path.normpath(path).lower()


def clean_query_groups(query_groups_raw):
    """
    清理和验证查询组配置

    返回:
        cleaned_groups: 有效的查询组列表
        skipped_groups: 跳过的查询组信息
    """
    cleaned_groups = []
    skipped_groups = []

    for idx, group in enumerate(query_groups_raw):
        skip_reason = None
        group_name = group.get('name', f'group_{idx}')
        query_img = group.get('query_image')

        # 检查查询图像是否存在
        if not query_img or not os.path.exists(query_img):
            skip_reason = f"Query image not found: {query_img}"
            skipped_groups.append({
                'index': idx,
                'name': group_name,
                'reason': skip_reason
            })
            continue

        # 检查相关文档
        same_ginsengs = group.get('same_ginsengs', [])
        valid_ginsengs = []
        for path in same_ginsengs:
            if os.path.exists(path):
                if os.path.normpath(path).lower() != os.path.normpath(query_img).lower():
                    valid_ginsengs.append(path)

        if len(valid_ginsengs) == 0:
            skip_reason = f"No valid relevant documents (had {len(same_ginsengs)}, all missing or invalid)"
            skipped_groups.append({
                'index': idx,
                'name': group_name,
                'reason': skip_reason
            })
            continue

        # 添加有效的查询组
        cleaned_group = {
            'name': group_name,
            'query_image': query_img,
            'same_ginsengs': valid_ginsengs,
        }
        cleaned_groups.append(cleaned_group)

    return cleaned_groups, skipped_groups


def print_cleaning_report(skipped_groups):
    """打印清理报告"""
    if not skipped_groups:
        print("✅ All query groups are valid!")
        return

    print("\n" + "=" * 100)
    print(f"⚠️  CLEANING REPORT: {len(skipped_groups)} query group(s) will be skipped")
    print("=" * 100)
    for skip_info in skipped_groups:
        print(f"\n  ❌ [{skip_info['index']:3d}] {skip_info['name']}")
        print(f"     Reason: {skip_info['reason']}")
    print("\n" + "=" * 100 + "\n")


# ============================================================
# 特征提取
# ============================================================
def get_fused_features(model, img_path, device):
    """Extract features according to config['feature_type'] (visual/topo/both)."""
    img = Image.open(img_path).convert("L").convert("RGB")
    img_tensor = image_preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feature_type = str(config.get('feature_type', 'both')).lower()
        if feature_type == 'visual':
            visual_feat = model.extract_features(
                img_tensor,
                use_query_encoder=True,
                feature_type='visual'
            )
            return visual_feat.squeeze()

        if feature_type == 'topo':
            if not model.use_topology:
                raise ValueError("Topology branch is disabled, cannot extract topo-only features.")
            topo_feat = model.extract_features(
                img_tensor,
                use_query_encoder=True,
                feature_type='topo'
            )
            return topo_feat.squeeze()

        visual_feat = model.extract_features(
            img_tensor,
            use_query_encoder=True,
            feature_type='visual'
        )
        if not model.use_topology:
            return visual_feat.squeeze()
        topo_feat = model.extract_features(
            img_tensor,
            use_query_encoder=True,
            feature_type='topo'
        )
        # Weighted fusion to avoid topology branch dominating retrieval.
        alpha = float(config.get('fusion_alpha', 0.15))
        alpha = max(0.0, min(1.0, alpha))
        visual_feat = F.normalize(visual_feat, dim=1)
        topo_feat = F.normalize(topo_feat, dim=1)
        fused_feat = torch.cat([(1.0 - alpha) * visual_feat, alpha * topo_feat], dim=1)
        fused_feat = F.normalize(fused_feat, dim=1)
        return fused_feat.squeeze()

# ============================================================
# 检索函数
# ============================================================
def retrieve_topk(query_img_path, model, feats_tensor, feats_paths, device, k=10):
    """
    检索Top-K最相似图像,并过滤掉查询图像本身

    返回:
        results: [{'path': str, 'score': float, 'rank': int}, ...] 列表
    """
    print(f"🔍 Retrieving similar images for: {Path(query_img_path).name}")

    # 提取查询特征
    query_feat = get_fused_features(model, query_img_path, device)

    if feats_tensor.device != device:
        print(f"⚠️  WARNING: feats_tensor on {feats_tensor.device}, moving to {device}")
        feats_tensor = feats_tensor.to(device)

    # 归一化特征
    feats_tensor_norm = F.normalize(feats_tensor, dim=1)
    query_feat_norm = F.normalize(query_feat.unsqueeze(0), dim=1).squeeze(0)

    # 计算相似度
    sims = torch.mv(feats_tensor_norm, query_feat_norm)

    # 获取Top-K (多取几个以防过滤后不够)
    topk_vals, topk_indices = torch.topk(sims, min(k + 5, len(sims)))

    # 过滤掉查询图像本身
    query_img_norm = normalize_path(query_img_path)
    results = []
    for j, i in enumerate(topk_indices):
        result_path = feats_paths[i]
        result_path_norm = normalize_path(result_path)

        if result_path_norm == query_img_norm:
            print(f"  ⚠️  Filtered out query image itself at position {j + 1}")
            continue

        results.append({
            'path': result_path,
            'score': float(topk_vals[j].cpu()),
            'rank': len(results) + 1
        })

        if len(results) >= k:
            break

    return results


# ============================================================
# 评估类
# ============================================================
class RetrievalEvaluator:
    """检索评估器,计算各种评估指标"""

    def __init__(self, results, relevant_paths, query_image_path=None, k_values=[1, 5, 10, 20]):
        """
        Args:
            results: 检索结果列表 [{'path': str, 'score': float, 'rank': int}, ...]
            relevant_paths: 相关文档路径列表
            query_image_path: 查询图像路径(用于过滤)
            k_values: 计算Recall的K值列表
        """
        self.results = results
        self.relevant_paths = set(normalize_path(p) for p in relevant_paths)
        self.query_image_path = query_image_path
        self.k_values = k_values
        self.metrics = {}

        # 从相关文档中移除查询图像本身
        if query_image_path is not None:
            query_norm = normalize_path(query_image_path)
            if query_norm in self.relevant_paths:
                self.relevant_paths.discard(query_norm)

        self._calculate_metrics()

    def _calculate_metrics(self):
        """计算所有评估指标"""
        self.metrics['mrr'] = self._calculate_mrr()
        self.metrics['map'] = self._calculate_map()
        for k in self.k_values:
            self.metrics[f'recall@{k}'] = self._calculate_recall_at_k(k)

    def _calculate_mrr(self):
        """Mean Reciprocal Rank: 第一个相关文档的倒数排名"""
        for result in self.results:
            if self._is_relevant(result['path']):
                return 1.0 / result['rank']
        return 0.0

    def _calculate_map(self):
        """Mean Average Precision"""
        if len(self.relevant_paths) == 0:
            return 0.0

        sum_precisions = 0.0
        num_relevant_found = 0

        for result in self.results:
            if self._is_relevant(result['path']):
                num_relevant_found += 1
                precision_at_k = num_relevant_found / result['rank']
                sum_precisions += precision_at_k

        return sum_precisions / len(self.relevant_paths)

    def _calculate_recall_at_k(self, k):
        """Recall@K: Top-K中找到的相关文档比例"""
        if len(self.relevant_paths) == 0:
            return 0.0

        relevant_in_topk = sum(1 for r in self.results[:k] if self._is_relevant(r['path']))
        return relevant_in_topk / len(self.relevant_paths)

    def _is_relevant(self, result_path):
        """判断结果是否相关"""
        return normalize_path(result_path) in self.relevant_paths

    def get_relevant_ranks(self):
        """获取所有相关文档的排名"""
        return sorted([r['rank'] for r in self.results if self._is_relevant(r['path'])])

    def print_report(self):
        """打印评估报告"""
        print("\n" + "-" * 100)
        print(f"  MRR:          {self.metrics['mrr']:.4f}")
        print(f"  MAP:          {self.metrics['map']:.4f}")
        print(f"  RECALL@1:     {self.metrics['recall@1']:.4f}")
        print(f"  RECALL@5:     {self.metrics['recall@5']:.4f}")
        print(f"  RECALL@10:    {self.metrics['recall@10']:.4f}")
        print(f"  RECALL@20:    {self.metrics['recall@20']:.4f}")
        print(f"  Found ranks:  {self.get_relevant_ranks()}")
        print("-" * 100)

    def get_metrics_dict(self):
        """返回指标字典"""
        return self.metrics


class BatchEvaluator:
    """批量评估器,汇总所有查询的评估结果"""

    def __init__(self, k_values=[1, 5, 10, 20]):
        self.k_values = k_values
        self.results = []

    def add_result(self, query_name, evaluator):
        """添加单个查询的评估结果"""
        self.results.append({
            'query_name': query_name,
            'metrics': evaluator.get_metrics_dict(),
            'num_relevant': len(evaluator.relevant_paths),
            'found': len(evaluator.get_relevant_ranks()),
        })

    def print_summary(self):
        """打印汇总统计"""
        if not self.results:
            print("❌ No results to summarize")
            return {}

        print("\n" + "=" * 100)
        print("📊 BATCH EVALUATION SUMMARY")
        print("=" * 100)
        print(f"Total Queries: {len(self.results)}\n")

        print(f"{'Query Name':<20} {'MRR':<10} {'MAP':<10} {'R@1':<10} {'R@5':<10} {'R@10':<10} {'R@20':<10} {'Found':<10}")
        print("-" * 100)

        avg_metrics = {
            'mrr': 0.0,
            'map': 0.0,
            'recall@1': 0.0,
            'recall@5': 0.0,
            'recall@10': 0.0,
            'recall@20': 0.0,
        }

        for result in self.results:
            m = result['metrics']
            found_str = f"{result['found']}/{result['num_relevant']}"
            print(
                f"{result['query_name']:<20} {m['mrr']:<10.4f} {m['map']:<10.4f} "
                f"{m['recall@1']:<10.4f} {m['recall@5']:<10.4f} {m['recall@10']:<10.4f} "
                f"{m.get('recall@20', 0.0):<10.4f} {found_str:<10}"
            )

            avg_metrics['mrr'] += m['mrr']
            avg_metrics['map'] += m['map']
            avg_metrics['recall@1'] += m['recall@1']
            avg_metrics['recall@5'] += m['recall@5']
            avg_metrics['recall@10'] += m['recall@10']
            avg_metrics['recall@20'] += m.get('recall@20', 0.0)

        n = len(self.results)
        print("-" * 100)
        if n > 0:
            print(
                f"{'AVERAGE':<20} {avg_metrics['mrr'] / n:<10.4f} {avg_metrics['map'] / n:<10.4f} "
                f"{avg_metrics['recall@1'] / n:<10.4f} {avg_metrics['recall@5'] / n:<10.4f} "
                f"{avg_metrics['recall@10'] / n:<10.4f} {avg_metrics['recall@20'] / n:<10.4f}"
            )
            avg_metrics = {k: v / n for k, v in avg_metrics.items()}

        print("=" * 100 + "\n")
        return avg_metrics

    def save_summary(self, output_folder):
        """保存评估报告到文件"""
        os.makedirs(output_folder, exist_ok=True)
        summary_file = os.path.join(output_folder, 'batch_summary.txt')

        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("📊 BATCH EVALUATION SUMMARY (Multi-Branch Topology Network)\n")
            f.write("=" * 100 + "\n")

            if not self.results:
                f.write("❌ NO RESULTS AVAILABLE\n")
                f.write("All queries failed during retrieval process.\n")
                f.write("=" * 100 + "\n")
                print(f"📄 Summary saved to: {summary_file}")
                return

            f.write(f"Total Queries: {len(self.results)}\n\n")
            f.write(
                f"{'Query Name':<20} {'MRR':<10} {'MAP':<10} "
                f"{'Recall@1':<12} {'Recall@5':<12} {'Recall@10':<12} "
                f"{'Recall@20':<12} {'Found':<12}\n"
            )
            f.write("-" * 100 + "\n")

            avg_metrics = {
                'mrr': 0.0,
                'map': 0.0,
                'recall@1': 0.0,
                'recall@5': 0.0,
                'recall@10': 0.0,
                'recall@20': 0.0,
            }

            for result in self.results:
                metrics = result['metrics']
                found_str = f"{result['found']}/{result['num_relevant']}"
                f.write(
                    f"{result['query_name']:<20} "
                    f"{metrics['mrr']:<10.4f} "
                    f"{metrics['map']:<10.4f} "
                    f"{metrics['recall@1']:<12.4f} "
                    f"{metrics['recall@5']:<12.4f} "
                    f"{metrics['recall@10']:<12.4f} "
                    f"{metrics.get('recall@20', 0.0):<12.4f} "
                    f"{found_str:<12}\n"
                )

                avg_metrics['mrr'] += metrics['mrr']
                avg_metrics['map'] += metrics['map']
                avg_metrics['recall@1'] += metrics['recall@1']
                avg_metrics['recall@5'] += metrics['recall@5']
                avg_metrics['recall@10'] += metrics['recall@10']
                avg_metrics['recall@20'] += metrics.get('recall@20', 0.0)

            n = len(self.results)
            f.write("-" * 100 + "\n")
            if n > 0:
                f.write(
                    f"{'AVERAGE':<20} "
                    f"{avg_metrics['mrr'] / n:<10.4f} "
                    f"{avg_metrics['map'] / n:<10.4f} "
                    f"{avg_metrics['recall@1'] / n:<12.4f} "
                    f"{avg_metrics['recall@5'] / n:<12.4f} "
                    f"{avg_metrics['recall@10'] / n:<12.4f} "
                    f"{avg_metrics['recall@20'] / n:<12.4f}\n"
                )
            f.write("=" * 100 + "\n")

        print(f"📄 Summary saved to: {summary_file}")


# ============================================================
# 可视化
# ============================================================
def visualize_results(query_img_path, results, same_ginseng_paths, output_folder,
                      ncols=5, nrows=2, img_per_colsize=6):
    """可视化检索结果并支持分页浏览"""
    os.makedirs(output_folder, exist_ok=True)

    # 保存查询图像
    query_img = Image.open(query_img_path).convert("L").convert("RGB")
    query_save_path = os.path.join(output_folder, 'query.jpg')
    query_img.save(query_save_path)
    print(f"📸 Query image saved: {query_save_path}")

    # 规范化相关路径
    relevant_paths_norm = set(normalize_path(path) for path in same_ginseng_paths)
    query_norm = normalize_path(query_img_path)
    if query_norm in relevant_paths_norm:
        print(f"⚠️  Warning: Query image found in relevant paths, removing it")
        relevant_paths_norm.discard(query_norm)

    def is_relevant(result_path):
        return normalize_path(result_path) in relevant_paths_norm

    # 分页信息
    num_results = len(results)
    imgs_per_page = ncols * nrows
    num_pages = math.ceil(num_results / imgs_per_page)

    if num_pages == 0:
        print("⚠️  No results to visualize")
        return

    def show_page(page_idx):
        """显示指定页"""
        plt.clf()
        start = page_idx * imgs_per_page
        end = min(start + imgs_per_page, num_results)

        for idx_in_page, result_idx in enumerate(range(start, end)):
            result = results[result_idx]
            img_path = result['path']
            score = result['score']

            try:
                img = Image.open(img_path).convert("L").convert("RGB")
            except Exception as e:
                print(f"⚠️  Failed to load image {img_path}: {e}")
                img = Image.new('RGB', (224, 224), color='gray')

            relevance_mark = "✅ RELEVANT" if is_relevant(img_path) else ""

            plt.subplot(nrows, ncols, idx_in_page + 1)
            plt.imshow(img)
            title_text = (f"{os.path.basename(img_path)}\n"
                          f"Cos: {score:.4f}\n"
                          f"{relevance_mark}")
            title_color = 'green' if is_relevant(img_path) else 'black'
            plt.title(title_text, fontsize=12, color=title_color, fontweight='bold')
            plt.axis('off')

        for idx_in_page in range(end - start, imgs_per_page):
            plt.subplot(nrows, ncols, idx_in_page + 1)
            plt.axis('off')

        relevant_count = sum(1 for r in results if is_relevant(r['path']))
        suptitle_text = (f"检索结果(第 {page_idx + 1} 页/共 {num_pages} 页) | "
                         f"相关样本: {relevant_count}/{len(relevant_paths_norm)}")
        plt.suptitle(suptitle_text, fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.draw()

    # 创建图窗
    fig = plt.figure(figsize=(ncols * img_per_colsize, nrows * img_per_colsize + 1))
    current_page = [0]

    # 添加翻页按钮
    axprev = plt.axes([0.35, 0.01, 0.1, 0.06])
    axnext = plt.axes([0.55, 0.01, 0.1, 0.06])
    bprev = Button(axprev, "← 上一页")
    bnext = Button(axnext, "下一页 →")

    def prev(event):
        if current_page[0] > 0:
            current_page[0] -= 1
            show_page(current_page[0])

    def next(event):
        if current_page[0] < num_pages - 1:
            current_page[0] += 1
            show_page(current_page[0])

    bprev.on_clicked(prev)
    bnext.on_clicked(next)

    def on_key(event):
        if event.key == "left":
            prev(None)
        elif event.key == "right":
            next(None)

    fig.canvas.mpl_connect('key_press_event', on_key)

    show_page(current_page[0])
    plt.show()


def save_results_to_file(query_group_name, query_img_path, results, same_ginseng_paths,
                         metrics, output_folder, query_tag=None):
    """保存检索结果到文件"""
    os.makedirs(output_folder, exist_ok=True)
    tag = f"_{query_tag}" if query_tag else ""
    result_file = os.path.join(output_folder, f'{query_group_name}{tag}_results.txt')

    relevant_paths_norm = set(normalize_path(path) for path in same_ginseng_paths)
    query_norm = normalize_path(query_img_path)
    if query_norm in relevant_paths_norm:
        relevant_paths_norm.discard(query_norm)

    def is_relevant(result_path):
        return normalize_path(result_path) in relevant_paths_norm

    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("=" * 120 + "\n")
        f.write(f"🔍 Image Retrieval Results and Evaluation Metrics (Multi-Branch Topology Network)\n")
        f.write("=" * 120 + "\n")
        f.write(f"Query Group: {query_group_name}\n")
        f.write(f"Query Image: {query_img_path}\n")
        f.write(f"Total Retrieved: {len(results)}\n")
        f.write(f"Total Relevant Documents Specified: {len(same_ginseng_paths)}\n")
        f.write("=" * 120 + "\n\n")

        f.write("📊 EVALUATION METRICS\n")
        f.write("-" * 120 + "\n")
        f.write(f"MRR (Mean Reciprocal Rank):    {metrics.get('mrr', 0):.4f}\n")
        f.write(f"MAP (Mean Average Precision):  {metrics.get('map', 0):.4f}\n")
        f.write(f"RECALL@1:                      {metrics.get('recall@1', 0):.4f}\n")
        f.write(f"RECALL@5:                      {metrics.get('recall@5', 0):.4f}\n")
        f.write(f"RECALL@10:                     {metrics.get('recall@10', 0):.4f}\n")
        f.write(f"RECALL@20:                     {metrics.get('recall@20', 0):.4f}\n")
        f.write("-" * 120 + "\n\n")

        f.write("📋 DETAILED RESULTS\n")
        f.write("-" * 120 + "\n")
        f.write(f"{'Rank':<6} {'Cosine Sim':<15} {'Relevance':<15} {'Image Path':<84}\n")
        f.write("-" * 120 + "\n")
        for result in results:
            relevance_mark = "✅ RELEVANT" if is_relevant(result['path']) else "❌ NOT RELEVANT"
            f.write(f"{result['rank']:<6} {result['score']:<15.4f} {relevance_mark:<15} {result['path']:<84}\n")

        f.write("\n" + "-" * 120 + "\n")
        f.write("✅ RELEVANT DOCUMENTS (Expected)\n")
        f.write("-" * 120 + "\n")
        for i, path in enumerate(same_ginseng_paths, 1):
            f.write(f"  {i}. {path}\n")

    print(f"📄 Results saved to: {result_file}")


# ============================================================
# 主函数
# ============================================================
def main():

    global config, image_preprocess
    config = load_runtime_config()
    image_preprocess = build_preprocess(config)
    device = torch.device("cuda" if config['use_gpu'] else "cpu")
    print("=" * 100)
    print(f"🔍 Batch Image Retrieval & Evaluation System (Multi-Branch Topology Network)")
    print("=" * 100)
    print(f"Device: {device}")
    print(f"Total Query Groups (raw): {len(config['query_groups'])}")
    print(f"Use topology: {config.get('use_topology', True)}")
    print(f"Feature type: {config.get('feature_type', 'both')}")
    print(f"Fusion alpha: {config.get('fusion_alpha', 0.15)}")
    print("=" * 100 + "\n")

    # 清理配置
    print("🧹 Cleaning configuration...")
    cleaned_groups, skipped_groups = clean_query_groups(config['query_groups'])
    print_cleaning_report(skipped_groups)
    print(f"✅ Valid Query Groups: {len(cleaned_groups)}")
    print(f"⏭️  Skipped Query Groups: {len(skipped_groups)}\n")

    if len(cleaned_groups) == 0:
        print("❌ No valid query groups after cleaning!")
        return

    # 加载模型
    print("🔧 Loading model...")
    if not os.path.exists(config['model_path']):
        print(f"❌ Model file not found: {config['model_path']}")
        return

    try:
        model = MoCoV3HybridTopo(
            feature_dim=config['feature_dim'],
            topo_dim=config['topo_dim'],
            K=config['K'],
            m=config['m'],
            T=config['T'],
            topo_weight=config['topo_weight'],
            use_topology=config.get('use_topology', True),
            use_legacy_branch=config['use_legacy_branch'],
            use_skeleton_branch=config['use_skeleton_branch'],
            use_edge_branch=config['use_edge_branch'],
            use_frequency_branch=config['use_frequency_branch'],
            device=device
        )
        checkpoint = torch.load(config['model_path'], map_location=device)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()
        feature_type = str(config.get('feature_type', 'both')).lower()
        if not model.use_topology and feature_type == 'topo':
            print("❌ 当前配置 use_topology=False，不能进行 topo-only 检索。")
            return
        if not model.use_topology and feature_type == 'both':
            print("⚠️  当前配置 use_topology=False，feature_type=both 将自动退化为 visual。")
        print(f"✅ Model loaded from {config['model_path']}\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 加载图库特征
    print("💾 Loading gallery features...")
    if not os.path.exists(config['feature_path']):
        print(f"❌ Feature file not found: {config['feature_path']}")
        return

    try:
        data = torch.load(config['feature_path'], map_location=device)

        if isinstance(data, dict):
            feats_tensor = data['features'].to(device)
            feats_paths = data['paths']
        else:
            feats_tensor = data.to(device)
            feats_paths = []

        print(f"✅ Loaded {len(feats_paths)} gallery features")
        print(f"   Feature tensor shape: {feats_tensor.shape}")
        print(f"   Feature tensor device: {feats_tensor.device}\n")
    except Exception as e:
        print(f"❌ Error loading features: {e}")
        return

    # 批量检索
    print("=" * 100)
    print("🚀 Starting batch retrieval...")
    print("=" * 100 + "\n")

    # 这里显式指定需要统计的K值,包含Recall@20
    batch_evaluator = BatchEvaluator(k_values=[1, 5, 10, 20])
    successful_queries = 0
    failed_queries = 0

    for group_idx, group in enumerate(cleaned_groups, 1):
        group_name = group['name']
        query_image = group['query_image']
        same_ginsengs = group['same_ginsengs']

        print(f"[{group_idx}/{len(cleaned_groups)}] Processing: {group_name}")
        print(f"  Query: {Path(query_image).name}")
        print(f"  Expected relevant docs: {len(same_ginsengs)}")

        try:
            # 执行检索
            results = retrieve_topk(
                query_image,
                model,
                feats_tensor,
                feats_paths,
                device,
                k=config['top_k']
            )

            if len(results) == 0:
                print(f"  ⚠️  No results retrieved")
                failed_queries += 1
                continue

            print(f"  ✅ Retrieved {len(results)} results")

        except Exception as e:
            print(f"  ❌ Error during retrieval: {e}")
            failed_queries += 1
            continue

        try:
            # 评估,同时计算Recall@1/5/10/20
            evaluator = RetrievalEvaluator(
                results=results,
                relevant_paths=same_ginsengs,
                query_image_path=query_image,
                k_values=[1, 5, 10, 20]
            )

            evaluator.print_report()
            batch_evaluator.add_result(group_name, evaluator)

            # 保存结果
            query_tag = Path(query_image).stem
            output_group_folder = os.path.join(config['output_folder'], f"{group_name}_{query_tag}")
            save_results_to_file(
                group_name,
                query_image,
                results,
                same_ginsengs,
                evaluator.get_metrics_dict(),
                output_group_folder,
                query_tag=query_tag
            )

            successful_queries += 1
        except Exception as e:
            print(f"  ❌ Error during evaluation: {e}")
            failed_queries += 1
            continue

        print()

    # 打印汇总报告
    print("\n" + "=" * 100)
    print("📊 BATCH PROCESSING SUMMARY")
    print("=" * 100)
    print(f"Total Queries Processed: {len(cleaned_groups)}")
    print(f"Successful: {successful_queries} ✅")
    print(f"Failed: {failed_queries} ❌")
    print("=" * 100 + "\n")

    avg_metrics = batch_evaluator.print_summary()
    batch_evaluator.save_summary(config['output_folder'])

    print(f"\n✅ Batch retrieval completed!")
    print(f"📂 All results saved to: {config['output_folder']}")

    return avg_metrics


if __name__ == '__main__':
    try:
        avg_metrics = main()
        print("\n🎉 Batch retrieval and evaluation completed successfully!")
    except KeyboardInterrupt:
        print("\n\n⚠️  Program interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
