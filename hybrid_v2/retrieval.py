"""
改进多支路拓扑网络的单张图像检索脚本

功能:
1. 加载预提取的图库特征
2. 对单张查询图像进行检索
3. 可视化Top-K检索结果并支持分页浏览

基于: hybrid/model.py (MoCoV3HybridTopo)
"""

import os
import sys
import shutil
import torch
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib
from PIL import Image
from torchvision import transforms

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 导入模型
from model import MoCoV3HybridTopo
from config import load_config

# ============================================================
# 配置参数
# ============================================================
config = load_config("retrieval")

preprocess_cfg = config.get('image_preprocess', {})
image_preprocess = transforms.Compose([
    transforms.Resize((int(preprocess_cfg.get('resize', 224)), int(preprocess_cfg.get('resize', 224)))),
    transforms.ToTensor(),
    transforms.Normalize(mean=preprocess_cfg.get('mean', [0.5, 0.5, 0.5]), std=preprocess_cfg.get('std', [0.5, 0.5, 0.5]))
])

# ============================================================
# 特征提取
# ============================================================
def get_fused_features(model, img_path, device):
    """提取融合特征(视觉 + 拓扑)"""
    img = Image.open(img_path).convert("L").convert("RGB")
    img_tensor = image_preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        # 提取视觉特征
        visual_feat = model.extract_features(
            img_tensor,
            use_query_encoder=True,
            feature_type='visual'
        )

        # 提取拓扑特征
        topo_feat = model.extract_features(
            img_tensor,
            use_query_encoder=True,
            feature_type='topo'
        )

        # 拼接特征
        fused_feat = torch.cat([visual_feat, topo_feat], dim=1)

        # L2归一化
        fused_feat = F.normalize(fused_feat, dim=1)

    return fused_feat.squeeze()


# ============================================================
# 检索函数
# ============================================================
def retrieve_topk(query_img_path, model, feats_tensor, feats_paths, device, k=10):
    """
    检索Top-K最相似图像

    Args:
        query_img_path: 查询图像路径
        model: 模型实例
        feats_tensor: 图库特征张量 [N, D]
        feats_paths: 图库图像路径列表
        device: 计算设备
        k: 返回Top-K结果

    Returns:
        results: [(图像路径, 相似度), ...] 列表
    """
    # 提取查询特征
    query_feat = get_fused_features(model, query_img_path, device)
    query_feat = query_feat.to(device)

    # 计算相似度(余弦相似度)
    sims = torch.mv(feats_tensor, query_feat)

    # 获取Top-K
    topk_vals, topk_indices = torch.topk(sims, k)

    # 构建结果
    results = [
        (feats_paths[int(topk_indices[j])], float(topk_vals[j]))
        for j in range(k)
    ]

    return results


# ============================================================
# 可视化
# ============================================================
def visualize_and_save_results(query_img_path, results, output_folder,
                               ncols=5, nrows=2, img_per_colsize=6):
    """
    可视化检索结果并支持分页浏览

    功能:
    1. 保存查询图像到输出文件夹
    2. 显示Top-K结果,支持分页
    3. 支持鼠标按钮翻页和键盘左右键翻页

    Args:
        query_img_path: 查询图像路径
        results: [(图像路径, 相似度), ...] 列表,已按相似度排序
        output_folder: 输出结果文件夹
        ncols: 每行显示的图像数(列数)
        nrows: 每页显示的行数
        img_per_colsize: 每个图像在屏幕上的显示尺寸(英寸)
    """
    # 创建输出目录
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # 保存查询图像
    query_img = Image.open(query_img_path).convert("L").convert("RGB")
    query_save_path = os.path.join(output_folder, 'query.jpg')
    query_img.save(query_save_path)
    print(f"\n📸 查询图像已保存: {query_save_path}")

    # 计算分页信息
    num_results = len(results)
    imgs_per_page = ncols * nrows
    num_pages = math.ceil(num_results / imgs_per_page)

    print(f"\n📊 检索结果统计:")
    print(f"   找到 {num_results} 张相似图像")
    print(f"   分 {num_pages} 页显示(每页 {imgs_per_page} 张)\n")

    def show_page(page_idx):
        """
        显示指定页码的检索结果

        Args:
            page_idx: 页码(从0开始)
        """
        plt.clf()
        start = page_idx * imgs_per_page
        end = min(start + imgs_per_page, num_results)

        # 显示当前页的结果图像
        for idx_in_page, idx in enumerate(range(start, end)):
            img_path, score = results[idx]
            img = Image.open(img_path).convert("L").convert("RGB")

            plt.subplot(nrows, ncols, idx_in_page + 1)
            plt.imshow(img)
            plt.title(
                f"{os.path.basename(img_path)}\n余弦相似度: {score:.4f}",
                fontsize=12,
                fontweight='bold'
            )
            plt.axis('off')

        # 隐去未填充的子图
        for idx_in_page in range(end - start, imgs_per_page):
            plt.subplot(nrows, ncols, idx_in_page + 1)
            plt.axis('off')

        plt.suptitle(
            f"多支路拓扑网络检索结果(第 {page_idx + 1} 页 / 共 {num_pages} 页)",
            fontsize=16,
            fontweight='bold'
        )
        plt.tight_layout(rect=[0, 0.08, 1, 0.96])
        plt.draw()

    # 创建图窗并初始化第一页
    fig = plt.figure(figsize=(ncols * img_per_colsize, nrows * img_per_colsize + 1))
    current_page = [0]  # 用列表包装以在闭包中修改

    # 添加翻页按钮
    axprev = plt.axes([0.35, 0.01, 0.1, 0.06])
    axnext = plt.axes([0.55, 0.01, 0.1, 0.06])
    bprev = Button(axprev, "← 上一页")
    bnext = Button(axnext, "下一页 →")

    def prev_page(event):
        """上一页按钮回调"""
        if current_page[0] > 0:
            current_page[0] -= 1
            show_page(current_page[0])

    def next_page(event):
        """下一页按钮回调"""
        if current_page[0] < num_pages - 1:
            current_page[0] += 1
            show_page(current_page[0])

    bprev.on_clicked(prev_page)
    bnext.on_clicked(next_page)

    # 支持键盘左右键翻页
    def on_key(event):
        """键盘事件处理"""
        if event.key == "left":
            prev_page(None)
        elif event.key == "right":
            next_page(None)

    fig.canvas.mpl_connect('key_press_event', on_key)

    # 显示第一页
    show_page(current_page[0])
    plt.show()


# ============================================================
# 主函数
# ============================================================
def main():
    print(f"\n{'=' * 70}")
    print(f"多支路拓扑网络图像检索系统")
    print(f"{'=' * 70}")

    device = torch.device("cuda" if config['use_gpu'] else "cpu")
    print(f"\n✅ 计算设备: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 加载模型
    print(f"\n⏳ 正在加载模型...")
    model = MoCoV3HybridTopo(
        feature_dim=config['feature_dim'],
        topo_dim=config['topo_dim'],
        K=config['K'],
        m=config['m'],
        T=config['T'],
        topo_weight=config['topo_weight'],
        use_legacy_branch=config['use_legacy_branch'],
        use_skeleton_branch=config['use_skeleton_branch'],
        use_edge_branch=config['use_edge_branch'],
        use_frequency_branch=config['use_frequency_branch']
    )

    checkpoint = torch.load(config['model_path'], map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    print(f"✅ 模型已加载: {config['model_path']}")

    # 加载图库特征
    print(f"\n⏳ 正在加载库特征...")
    features_data = torch.load(config['feature_path'], map_location=device)

    if isinstance(features_data, dict):
        feats_tensor = features_data['features'].to(device)
        feats_paths = features_data['paths']
        print(f"✅ 库特征已加载")
        print(f"   特征维度: {feats_tensor.shape}")
        print(f"   库规模: {len(feats_paths)} 张图像")

        # 打印模型配置信息
        if 'model_config' in features_data:
            print(f"   支路配置:")
            for key, value in features_data['model_config'].items():
                print(f"     - {key}: {value}")
    else:
        feats_tensor = features_data.to(device)
        feats_paths = []
        print(f"✅ 库特征已加载(旧格式)")

    # 检查查询图像
    if not os.path.exists(config['query_image']):
        print(f"\n❌ 错误: 查询图像不存在 -> {config['query_image']}")
        return

    # 执行检索
    print(f"\n⏳ 正在执行检索...")
    results = retrieve_topk(
        query_img_path=config['query_image'],
        model=model,
        feats_tensor=feats_tensor,
        feats_paths=feats_paths,
        device=device,
        k=config['top_k']
    )

    print(f"\n✅ 检索完成!")
    print(f"\n{'=' * 70}")
    print(f"Top-{config['top_k']} 检索结果")
    print(f"{'=' * 70}\n")

    for rank, (img_path, similarity) in enumerate(results, 1):
        print(f"排名 {rank:2d}: {img_path}")
        print(f"          余弦相似度: {similarity:.6f}")
        print()

    print(f"{'=' * 70}\n")

    # 可视化结果
    visualize_and_save_results(
        query_img_path=config['query_image'],
        results=results,
        output_folder=config['output_folder'],
        ncols=5,
        nrows=2,
        img_per_colsize=6
    )


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ 程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
