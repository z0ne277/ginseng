\
\
\
\
\
\
\
\
\


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


matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


from model import MoCoV3HybridTopo
from config import load_config




config = load_config("retrieval")

preprocess_cfg = config.get('image_preprocess', {})
image_preprocess = transforms.Compose([
    transforms.Resize((int(preprocess_cfg.get('resize', 224)), int(preprocess_cfg.get('resize', 224)))),
    transforms.ToTensor(),
    transforms.Normalize(mean=preprocess_cfg.get('mean', [0.5, 0.5, 0.5]), std=preprocess_cfg.get('std', [0.5, 0.5, 0.5]))
])




def get_fused_features(model, img_path, device):

    img = Image.open(img_path).convert("L").convert("RGB")
    img_tensor = image_preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():

        visual_feat = model.extract_features(
            img_tensor,
            use_query_encoder=True,
            feature_type='visual'
        )


        topo_feat = model.extract_features(
            img_tensor,
            use_query_encoder=True,
            feature_type='topo'
        )


        fused_feat = torch.cat([visual_feat, topo_feat], dim=1)


        fused_feat = F.normalize(fused_feat, dim=1)

    return fused_feat.squeeze()





def retrieve_topk(query_img_path, model, feats_tensor, feats_paths, device, k=10):
\
\
\
\
\
\
\
\
\
\
\
\
\


    query_feat = get_fused_features(model, query_img_path, device)
    query_feat = query_feat.to(device)


    sims = torch.mv(feats_tensor, query_feat)


    topk_vals, topk_indices = torch.topk(sims, k)


    results = [
        (feats_paths[int(topk_indices[j])], float(topk_vals[j]))
        for j in range(k)
    ]

    return results





def visualize_and_save_results(query_img_path, results, output_folder,
                               ncols=5, nrows=2, img_per_colsize=6):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\


    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)


    query_img = Image.open(query_img_path).convert("L").convert("RGB")
    query_save_path = os.path.join(output_folder, 'query.jpg')
    query_img.save(query_save_path)
    print(f"\n📸 查询图像已保存: {query_save_path}")


    num_results = len(results)
    imgs_per_page = ncols * nrows
    num_pages = math.ceil(num_results / imgs_per_page)

    print(f"\n📊 检索结果统计:")
    print(f"   找到 {num_results} 张相似图像")
    print(f"   分 {num_pages} 页显示(每页 {imgs_per_page} 张)\n")

    def show_page(page_idx):
\
\
\
\
\

        plt.clf()
        start = page_idx * imgs_per_page
        end = min(start + imgs_per_page, num_results)


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


    fig = plt.figure(figsize=(ncols * img_per_colsize, nrows * img_per_colsize + 1))
    current_page = [0]


    axprev = plt.axes([0.35, 0.01, 0.1, 0.06])
    axnext = plt.axes([0.55, 0.01, 0.1, 0.06])
    bprev = Button(axprev, "← 上一页")
    bnext = Button(axnext, "下一页 →")

    def prev_page(event):

        if current_page[0] > 0:
            current_page[0] -= 1
            show_page(current_page[0])

    def next_page(event):

        if current_page[0] < num_pages - 1:
            current_page[0] += 1
            show_page(current_page[0])

    bprev.on_clicked(prev_page)
    bnext.on_clicked(next_page)


    def on_key(event):

        if event.key == "left":
            prev_page(None)
        elif event.key == "right":
            next_page(None)

    fig.canvas.mpl_connect('key_press_event', on_key)


    show_page(current_page[0])
    plt.show()





def main():
    print(f"\n{'=' * 70}")
    print(f"多支路拓扑网络图像检索系统")
    print(f"{'=' * 70}")

    device = torch.device("cuda" if config['use_gpu'] else "cpu")
    print(f"\n✅ 计算设备: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


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


    print(f"\n⏳ 正在加载库特征...")
    features_data = torch.load(config['feature_path'], map_location=device)

    if isinstance(features_data, dict):
        feats_tensor = features_data['features'].to(device)
        feats_paths = features_data['paths']
        print(f"✅ 库特征已加载")
        print(f"   特征维度: {feats_tensor.shape}")
        print(f"   库规模: {len(feats_paths)} 张图像")


        if 'model_config' in features_data:
            print(f"   支路配置:")
            for key, value in features_data['model_config'].items():
                print(f"     - {key}: {value}")
    else:
        feats_tensor = features_data.to(device)
        feats_paths = []
        print(f"✅ 库特征已加载(旧格式)")


    if not os.path.exists(config['query_image']):
        print(f"\n❌ 错误: 查询图像不存在 -> {config['query_image']}")
        return


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
