"""
改进多支路拓扑网络的特征提取脚本

功能:
1. 批量提取图库特征(视觉+拓扑融合特征)
2. 支持递归扫描目录
3. 保存特征文件供检索使用

基于: hybrid/model.py (MoCoV3HybridTopo)
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from datetime import datetime

# 导入模型
from model import MoCoV3HybridTopo
from config import load_config

# ============================================================
# 配置参数
IMAGE_SUFFIXES_DEFAULT = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

# Global runtime config (filled in main)
config = {}
image_preprocess = None

def parse_args():
    parser = argparse.ArgumentParser(description='Extract gallery features')
    parser.add_argument('--config', type=str, default=None, help='Optional config JSON path')
    parser.add_argument('--override', action='append', default=[], help='Override key=value (repeatable)')
    return parser.parse_args()

def load_runtime_config():
    args = parse_args()
    cfg = load_config('extraction', config_path=args.config, kv_overrides=args.override)
    cfg['img_suffix'] = tuple(cfg.get('img_suffix', IMAGE_SUFFIXES_DEFAULT))
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
def get_all_img_paths(root_dir, suffix):
    """递归获取目录下所有指定后缀的图像路径"""
    file_list = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith(suffix):
                file_list.append(os.path.join(dirpath, f))
    return file_list


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
            return visual_feat.cpu().squeeze()

        if feature_type == 'topo':
            topo_feat = model.extract_features(
                img_tensor,
                use_query_encoder=True,
                feature_type='topo'
            )
            return topo_feat.cpu().squeeze()

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
        return fused_feat.cpu().squeeze()

# ============================================================
# 主函数
# ============================================================
def main():
    """主函数:批量提取图库特征"""

    global config, image_preprocess
    config = load_runtime_config()
    image_preprocess = build_preprocess(config)

    # 初始化日志
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'=' * 70}")
    print(f"[{timestamp}] 启动多支路拓扑网络特征提取流程")
    print(f"{'=' * 70}")

    # 设备和模型初始化
    device = torch.device("cuda" if config['use_gpu'] else "cpu")
    print(f"\n✅ 计算设备: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 加载模型
    print(f"\n📦 加载多支路拓扑模型...")
    print(f"   支路配置:")
    print(f"     - 旧版拓扑支路: {config['use_legacy_branch']}")
    print(f"     - 骨架化支路: {config['use_skeleton_branch']}")
    print(f"     - 边缘检测支路: {config['use_edge_branch']}")
    print(f"     - 频域支路: {config['use_frequency_branch']}")

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
        use_frequency_branch=config['use_frequency_branch'],
        device=device
    )

    try:
        model.load_state_dict(torch.load(config['model_path'], map_location=device))
        print(f"✅ 模型加载成功: {config['model_path']}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print(f"   请检查模型路径和参数配置是否与训练时一致")
        return

    model.to(device)
    model.eval()

    # 获取所有图像路径
    print(f"\n🔍 扫描图库目录: {config['image_dir']}")
    img_paths = get_all_img_paths(config['image_dir'], config['img_suffix'])
    print(f"✅ 共发现 {len(img_paths)} 张图片")

    if len(img_paths) == 0:
        print(f"⚠️  没有找到任何图片,程序退出")
        return

    # 批量提取特征
    print(f"\n📊 开始提取融合特征...")
    print(f"   视觉特征维度: {config['feature_dim']}")
    print(f"   拓扑特征维度: {config['topo_dim']}")
    print(f"   融合后维度: {config['feature_dim'] + config['topo_dim']}\n")

    features = []
    valid_paths = []

    for i, img_path in enumerate(img_paths):
        try:
            feat = get_fused_features(model, img_path, device)
            features.append(feat)
            valid_paths.append(img_path)

            # 每20张或最后一张时打印进度
            if (i + 1) % 20 == 0 or i == len(img_paths) - 1:
                progress_pct = (i + 1) / len(img_paths) * 100
                print(f"   [{i + 1:>4d}/{len(img_paths)}] ({progress_pct:>5.1f}%) ✅ {os.path.basename(img_path)}")

        except Exception as e:
            print(f"   [{i + 1:>4d}/{len(img_paths)}] ❌ 跳过损坏图片: {os.path.basename(img_path)} -> {str(e)[:50]}")

    # 保存特征
    print(f"\n💾 保存特征...")
    features_tensor = torch.stack(features)

    os.makedirs(os.path.dirname(config['output_feats']), exist_ok=True)
    torch.save(
        {
            'features': features_tensor,
            'paths': valid_paths,
            'feature_dim': config['feature_dim'],
            'topo_dim': config['topo_dim'],
            'total_dim': config['feature_dim'] + config['topo_dim'],
            'num_images': len(valid_paths),
            'model_config': {
                'use_legacy_branch': config['use_legacy_branch'],
                'use_skeleton_branch': config['use_skeleton_branch'],
                'use_edge_branch': config['use_edge_branch'],
                'use_frequency_branch': config['use_frequency_branch'],
            }
        },
        config['output_feats']
    )

    print(f"✅ 图库特征保存成功")
    print(f"   路径: {config['output_feats']}")
    print(f"   有效图片数: {len(valid_paths)}")
    print(f"   特征张量形状: {features_tensor.shape}")
    print(f"\n{'=' * 70}\n")


if __name__ == '__main__':
    main()
