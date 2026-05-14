"""
改进的多拓扑支路特征提取网络

核心改进:
1. 多种拓扑提取方式组合:
   - 膨胀腐蚀支路 (Erosion/Dilation)
   - 骨架化支路 (Skeletonization)
   - 边缘检测支路 (Edge Detection)
   - 频域拓扑支路 (Frequency Domain)
2. 保持固定的形态学操作（不使用可学习腐蚀，避免退化）
3. 简化融合策略，避免过拟合
4. 支持灵活的支路组合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple, Dict, Optional, List
import math


# ============================================================
# 基础形态学操作（固定，不可学习）
# ============================================================
class MorphologicalErosion(nn.Module):
    """形态学腐蚀层：固定操作，保证拓扑特征的稳定性"""

    def __init__(self, kernel_size: int = 3, num_erosions: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_erosions = num_erosions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_eroded = x
        for _ in range(self.num_erosions):
            x_eroded = -F.max_pool2d(
                -x_eroded,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2
            )
        return x_eroded


class MorphologicalDilation(nn.Module):
    """形态学膨胀层"""

    def __init__(self, kernel_size: int = 3, num_dilations: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_dilations = num_dilations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dilated = x
        for _ in range(self.num_dilations):
            x_dilated = F.max_pool2d(
                x_dilated,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2
            )
        return x_dilated


class LegacyTopoBranch(nn.Module):
    """
    旧版拓扑支路：直接复刻 old 代码里的多尺度腐蚀特征提取流程
    """

    def __init__(
        self,
        input_channels: int = 2048,
        num_erosion_levels: int = 4,
        output_dim: int = 128
    ):
        super().__init__()
        self.num_erosion_levels = num_erosion_levels
        self.output_dim = output_dim

        self.erosion_layers = nn.ModuleList([
            MorphologicalErosion(kernel_size=3, num_erosions=i)
            for i in range(num_erosion_levels)
        ])

        self.topo_encoders = nn.ModuleList()
        for _ in range(num_erosion_levels):
            encoder = nn.Sequential(
                nn.Conv2d(input_channels, 512, kernel_size=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Conv2d(512, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(256, output_dim)
            )
            self.topo_encoders.append(encoder)

        self.topo_fusion = nn.Sequential(
            nn.Linear(output_dim * num_erosion_levels, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, output_dim)
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        topo_features = []
        for erosion, encoder in zip(self.erosion_layers, self.topo_encoders):
            eroded_feat = erosion(feat)
            topo_feat = encoder(eroded_feat)
            topo_features.append(topo_feat)

        topo_concat = torch.cat(topo_features, dim=1)
        topo_final = self.topo_fusion(topo_concat)
        return F.normalize(topo_final, dim=1)


# ============================================================
# 支路1: 膨胀腐蚀支路 (Erosion-Dilation Branch)
# ============================================================
class ErosionDilationBranch(nn.Module):
    """
    改进的膨胀/腐蚀支路：引入更深的瓶颈结构和局部上下文聚合，输出更高维特征。
    """

    def __init__(
        self,
        input_channels: int = 2048,
        num_levels: int = 4,
        output_dim: int = 128
    ):
        super().__init__()
        self.num_levels = num_levels
        self.output_dim = output_dim

        self.erosion_layers = nn.ModuleList([
            MorphologicalErosion(kernel_size=3, num_erosions=i)
            for i in range(num_levels)
        ])

        self.encoders = nn.ModuleList()
        for _ in range(num_levels):
            encoder = nn.Sequential(
                nn.Conv2d(input_channels, 512, kernel_size=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 256, kernel_size=3, padding=1, groups=32, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((2, 2)),
                nn.Flatten(),
                nn.Linear(256 * 4, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU()
            )
            self.encoders.append(encoder)

        self.fusion = nn.Sequential(
            nn.Linear(output_dim * num_levels, output_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim)
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        features = []
        for erosion, encoder in zip(self.erosion_layers, self.encoders):
            eroded = erosion(feat)
            encoded = encoder(eroded)
            features.append(encoded)

        concat = torch.cat(features, dim=1)
        output = self.fusion(concat)
        return output


# ============================================================
# 支路2: 骨架化支路 (Skeletonization Branch)
# ============================================================
class SkeletonBranch(nn.Module):
    """
    改进的骨架化支路：多尺度 + 可学习融合 + 注意力加权。
    """

    def __init__(
        self,
        input_channels: int = 2048,
        num_scales: int = 4,
        output_dim: int = 128
    ):
        super().__init__()
        self.num_scales = num_scales
        self.output_dim = output_dim

        self.erosion_ops = nn.ModuleList([
            MorphologicalErosion(kernel_size=3, num_erosions=i + 1)
            for i in range(num_scales)
        ])
        self.dilation_ops = nn.ModuleList([
            MorphologicalDilation(kernel_size=3, num_dilations=i + 1)
            for i in range(num_scales)
        ])

        self.encoders = nn.ModuleList()
        for scale in range(num_scales):
            encoder = nn.Sequential(
                nn.Conv2d(input_channels, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    256,
                    256,
                    kernel_size=3,
                    padding=1 + scale % 2,
                    dilation=1 + scale % 2,
                    bias=False
                ),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((2, 2)),
                nn.Flatten(),
                nn.Linear(256 * 4, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU()
            )
            self.encoders.append(encoder)

        self.scale_attention = nn.Sequential(
            nn.Linear(output_dim * num_scales, 128),
            nn.GELU(),
            nn.Linear(128, num_scales),
            nn.Softmax(dim=1)
        )

        self.post_fusion = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        skeleton_features = []

        for erosion, dilation, encoder in zip(
            self.erosion_ops, self.dilation_ops, self.encoders
        ):
            eroded = erosion(feat)
            opened = dilation(eroded)
            skeleton = feat - opened
            encoded = encoder(skeleton)
            skeleton_features.append(encoded)

        stacked = torch.stack(skeleton_features, dim=1)  # [B, num_scales, output_dim]
        concat = torch.cat(skeleton_features, dim=1)
        attn = self.scale_attention(concat).unsqueeze(-1)
        fused = (stacked * attn).sum(dim=1)
        fused = self.post_fusion(fused)
        return fused


# ============================================================
# 支路3: 边缘检测支路 (Edge Detection Branch)
# ============================================================
class EdgeDetectionBranch(nn.Module):
    """
    改进的边缘检测支路：多算子特征 + 轻量注意力融合。
    """

    def __init__(
        self,
        input_channels: int = 2048,
        output_dim: int = 128
    ):
        super().__init__()
        self.output_dim = output_dim

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3).repeat(input_channels, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3).repeat(input_channels, 1, 1, 1))

        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.register_buffer('laplacian', laplacian.view(1, 1, 3, 3).repeat(input_channels, 1, 1, 1))

        self.input_channels = input_channels
        self.erosion = MorphologicalErosion(kernel_size=3, num_erosions=1)
        self.dilation = MorphologicalDilation(kernel_size=3, num_dilations=1)

        def make_encoder():
            return nn.Sequential(
                nn.Conv2d(input_channels, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False, groups=32),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((2, 2)),
                nn.Flatten(),
                nn.Linear(256 * 4, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU()
            )

        self.sobel_encoder = make_encoder()
        self.laplacian_encoder = make_encoder()
        self.morph_grad_encoder = make_encoder()

        self.edge_attention = nn.Sequential(
            nn.Linear(output_dim * 3, 128),
            nn.GELU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=1)
        )

        self.fusion = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        edge_x = F.conv2d(feat, self.sobel_x, padding=1, groups=self.input_channels)
        edge_y = F.conv2d(feat, self.sobel_y, padding=1, groups=self.input_channels)
        sobel_edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-8)

        laplacian_edge = torch.abs(F.conv2d(feat, self.laplacian, padding=1, groups=self.input_channels))

        dilated = self.dilation(feat)
        eroded = self.erosion(feat)
        morph_gradient = dilated - eroded

        sobel_feat = self.sobel_encoder(sobel_edge)
        laplacian_feat = self.laplacian_encoder(laplacian_edge)
        morph_feat = self.morph_grad_encoder(morph_gradient)

        stacked = torch.stack([sobel_feat, laplacian_feat, morph_feat], dim=1)
        concat = torch.cat([sobel_feat, laplacian_feat, morph_feat], dim=1)
        attn = self.edge_attention(concat).unsqueeze(-1)
        weighted = (stacked * attn).sum(dim=1)
        output = self.fusion(weighted)
        return output


# ============================================================
# 支路4: 频域拓扑支路 (Frequency Domain Branch)
# ============================================================
class FrequencyDomainBranch(nn.Module):
    """
    改进的频域支路：分离低/中/高频，并通过自适应融合捕获整体+细节。
    """

    def __init__(
        self,
        input_channels: int = 2048,
        output_dim: int = 128
    ):
        super().__init__()
        self.output_dim = output_dim

        def make_encoder(pool_size: int):
            return nn.Sequential(
                nn.AdaptiveAvgPool2d((pool_size, pool_size)),
                nn.Conv2d(input_channels, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(256, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU()
            )

        self.low_freq_encoder = make_encoder(6)
        self.band_pass_encoder = make_encoder(4)

        self.high_freq_conv = nn.Sequential(
            nn.Conv2d(
                input_channels,
                input_channels,
                kernel_size=3,
                padding=1,
                groups=input_channels,
                bias=False
            ),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True)
        )

        self.high_freq_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )

        self.freq_attention = nn.Sequential(
            nn.Linear(output_dim * 3, 128),
            nn.GELU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=1)
        )

        self.fusion = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        low_freq = self.low_freq_encoder(feat)

        blur = F.avg_pool2d(feat, kernel_size=5, stride=1, padding=2)
        high_freq_map = self.high_freq_conv(feat - blur)
        high_freq = self.high_freq_encoder(high_freq_map)

        band_map = F.avg_pool2d(feat, kernel_size=3, stride=2, padding=1)
        band_map = F.interpolate(band_map, size=feat.shape[-2:], mode='bilinear', align_corners=False)
        band_pass = self.band_pass_encoder(band_map)

        stacked = torch.stack([low_freq, band_pass, high_freq], dim=1)
        concat = torch.cat([low_freq, band_pass, high_freq], dim=1)
        attn = self.freq_attention(concat).unsqueeze(-1)
        fused = (stacked * attn).sum(dim=1)
        output = self.fusion(fused)
        return output


# ============================================================
# 多支路拓扑特征提取器
# ============================================================
class MultiTopoFeatureExtractor(nn.Module):
    """
    多支路拓扑特征提取器：旧版腐蚀支路 + 强化版三支路 + 自适应权重。
    """

    def __init__(
        self,
        input_channels: int = 2048,
        topo_dim: int = 128,
        use_legacy_branch: bool = True,
        use_skeleton_branch: bool = True,
        use_edge_branch: bool = True,
        use_frequency_branch: bool = True,
        legacy_branch_dim: int = 128,
        branch_dim: int = 128,
        preserve_legacy_residual: bool = True,
        fusion_alpha_init: float = -5.0
    ):
        super().__init__()
        self.topo_dim = topo_dim
        self.use_adaptive_weights = True
        self.preserve_legacy_residual = preserve_legacy_residual
        # alpha = sigmoid(fusion_alpha) in (0, 1); init small => output ~ legacy
        self.fusion_alpha = nn.Parameter(torch.tensor(float(fusion_alpha_init)))

        self.branches = nn.ModuleDict()
        self.branch_norms = nn.ModuleDict()
        self.branch_names: List[str] = []
        self.branch_dims: List[int] = []

        def register_branch(name: str, module: nn.Module, dim: int):
            self.branches[name] = module
            self.branch_norms[name] = nn.LayerNorm(dim)
            self.branch_names.append(name)
            self.branch_dims.append(dim)

        if use_legacy_branch:
            register_branch(
                'legacy',
                LegacyTopoBranch(
                    input_channels=input_channels,
                    num_erosion_levels=4,
                    output_dim=legacy_branch_dim
                ),
                legacy_branch_dim
            )

        if use_skeleton_branch:
            register_branch(
                'skeleton',
                SkeletonBranch(
                    input_channels=input_channels,
                    num_scales=4,
                    output_dim=branch_dim
                ),
                branch_dim
            )

        if use_edge_branch:
            register_branch(
                'edge',
                EdgeDetectionBranch(
                    input_channels=input_channels,
                    output_dim=branch_dim
                ),
                branch_dim
            )

        if use_frequency_branch:
            register_branch(
                'frequency',
                FrequencyDomainBranch(
                    input_channels=input_channels,
                    output_dim=branch_dim
                ),
                branch_dim
            )

        self.num_branches = len(self.branch_names)
        assert self.num_branches > 0, "At least one topology branch must be enabled"

        self.total_branch_dim = sum(self.branch_dims)
        self.static_branch_weights = nn.Parameter(
            torch.ones(self.num_branches) / self.num_branches
        )

        self.branch_dropout = nn.Dropout(p=0.1)
        self.context_projector = nn.Sequential(
            nn.Linear(self.total_branch_dim, self.total_branch_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.adaptive_weight_net = nn.Sequential(
            nn.Linear(self.total_branch_dim, 128),
            nn.GELU(),
            nn.Linear(128, self.num_branches),
            nn.Softmax(dim=1)
        )

        self.final_projection = nn.Sequential(
            nn.Linear(self.total_branch_dim * 2, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, topo_dim)
        )

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Special-case: only legacy branch enabled => exact legacy output (no extra fusion MLP)
        if self.num_branches == 1 and self.branch_names[0] == 'legacy':
            legacy_feat = self.branches['legacy'](feat)
            return legacy_feat, {'legacy': legacy_feat}

        B = feat.size(0)
        branch_outputs: Dict[str, torch.Tensor] = {}
        branch_features: List[torch.Tensor] = []

        for name in self.branch_names:
            branch_feat = self.branches[name](feat)
            branch_feat = self.branch_norms[name](branch_feat)
            branch_outputs[name] = branch_feat
            branch_features.append(branch_feat)

        raw_concat = torch.cat(branch_features, dim=1)
        raw_concat = self.branch_dropout(raw_concat)
        context = self.context_projector(raw_concat)

        if self.use_adaptive_weights:
            weights = self.adaptive_weight_net(context)
        else:
            weights = F.softmax(self.static_branch_weights, dim=0).unsqueeze(0).expand(B, -1)

        # Preserve legacy branch as a residual base: let the fusion branch focus on "others".
        legacy_raw = None
        if self.preserve_legacy_residual and ('legacy' in self.branch_names):
            legacy_raw = self.branches['legacy'](feat)  # LegacyTopoBranch already returns L2-normalized
            legacy_idx = self.branch_names.index('legacy')
            weights = weights.clone()
            weights[:, legacy_idx] = 0.0
            denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
            weights = weights / denom

        scaled_features = []
        for idx, branch_feat in enumerate(branch_features):
            weight = weights[:, idx].unsqueeze(1)
            scaled_features.append(branch_feat * weight)

        weighted_concat = torch.cat(scaled_features, dim=1)
        combined = torch.cat([weighted_concat, context], dim=1)
        topo_final = self.final_projection(combined)
        topo_final = F.normalize(topo_final, dim=1)

        if legacy_raw is not None:
            alpha = torch.sigmoid(self.fusion_alpha)
            topo_final = F.normalize((1 - alpha) * legacy_raw + alpha * topo_final, dim=1)

        return topo_final, branch_outputs


class MoCoV3HybridTopo(nn.Module):
    """
    MoCoV3 + 多支路拓扑网络

    特点:
    1. 多种拓扑提取方式组合
    2. 固定形态学操作，保证稳定性
    3. 简单融合策略，避免过拟合
    4. 支持灵活的支路启用/禁用
    """

    def __init__(
        self,
        feature_dim: int = 256,
        topo_dim: int = 128,
        K: int = 4096,
        m: float = 0.999,
        T: float = 0.07,
        topo_weight: float = 0.35,
        use_legacy_branch: bool = True,
        use_skeleton_branch: bool = True,
        use_edge_branch: bool = True,
        use_frequency_branch: bool = True,
        device: Optional[torch.device] = None
    ):
        super().__init__()

        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.feature_dim = feature_dim
        self.topo_dim = topo_dim
        self.K = K
        self.m = m
        self.T = T
        self.topo_weight = topo_weight

        print(f"\n{'=' * 70}")
        print(f"[MoCoV3HybridTopo] Initialized with Hybrid Topology Network")
        print(f"{'=' * 70}")
        print(f"  Device: {self.device}")
        print(f"  Visual feature dim: {feature_dim}")
        print(f"  Topology feature dim: {topo_dim}")
        print(f"  Queue size: {K}")
        print(f"  Topology weight: {topo_weight}")
        print(f"  Branches enabled:")
        print(f"    - Legacy (old erosion): {use_legacy_branch}")
        print(f"    - Skeleton: {use_skeleton_branch}")
        print(f"    - Edge Detection: {use_edge_branch}")
        print(f"    - Frequency Domain: {use_frequency_branch}")
        print(f"{'=' * 70}\n")

        # ========== 查询编码器（可训练） ==========
        self.encoder_q = models.resnet50(weights="IMAGENET1K_V1")
        resnet_dim = self.encoder_q.fc.in_features  # 2048

        # 移除分类头，保留到avgpool前的特征
        self.encoder_q = nn.Sequential(*list(self.encoder_q.children())[:-2])

        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 视觉特征投影头
        self.visual_mlp_q = nn.Sequential(
            nn.Linear(resnet_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim)
        )

        # ========== 键编码器（动量更新） ==========
        self.encoder_k = models.resnet50(weights="IMAGENET1K_V1")
        self.encoder_k = nn.Sequential(*list(self.encoder_k.children())[:-2])

        self.visual_mlp_k = nn.Sequential(
            nn.Linear(resnet_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim)
        )

        # ========== 多支路拓扑提取器 ==========
        self.topo_extractor_q = MultiTopoFeatureExtractor(
            input_channels=resnet_dim,
            topo_dim=topo_dim,
            use_legacy_branch=use_legacy_branch,
            use_skeleton_branch=use_skeleton_branch,
            use_edge_branch=use_edge_branch,
            use_frequency_branch=use_frequency_branch
        )
        self.topo_extractor_k = MultiTopoFeatureExtractor(
            input_channels=resnet_dim,
            topo_dim=topo_dim,
            use_legacy_branch=use_legacy_branch,
            use_skeleton_branch=use_skeleton_branch,
            use_edge_branch=use_edge_branch,
            use_frequency_branch=use_frequency_branch
        )

        # ========== 队列：视觉特征队列 ==========
        self.register_buffer("visual_queue", torch.randn(feature_dim, K))
        self.visual_queue = F.normalize(self.visual_queue, dim=0)

        # ========== 队列：拓扑特征队列 ==========
        self.register_buffer("topo_queue", torch.randn(topo_dim, K))
        self.topo_queue = F.normalize(self.topo_queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # ========== 初始化键编码器 ==========
        self._init_key_encoder()

        self.to(self.device)

    def _init_key_encoder(self):
        """初始化键编码器（复制查询编码器权重）"""
        # 编码器
        for pq, pk in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            pk.data.copy_(pq.data)
            pk.requires_grad = False

        # 视觉投影头
        for pq, pk in zip(
            self.visual_mlp_q.parameters(), self.visual_mlp_k.parameters()
        ):
            pk.data.copy_(pq.data)
            pk.requires_grad = False

        # 拓扑提取器
        for pq, pk in zip(
            self.topo_extractor_q.parameters(),
            self.topo_extractor_k.parameters()
        ):
            pk.data.copy_(pq.data)
            pk.requires_grad = False

    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播"""
        img1, img2 = img1.to(self.device), img2.to(self.device)

        # ========== 查询编码路径（可训练） ==========
        feat_q = self.encoder_q(img1)

        # 视觉特征路径
        visual_q = self.avgpool(feat_q).flatten(1)
        visual_q = self.visual_mlp_q(visual_q)
        visual_q = F.normalize(visual_q, dim=1)

        # 拓扑特征路径
        topo_q, _ = self.topo_extractor_q(feat_q)

        # ========== 键编码路径（无梯度） ==========
        with torch.no_grad():
            self.momentum_update_key_encoder()

            feat_k = self.encoder_k(img2)

            # 视觉特征路径
            visual_k = self.avgpool(feat_k).flatten(1)
            visual_k = self.visual_mlp_k(visual_k)
            visual_k = F.normalize(visual_k, dim=1)

            # 拓扑特征路径
            topo_k, _ = self.topo_extractor_k(feat_k)

        return visual_q, visual_k, topo_q, topo_k

    @torch.no_grad()
    def momentum_update_key_encoder(self):
        """使用动量更新键编码器"""
        # 编码器
        for pq, pk in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            pk.data.mul_(self.m).add_(pq.data, alpha=1 - self.m)

        # 视觉投影头
        for pq, pk in zip(
            self.visual_mlp_q.parameters(), self.visual_mlp_k.parameters()
        ):
            pk.data.mul_(self.m).add_(pq.data, alpha=1 - self.m)

        # 拓扑提取器
        for pq, pk in zip(
            self.topo_extractor_q.parameters(),
            self.topo_extractor_k.parameters()
        ):
            pk.data.mul_(self.m).add_(pq.data, alpha=1 - self.m)

    def contrastive_loss(
        self,
        visual_q: torch.Tensor,
        visual_k: torch.Tensor,
        topo_q: torch.Tensor,
        topo_k: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """双路对比损失：视觉 + 拓扑"""
        N = visual_q.shape[0]

        # 视觉对比损失
        visual_l_pos = torch.einsum('nc,nc->n', visual_q, visual_k).unsqueeze(-1)
        visual_l_neg = torch.einsum(
            'nc,ck->nk', visual_q, self.visual_queue.clone().detach()
        )

        visual_logits = torch.cat([visual_l_pos, visual_l_neg], dim=1) / self.T
        labels = torch.zeros(N, dtype=torch.long).to(visual_q.device)
        visual_loss = F.cross_entropy(visual_logits, labels)

        # 拓扑对比损失
        topo_l_pos = torch.einsum('nc,nc->n', topo_q, topo_k).unsqueeze(-1)
        topo_l_neg = torch.einsum(
            'nc,ck->nk', topo_q, self.topo_queue.clone().detach()
        )

        topo_logits = torch.cat([topo_l_pos, topo_l_neg], dim=1) / self.T
        topo_loss = F.cross_entropy(topo_logits, labels)

        # 加权组合
        total_loss = (1 - self.topo_weight) * visual_loss + self.topo_weight * topo_loss

        diagnostics = {
            'visual_loss': float(visual_loss.item()),
            'topo_loss': float(topo_loss.item()),
            'total_loss': float(total_loss.item()),
        }

        return total_loss, diagnostics

    @torch.no_grad()
    def update_queue(self, visual_k: torch.Tensor, topo_k: torch.Tensor):
        """更新队列"""
        visual_k = F.normalize(visual_k, dim=1)
        topo_k = F.normalize(topo_k, dim=1)

        batch_size = visual_k.shape[0]
        ptr = int(self.queue_ptr)

        if ptr + batch_size > self.K:
            num_first = self.K - ptr
            num_next = batch_size - num_first

            self.visual_queue[:, ptr:] = visual_k[:num_first].T
            self.visual_queue[:, :num_next] = visual_k[num_first:].T

            self.topo_queue[:, ptr:] = topo_k[:num_first].T
            self.topo_queue[:, :num_next] = topo_k[num_first:].T

            self.queue_ptr[0] = num_next
        else:
            self.visual_queue[:, ptr:ptr + batch_size] = visual_k.T
            self.topo_queue[:, ptr:ptr + batch_size] = topo_k.T

            self.queue_ptr[0] = (ptr + batch_size) % self.K

    def extract_features(
        self,
        img: torch.Tensor,
        use_query_encoder: bool = True,
        feature_type: str = 'visual'
    ) -> torch.Tensor:
        """提取特征"""
        img = img.to(self.device)

        with torch.no_grad():
            if use_query_encoder:
                feat = self.encoder_q(img)
            else:
                feat = self.encoder_k(img)

            if feature_type == 'visual':
                feat = self.avgpool(feat).flatten(1)
                if use_query_encoder:
                    feat = self.visual_mlp_q(feat)
                else:
                    feat = self.visual_mlp_k(feat)
                feat = F.normalize(feat, dim=1)

            elif feature_type == 'topo':
                if use_query_encoder:
                    feat, _ = self.topo_extractor_q(feat)
                else:
                    feat, _ = self.topo_extractor_k(feat)

            elif feature_type == 'both':
                visual_feat = self.avgpool(feat).flatten(1)
                visual_feat = self.visual_mlp_q(visual_feat) if use_query_encoder else self.visual_mlp_k(visual_feat)
                topo_feat, _ = self.topo_extractor_q(feat) if use_query_encoder else self.topo_extractor_k(feat)

                visual_feat = F.normalize(visual_feat, dim=1)
                topo_feat = F.normalize(topo_feat, dim=1)
                feat = torch.cat([visual_feat, topo_feat], dim=1)
            else:
                raise ValueError(f"Unknown feature_type: {feature_type}")

        return feat


# ============================================================
# 用于ReID的拓扑模块（可插拔）
# ============================================================
class TopologyModule(nn.Module):
    """
    可插拔的拓扑模块，用于添加到其他ReID模型中

    使用方式:
    1. 在backbone输出后添加此模块
    2. 将拓扑特征与原有特征拼接/相加
    """

    def __init__(
        self,
        input_channels: int = 2048,
        topo_dim: int = 128,
        use_legacy_branch: bool = True,
        use_skeleton_branch: bool = True,
        use_edge_branch: bool = False,  # ReID场景可能不需要
        use_frequency_branch: bool = False
    ):
        super().__init__()

        self.topo_extractor = MultiTopoFeatureExtractor(
            input_channels=input_channels,
            topo_dim=topo_dim,
            use_legacy_branch=use_legacy_branch,
            use_skeleton_branch=use_skeleton_branch,
            use_edge_branch=use_edge_branch,
            use_frequency_branch=use_frequency_branch
        )

        self.topo_dim = topo_dim

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: Backbone输出的特征图 [B, C, H, W]

        Returns:
            topo_feat: 拓扑特征 [B, topo_dim]
        """
        topo_feat, _ = self.topo_extractor(feat)
        return topo_feat
