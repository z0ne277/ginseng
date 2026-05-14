import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple, Dict, Optional


class MorphologicalErosion(nn.Module):
    """
    形态学腐蚀层：在特征图上消除细节，保留拓扑结构

    原理：腐蚀 = 最小池化
    """

    def __init__(self, kernel_size: int = 3, num_erosions: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_erosions = num_erosions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            eroded: [B, C, H, W] (可能尺寸稍小)
        """
        x_eroded = x
        for _ in range(self.num_erosions):
            # 腐蚀 = -max_pool(-x)（最小池化）
            x_eroded = -F.max_pool2d(
                -x_eroded,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2
            )
        return x_eroded


class TopoFeatureExtractor(nn.Module):
    """
    拓扑特征提取支线：多尺度腐蚀特征

    核心思想：
    - 在高层特征处分叉
    - 通过逐级腐蚀得到多层次形状信息
    - 每个腐蚀层级的特征独立编码
    - 最后融合成紧凑的拓扑表示
    """

    def __init__(
            self,
            input_channels: int = 2048,
            num_erosion_levels: int = 4,
            topo_dim: int = 128
    ):
        """
        Args:
            input_channels: 输入特征通道数 (ResNet50为2048)
            num_erosion_levels: 腐蚀层级数 (0, 1, 2, ..., n)
            topo_dim: 最终拓扑特征维度
        """
        super().__init__()
        self.num_erosion_levels = num_erosion_levels
        self.input_channels = input_channels
        self.topo_dim = topo_dim

        # 每个腐蚀层级对应一个腐蚀操作器
        self.erosion_layers = nn.ModuleList([
            MorphologicalErosion(kernel_size=3, num_erosions=i)
            for i in range(num_erosion_levels)
        ])

        # 每个腐蚀层级的特征编码器
        # 通过小型卷积网络处理腐蚀后的特征
        self.topo_encoders = nn.ModuleList()
        for _ in range(num_erosion_levels):
            encoder = nn.Sequential(
                # 降维卷积：2048 -> 512
                nn.Conv2d(input_channels, 512, kernel_size=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),

                # 自适应池化到固定大小 (4x4)
                nn.AdaptiveAvgPool2d((4, 4)),

                # 进一步降维：512 -> 256
                nn.Conv2d(512, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),

                # 全局平均池化
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),

                # 线性投影到topo_dim
                nn.Linear(256, topo_dim)
            )
            self.topo_encoders.append(encoder)

        # 融合所有层级特征的MLP
        self.topo_fusion = nn.Sequential(
            nn.Linear(topo_dim * num_erosion_levels, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, topo_dim)
        )

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """
        从高层特征提取多尺度拓扑信息

        Args:
            feat: 高层特征 [B, 2048, H, W]

        Returns:
            topo_final: 融合的拓扑特征 [B, topo_dim]
            topo_features_list: 各层级特征列表 (用于debug)
        """
        topo_features = []

        # 收集多层次腐蚀特征
        for i, (erosion, encoder) in enumerate(
                zip(self.erosion_layers, self.topo_encoders)
        ):
            # 执行腐蚀
            eroded_feat = erosion(feat)  # [B, 2048, H, W]

            # 编码腐蚀特征
            topo_feat = encoder(eroded_feat)  # [B, topo_dim]
            topo_features.append(topo_feat)

        # 拼接所有层级的特征
        topo_concat = torch.cat(topo_features, dim=1)  # [B, topo_dim * num_levels]

        # 通过MLP融合
        topo_final = self.topo_fusion(topo_concat)  # [B, topo_dim]

        return F.normalize(topo_final, dim=1), topo_features


class CBAM(nn.Module):
    """通道 + 空间联合注意力机制"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 通道注意力
        ca = self.channel_attention(x) * x

        # 空间注意力
        avg_out = torch.mean(ca, dim=1, keepdim=True)
        max_out = torch.max(ca, dim=1, keepdim=True)[0]
        sa = self.spatial_attention(
            torch.cat([avg_out, max_out], dim=1)
        ) * ca

        return sa


class ImprovedMoCoV3WithTopoSideline(nn.Module):
    """
    改进的MoCoV3：支线拓扑网络 + 多任务对比损失

    改进点：
    1. 添加独立的拓扑提取支线（不影响主路径）
    2. 使用多尺度腐蚀特征捕捉拓扑信息
    3. 双路对比学习：视觉损失 + 拓扑损失
    4. 同时维护两个队列（视觉特征队列 + 拓扑特征队列）
    """

    def __init__(
            self,
            feature_dim: int = 256,
            topo_dim: int = 128,
            K: int = 4096,
            m: float = 0.999,
            T: float = 0.07,
            topo_weight: float = 0.35,
            num_erosion_levels: int = 4,
            device: Optional[torch.device] = None
    ):
        """
        Args:
            feature_dim: 视觉特征维度
            topo_dim: 拓扑特征维度
            K: 队列大小
            m: 动量系数
            T: 温度参数
            topo_weight: 拓扑损失权重 (0.2-0.4)
            num_erosion_levels: 腐蚀层级数
            device: 计算设备
        """
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
        self.num_erosion_levels = num_erosion_levels

        print(f"\n{'=' * 60}")
        print(f"[ImprovedMoCoV3] Initialized with Topology Sideline")
        print(f"{'=' * 60}")
        print(f"  Device: {self.device}")
        print(f"  Visual feature dim: {feature_dim}")
        print(f"  Topology feature dim: {topo_dim}")
        print(f"  Queue size: {K}")
        print(f"  Topology weight: {topo_weight}")
        print(f"  Erosion levels: {num_erosion_levels}")
        print(f"{'=' * 60}\n")

        # ========== 查询编码器（可训练） ==========
        self.encoder_q = models.resnet50(weights="IMAGENET1K_V1")
        resnet_dim = self.encoder_q.fc.in_features  # 2048

        # 移除分类头，保留到avgpool前的特征
        self.encoder_q = nn.Sequential(*list(self.encoder_q.children())[:-2])

        # CBAM注意力模块
        self.cbam_q = CBAM(resnet_dim)

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
        self.cbam_k = CBAM(resnet_dim)

        self.visual_mlp_k = nn.Sequential(
            nn.Linear(resnet_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim)
        )

        # ========== 【新增】拓扑提取支线 ==========
        self.topo_extractor_q = TopoFeatureExtractor(
            input_channels=resnet_dim,
            num_erosion_levels=num_erosion_levels,
            topo_dim=topo_dim
        )
        self.topo_extractor_k = TopoFeatureExtractor(
            input_channels=resnet_dim,
            num_erosion_levels=num_erosion_levels,
            topo_dim=topo_dim
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

        # CBAM
        for pq, pk in zip(self.cbam_q.parameters(), self.cbam_k.parameters()):
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
        """
        前向传播：同时提取视觉和拓扑特征

        Args:
            img1: 第一个视图 [B, 3, H, W]
            img2: 第二个视图 [B, 3, H, W]

        Returns:
            visual_q: 查询视觉特征 [B, feature_dim]
            visual_k: 键视觉特征 [B, feature_dim]
            topo_q: 查询拓扑特征 [B, topo_dim]
            topo_k: 键拓扑特征 [B, topo_dim]
        """
        img1, img2 = img1.to(self.device), img2.to(self.device)

        # ========== 查询编码路径（可训练） ==========
        feat_q = self.encoder_q(img1)  # [B, 2048, H', W']
        feat_q_cbam = self.cbam_q(feat_q)

        # 视觉特征路径
        visual_q = self.avgpool(feat_q_cbam).flatten(1)  # [B, 2048]
        visual_q = self.visual_mlp_q(visual_q)  # [B, feature_dim]
        visual_q = F.normalize(visual_q, dim=1)

        # 拓扑特征路径（从原始高层特征）
        topo_q, _ = self.topo_extractor_q(feat_q_cbam)  # [B, topo_dim]

        # ========== 键编码路径（无梯度） ==========
        with torch.no_grad():
            self.momentum_update_key_encoder()

            feat_k = self.encoder_k(img2)  # [B, 2048, H', W']
            feat_k_cbam = self.cbam_k(feat_k)

            # 视觉特征路径
            visual_k = self.avgpool(feat_k_cbam).flatten(1)
            visual_k = self.visual_mlp_k(visual_k)
            visual_k = F.normalize(visual_k, dim=1)

            # 拓扑特征路径
            topo_k, _ = self.topo_extractor_k(feat_k_cbam)

        return visual_q, visual_k, topo_q, topo_k

    @torch.no_grad()
    def momentum_update_key_encoder(self):
        """使用动量更新键编码器及其所有模块"""
        # 编码器
        for pq, pk in zip(
                self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            pk.data.mul_(self.m).add_(pq.data, alpha=1 - self.m)

        # CBAM
        for pq, pk in zip(self.cbam_q.parameters(), self.cbam_k.parameters()):
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
        """
        双路对比损失：视觉 + 拓扑

        Args:
            visual_q: 查询视觉特征 [B, feature_dim]
            visual_k: 键视觉特征 [B, feature_dim]
            topo_q: 查询拓扑特征 [B, topo_dim]
            topo_k: 键拓扑特征 [B, topo_dim]

        Returns:
            total_loss: 加权组合损失
            diagnostics: 诊断信息字典
        """
        N = visual_q.shape[0]

        # ========== 视觉对比损失 ==========
        visual_l_pos = torch.einsum('nc,nc->n', visual_q, visual_k).unsqueeze(-1)  # [N, 1]
        visual_l_neg = torch.einsum(
            'nc,ck->nk', visual_q, self.visual_queue.clone().detach()
        )  # [N, K]

        visual_logits = torch.cat([visual_l_pos, visual_l_neg], dim=1) / self.T
        labels = torch.zeros(N, dtype=torch.long).to(visual_q.device)
        visual_loss = F.cross_entropy(visual_logits, labels)

        # ========== 拓扑对比损失 ==========
        topo_l_pos = torch.einsum('nc,nc->n', topo_q, topo_k).unsqueeze(-1)  # [N, 1]
        topo_l_neg = torch.einsum(
            'nc,ck->nk', topo_q, self.topo_queue.clone().detach()
        )  # [N, K]

        topo_logits = torch.cat([topo_l_pos, topo_l_neg], dim=1) / self.T
        topo_loss = F.cross_entropy(topo_logits, labels)

        # ========== 加权组合 ==========
        total_loss = (1 - self.topo_weight) * visual_loss + self.topo_weight * topo_loss

        # ========== 诊断信息 ==========
        diagnostics = {
            'visual_loss': float(visual_loss.item()),
            'topo_loss': float(topo_loss.item()),
            'total_loss': float(total_loss.item()),
        }

        return total_loss, diagnostics

    @torch.no_grad()
    def update_queue(
            self,
            visual_k: torch.Tensor,
            topo_k: torch.Tensor
    ):
        """
        同时更新视觉和拓扑队列

        Args:
            visual_k: 键视觉特征 [B, feature_dim]
            topo_k: 键拓扑特征 [B, topo_dim]
        """
        visual_k = F.normalize(visual_k, dim=1)
        topo_k = F.normalize(topo_k, dim=1)

        batch_size = visual_k.shape[0]
        ptr = int(self.queue_ptr)

        # 处理队列溢出
        if ptr + batch_size > self.K:
            num_first = self.K - ptr
            num_next = batch_size - num_first

            # 视觉队列
            self.visual_queue[:, ptr:] = visual_k[:num_first].T
            self.visual_queue[:, :num_next] = visual_k[num_first:].T

            # 拓扑队列
            self.topo_queue[:, ptr:] = topo_k[:num_first].T
            self.topo_queue[:, :num_next] = topo_k[num_first:].T

            self.queue_ptr[0] = num_next
        else:
            # 视觉队列
            self.visual_queue[:, ptr:ptr + batch_size] = visual_k.T

            # 拓扑队列
            self.topo_queue[:, ptr:ptr + batch_size] = topo_k.T

            self.queue_ptr[0] = (ptr + batch_size) % self.K

    def extract_features(
            self,
            img: torch.Tensor,
            use_query_encoder: bool = True,
            feature_type: str = 'visual'
    ) -> torch.Tensor:
        """
        提取特征（用于推理/评估）

        Args:
            img: 输入图像 [B, 3, H, W]
            use_query_encoder: 使用查询编码器还是键编码器
            feature_type: 'visual' 或 'topo'

        Returns:
            features: 标准化特征 [B, feature_dim或topo_dim]
        """
        img = img.to(self.device)

        with torch.no_grad():
            if use_query_encoder:
                feat = self.encoder_q(img)
                feat = self.cbam_q(feat)
            else:
                feat = self.encoder_k(img)
                feat = self.cbam_k(feat)

            if feature_type == 'visual':
                feat = self.avgpool(feat).flatten(1)
                if use_query_encoder:
                    feat = self.visual_mlp_q(feat)
                else:
                    feat = self.visual_mlp_k(feat)
            elif feature_type == 'topo':
                if use_query_encoder:
                    feat, _ = self.topo_extractor_q(feat)
                else:
                    feat, _ = self.topo_extractor_k(feat)
            else:
                raise ValueError(f"Unknown feature_type: {feature_type}")

            feat = F.normalize(feat, dim=1)

        return feat
