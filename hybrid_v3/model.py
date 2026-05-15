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


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple, Dict, Optional, List
import math





class MorphologicalErosion(nn.Module):


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
\
\


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





class ErosionDilationBranch(nn.Module):
\
\


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





class SkeletonBranch(nn.Module):
\
\


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

        stacked = torch.stack(skeleton_features, dim=1)
        concat = torch.cat(skeleton_features, dim=1)
        attn = self.scale_attention(concat).unsqueeze(-1)
        fused = (stacked * attn).sum(dim=1)
        fused = self.post_fusion(fused)
        return fused





class EdgeDetectionBranch(nn.Module):
\
\


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





class FrequencyDomainBranch(nn.Module):
\
\


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





class MultiTopoFeatureExtractor(nn.Module):
\
\


    def __init__(
        self,
        input_channels: int = 2048,
        topo_dim: int = 128,
        use_legacy_branch: bool = True,
        use_skeleton_branch: bool = False,
        use_edge_branch: bool = True,
        use_frequency_branch: bool = False,
        legacy_branch_dim: int = 128,
        branch_dim: int = 128,
        preserve_legacy_residual: bool = True,
        fusion_alpha_init: float = -5.0
    ):
        super().__init__()
        self.topo_dim = topo_dim
        self.use_adaptive_weights = True
        self.preserve_legacy_residual = preserve_legacy_residual

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


        legacy_raw = None
        if self.preserve_legacy_residual and ('legacy' in self.branch_names):
            legacy_raw = self.branches['legacy'](feat)
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
\
\
\
\
\
\
\
\


    def __init__(
        self,
        feature_dim: int = 256,
        topo_dim: int = 128,
        K: int = 4096,
        m: float = 0.999,
        T: float = 0.07,
        topo_weight: float = 0.15,
        use_topology: bool = True,
        use_legacy_branch: bool = True,
        use_skeleton_branch: bool = False,
        use_edge_branch: bool = True,
        use_frequency_branch: bool = False,
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
        self.use_topology = use_topology

        print(f"\n{'=' * 70}")
        print(f"[MoCoV3HybridTopo] Initialized with Hybrid Topology Network")
        print(f"{'=' * 70}")
        print(f"  Device: {self.device}")
        print(f"  Visual feature dim: {feature_dim}")
        print(f"  Topology feature dim: {topo_dim}")
        print(f"  Queue size: {K}")
        print(f"  Topology weight: {topo_weight}")
        print(f"  Use topology branch: {use_topology}")
        print(f"  Branches enabled:")
        print(f"    - Legacy (old erosion): {use_legacy_branch}")
        print(f"    - Skeleton: {use_skeleton_branch}")
        print(f"    - Edge Detection: {use_edge_branch}")
        print(f"    - Frequency Domain: {use_frequency_branch}")
        print(f"{'=' * 70}\n")


        self.encoder_q = models.resnet50(weights="IMAGENET1K_V1")
        resnet_dim = self.encoder_q.fc.in_features


        self.encoder_q = nn.Sequential(*list(self.encoder_q.children())[:-2])


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


        self.visual_mlp_q = nn.Sequential(
            nn.Linear(resnet_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim)
        )


        self.encoder_k = models.resnet50(weights="IMAGENET1K_V1")
        self.encoder_k = nn.Sequential(*list(self.encoder_k.children())[:-2])

        self.visual_mlp_k = nn.Sequential(
            nn.Linear(resnet_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim)
        )


        self.topo_extractor_q: Optional[MultiTopoFeatureExtractor] = None
        self.topo_extractor_k: Optional[MultiTopoFeatureExtractor] = None
        if self.use_topology:
            if not any([use_legacy_branch, use_skeleton_branch, use_edge_branch, use_frequency_branch]):
                raise ValueError(
                    "use_topology=True but no topology branch is enabled. "
                    "Please enable at least one branch or set use_topology=False."
                )
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


        self.register_buffer("visual_queue", torch.randn(feature_dim, K))
        self.visual_queue = F.normalize(self.visual_queue, dim=0)


        if self.use_topology:
            self.register_buffer("topo_queue", torch.randn(topo_dim, K))
            self.topo_queue = F.normalize(self.topo_queue, dim=0)
        else:
            self.register_buffer("topo_queue", torch.empty(0, K))

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


        self._init_key_encoder()

        self.to(self.device)

    def _init_key_encoder(self):


        for pq, pk in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            pk.data.copy_(pq.data)
            pk.requires_grad = False


        for pq, pk in zip(
            self.visual_mlp_q.parameters(), self.visual_mlp_k.parameters()
        ):
            pk.data.copy_(pq.data)
            pk.requires_grad = False


        if self.use_topology and self.topo_extractor_q is not None and self.topo_extractor_k is not None:
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
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:

        img1, img2 = img1.to(self.device), img2.to(self.device)


        feat_q = self.encoder_q(img1)


        visual_q = self.avgpool(feat_q).flatten(1)
        visual_q = self.visual_mlp_q(visual_q)
        visual_q = F.normalize(visual_q, dim=1)


        topo_q: Optional[torch.Tensor] = None
        if self.use_topology and self.topo_extractor_q is not None:
            topo_q, _ = self.topo_extractor_q(feat_q)


        with torch.no_grad():
            self.momentum_update_key_encoder()

            feat_k = self.encoder_k(img2)


            visual_k = self.avgpool(feat_k).flatten(1)
            visual_k = self.visual_mlp_k(visual_k)
            visual_k = F.normalize(visual_k, dim=1)


            topo_k: Optional[torch.Tensor] = None
            if self.use_topology and self.topo_extractor_k is not None:
                topo_k, _ = self.topo_extractor_k(feat_k)

        return visual_q, visual_k, topo_q, topo_k

    @torch.no_grad()
    def momentum_update_key_encoder(self):


        for pq, pk in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            pk.data.mul_(self.m).add_(pq.data, alpha=1 - self.m)


        for pq, pk in zip(
            self.visual_mlp_q.parameters(), self.visual_mlp_k.parameters()
        ):
            pk.data.mul_(self.m).add_(pq.data, alpha=1 - self.m)


        if self.use_topology and self.topo_extractor_q is not None and self.topo_extractor_k is not None:
            for pq, pk in zip(
                self.topo_extractor_q.parameters(),
                self.topo_extractor_k.parameters()
            ):
                pk.data.mul_(self.m).add_(pq.data, alpha=1 - self.m)

    def contrastive_loss(
        self,
        visual_q: torch.Tensor,
        visual_k: torch.Tensor,
        topo_q: Optional[torch.Tensor],
        topo_k: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        N = visual_q.shape[0]


        visual_l_pos = torch.einsum('nc,nc->n', visual_q, visual_k).unsqueeze(-1)
        visual_l_neg = torch.einsum(
            'nc,ck->nk', visual_q, self.visual_queue.clone().detach()
        )

        visual_logits = torch.cat([visual_l_pos, visual_l_neg], dim=1) / self.T
        labels = torch.zeros(N, dtype=torch.long).to(visual_q.device)
        visual_loss = F.cross_entropy(visual_logits, labels)

        topo_loss = torch.zeros((), device=visual_q.device)
        if self.use_topology:
            if topo_q is None or topo_k is None:
                raise ValueError("Topology features are required when use_topology=True.")
            topo_l_pos = torch.einsum('nc,nc->n', topo_q, topo_k).unsqueeze(-1)
            topo_l_neg = torch.einsum(
                'nc,ck->nk', topo_q, self.topo_queue.clone().detach()
            )

            topo_logits = torch.cat([topo_l_pos, topo_l_neg], dim=1) / self.T
            topo_loss = F.cross_entropy(topo_logits, labels)
            total_loss = (1 - self.topo_weight) * visual_loss + self.topo_weight * topo_loss
        else:
            total_loss = visual_loss

        diagnostics = {
            'visual_loss': float(visual_loss.item()),
            'topo_loss': float(topo_loss.item()),
            'total_loss': float(total_loss.item()),
        }

        return total_loss, diagnostics

    @torch.no_grad()
    def update_queue(self, visual_k: torch.Tensor, topo_k: Optional[torch.Tensor]):

        visual_k = F.normalize(visual_k, dim=1)
        if self.use_topology:
            if topo_k is None:
                raise ValueError("topo_k is required when use_topology=True.")
            topo_k = F.normalize(topo_k, dim=1)

        batch_size = visual_k.shape[0]
        ptr = int(self.queue_ptr)

        if ptr + batch_size > self.K:
            num_first = self.K - ptr
            num_next = batch_size - num_first

            self.visual_queue[:, ptr:] = visual_k[:num_first].T
            self.visual_queue[:, :num_next] = visual_k[num_first:].T

            if self.use_topology:
                self.topo_queue[:, ptr:] = topo_k[:num_first].T
                self.topo_queue[:, :num_next] = topo_k[num_first:].T

            self.queue_ptr[0] = num_next
        else:
            self.visual_queue[:, ptr:ptr + batch_size] = visual_k.T
            if self.use_topology:
                self.topo_queue[:, ptr:ptr + batch_size] = topo_k.T

            self.queue_ptr[0] = (ptr + batch_size) % self.K

    def _extract_visual_feat(self, feat_map: torch.Tensor, use_query_encoder: bool) -> torch.Tensor:
        visual_feat = self.avgpool(feat_map).flatten(1)
        if use_query_encoder:
            visual_feat = self.visual_mlp_q(visual_feat)
        else:
            visual_feat = self.visual_mlp_k(visual_feat)
        return F.normalize(visual_feat, dim=1)

    def extract_features(
        self,
        img: torch.Tensor,
        use_query_encoder: bool = True,
        feature_type: str = 'visual'
    ) -> torch.Tensor:

        img = img.to(self.device)
        feature_type = str(feature_type).lower()

        with torch.no_grad():
            if use_query_encoder:
                feat = self.encoder_q(img)
            else:
                feat = self.encoder_k(img)

            if feature_type == 'visual':
                feat = self._extract_visual_feat(feat, use_query_encoder)

            elif feature_type == 'topo':
                if not self.use_topology:
                    raise ValueError("Topology branch is disabled. feature_type='topo' is unavailable.")
                if use_query_encoder:
                    feat, _ = self.topo_extractor_q(feat)
                else:
                    feat, _ = self.topo_extractor_k(feat)

            elif feature_type == 'both':
                visual_feat = self._extract_visual_feat(feat, use_query_encoder)
                if not self.use_topology:
                    feat = visual_feat
                else:
                    topo_feat, _ = self.topo_extractor_q(feat) if use_query_encoder else self.topo_extractor_k(feat)
                    topo_feat = F.normalize(topo_feat, dim=1)
                    feat = torch.cat([visual_feat, topo_feat], dim=1)
            else:
                raise ValueError(f"Unknown feature_type: {feature_type}")

        return feat





class TopologyModule(nn.Module):
\
\
\
\
\
\


    def __init__(
        self,
        input_channels: int = 2048,
        topo_dim: int = 128,
        use_legacy_branch: bool = True,
        use_skeleton_branch: bool = False,
        use_edge_branch: bool = False,
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
\
\
\
\
\
\

        topo_feat, _ = self.topo_extractor(feat)
        return topo_feat
