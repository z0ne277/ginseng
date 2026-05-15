import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple, Dict, Optional


class MorphologicalErosion(nn.Module):
\
\
\
\


    def __init__(self, kernel_size: int = 3, num_erosions: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_erosions = num_erosions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
\
\
\
\
\

        x_eroded = x
        for _ in range(self.num_erosions):

            x_eroded = -F.max_pool2d(
                -x_eroded,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2
            )
        return x_eroded


class TopoFeatureExtractor(nn.Module):
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
            input_channels: int = 2048,
            num_erosion_levels: int = 4,
            topo_dim: int = 128
    ):
\
\
\
\
\

        super().__init__()
        self.num_erosion_levels = num_erosion_levels
        self.input_channels = input_channels
        self.topo_dim = topo_dim


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


                nn.Linear(256, topo_dim)
            )
            self.topo_encoders.append(encoder)


        self.topo_fusion = nn.Sequential(
            nn.Linear(topo_dim * num_erosion_levels, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, topo_dim)
        )

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, list]:
\
\
\
\
\
\
\
\
\

        topo_features = []


        for i, (erosion, encoder) in enumerate(
                zip(self.erosion_layers, self.topo_encoders)
        ):

            eroded_feat = erosion(feat)


            topo_feat = encoder(eroded_feat)
            topo_features.append(topo_feat)


        topo_concat = torch.cat(topo_features, dim=1)


        topo_final = self.topo_fusion(topo_concat)

        return F.normalize(topo_final, dim=1), topo_features


class CBAM(nn.Module):


    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()


        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )


        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        ca = self.channel_attention(x) * x


        avg_out = torch.mean(ca, dim=1, keepdim=True)
        max_out = torch.max(ca, dim=1, keepdim=True)[0]
        sa = self.spatial_attention(
            torch.cat([avg_out, max_out], dim=1)
        ) * ca

        return sa


class ImprovedMoCoV3WithTopoSideline(nn.Module):
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
            topo_weight: float = 0.35,
            num_erosion_levels: int = 4,
            device: Optional[torch.device] = None
    ):
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


        self.encoder_q = models.resnet50(weights="IMAGENET1K_V1")
        resnet_dim = self.encoder_q.fc.in_features


        self.encoder_q = nn.Sequential(*list(self.encoder_q.children())[:-2])


        self.cbam_q = CBAM(resnet_dim)


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


        self.visual_mlp_q = nn.Sequential(
            nn.Linear(resnet_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim)
        )


        self.encoder_k = models.resnet50(weights="IMAGENET1K_V1")
        self.encoder_k = nn.Sequential(*list(self.encoder_k.children())[:-2])
        self.cbam_k = CBAM(resnet_dim)

        self.visual_mlp_k = nn.Sequential(
            nn.Linear(resnet_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim)
        )


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


        self.register_buffer("visual_queue", torch.randn(feature_dim, K))
        self.visual_queue = F.normalize(self.visual_queue, dim=0)


        self.register_buffer("topo_queue", torch.randn(topo_dim, K))
        self.topo_queue = F.normalize(self.topo_queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


        self._init_key_encoder()

        self.to(self.device)

    def _init_key_encoder(self):


        for pq, pk in zip(
                self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            pk.data.copy_(pq.data)
            pk.requires_grad = False


        for pq, pk in zip(self.cbam_q.parameters(), self.cbam_k.parameters()):
            pk.data.copy_(pq.data)
            pk.requires_grad = False


        for pq, pk in zip(
                self.visual_mlp_q.parameters(), self.visual_mlp_k.parameters()
        ):
            pk.data.copy_(pq.data)
            pk.requires_grad = False


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

        img1, img2 = img1.to(self.device), img2.to(self.device)


        feat_q = self.encoder_q(img1)
        feat_q_cbam = self.cbam_q(feat_q)


        visual_q = self.avgpool(feat_q_cbam).flatten(1)
        visual_q = self.visual_mlp_q(visual_q)
        visual_q = F.normalize(visual_q, dim=1)


        topo_q, _ = self.topo_extractor_q(feat_q_cbam)


        with torch.no_grad():
            self.momentum_update_key_encoder()

            feat_k = self.encoder_k(img2)
            feat_k_cbam = self.cbam_k(feat_k)


            visual_k = self.avgpool(feat_k_cbam).flatten(1)
            visual_k = self.visual_mlp_k(visual_k)
            visual_k = F.normalize(visual_k, dim=1)


            topo_k, _ = self.topo_extractor_k(feat_k_cbam)

        return visual_q, visual_k, topo_q, topo_k

    @torch.no_grad()
    def momentum_update_key_encoder(self):


        for pq, pk in zip(
                self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            pk.data.mul_(self.m).add_(pq.data, alpha=1 - self.m)


        for pq, pk in zip(self.cbam_q.parameters(), self.cbam_k.parameters()):
            pk.data.mul_(self.m).add_(pq.data, alpha=1 - self.m)


        for pq, pk in zip(
                self.visual_mlp_q.parameters(), self.visual_mlp_k.parameters()
        ):
            pk.data.mul_(self.m).add_(pq.data, alpha=1 - self.m)


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

        N = visual_q.shape[0]


        visual_l_pos = torch.einsum('nc,nc->n', visual_q, visual_k).unsqueeze(-1)
        visual_l_neg = torch.einsum(
            'nc,ck->nk', visual_q, self.visual_queue.clone().detach()
        )

        visual_logits = torch.cat([visual_l_pos, visual_l_neg], dim=1) / self.T
        labels = torch.zeros(N, dtype=torch.long).to(visual_q.device)
        visual_loss = F.cross_entropy(visual_logits, labels)


        topo_l_pos = torch.einsum('nc,nc->n', topo_q, topo_k).unsqueeze(-1)
        topo_l_neg = torch.einsum(
            'nc,ck->nk', topo_q, self.topo_queue.clone().detach()
        )

        topo_logits = torch.cat([topo_l_pos, topo_l_neg], dim=1) / self.T
        topo_loss = F.cross_entropy(topo_logits, labels)


        total_loss = (1 - self.topo_weight) * visual_loss + self.topo_weight * topo_loss


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
\
\
\
\
\
\

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
