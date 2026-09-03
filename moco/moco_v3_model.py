import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import copy


class MoCo_V3(nn.Module):
    def __init__(self, base_model='resnet50', feature_dim=256, K=4096, m=0.999, T=0.07, device=None):
        super(MoCo_V3, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if base_model == 'resnet50':
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif base_model == 'resnet101':
            backbone = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unknown base model: {base_model}")

        feature_dim_backbone = backbone.fc.in_features
        self.encoder_q = nn.Sequential(*list(backbone.children())[:-1])
        self.encoder_k = copy.deepcopy(self.encoder_q)

        self.proj_dim = feature_dim_backbone
        self.feature_dim = feature_dim

        self.mlp_q = nn.Sequential(
            nn.Linear(self.proj_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, feature_dim)
        )

        self.mlp_k = nn.Sequential(
            nn.Linear(self.proj_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, feature_dim)
        )

        self.register_buffer("queue", torch.randn(feature_dim, K).to(self.device))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long).to(self.device))

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        for param_q, param_k in zip(self.mlp_q.parameters(), self.mlp_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.to(self.device)

    def forward(self, img_q, img_k):
        img_q = img_q.to(self.device)
        img_k = img_k.to(self.device)

        q = self.encoder_q(img_q)  # [N, 2048, 1, 1]
        q = torch.flatten(q, 1)  # [N, 2048]
        q = self.mlp_q(q)  # [N, feature_dim]
        q = F.normalize(q, dim=1)  # [N, feature_dim]

        with torch.no_grad():
            self._momentum_update_key_encoder()

            k = self.encoder_k(img_k)  # [N, 2048, 1, 1]
            k = torch.flatten(k, 1)  # [N, 2048]
            k = self.mlp_k(k)  # [N, feature_dim]
            k = F.normalize(k, dim=1)  # [N, feature_dim]

        loss = self.contrastive_loss(q, k)

        with torch.no_grad():
            self._update_queue(k)

        return loss, q, k

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)

        for param_q, param_k in zip(self.mlp_q.parameters(), self.mlp_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)

    def contrastive_loss(self, q, k):
        N = q.shape[0]

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        labels = torch.zeros(N, dtype=torch.long).to(q.device)

        loss = F.cross_entropy(logits, labels)
        return loss

    @torch.no_grad()
    def _update_queue(self, k):
        batch_size = k.shape[0]
        ptr = int(self.queue_ptr)

        if ptr + batch_size <= self.K:
            self.queue[:, ptr:ptr + batch_size] = k.T
        else:
            num_first = self.K - ptr
            self.queue[:, ptr:] = k[:num_first].T
            self.queue[:, :batch_size - num_first] = k[num_first:].T

        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def get_features(self, x):
        x = x.to(self.device)
        feat = self.encoder_q(x)
        feat = torch.flatten(feat, 1)
        feat = self.mlp_q(feat)
        feat = F.normalize(feat, dim=1)
        return feat
