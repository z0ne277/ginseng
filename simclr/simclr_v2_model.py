import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SimCLR_V2(nn.Module):

    def __init__(self, base_model='resnet50', feature_dim=128, hidden_dim=2048,
                 temperature=0.07, device=None):
        super(SimCLR_V2, self).__init__()

        self.temperature = temperature
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.projection_mid_dim = hidden_dim // 2
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if base_model == 'resnet50':
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif base_model == 'resnet101':
            backbone = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unknown base model: {base_model}")

        self.backbone_dim = backbone.fc.in_features  # 2048
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # Keep projection head compatible with existing training checkpoints:
        # Linear -> BN -> ReLU -> Linear -> BN -> ReLU -> Linear
        self.projection_head = nn.Sequential(
            nn.Linear(self.backbone_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.projection_mid_dim),
            nn.BatchNorm1d(self.projection_mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.projection_mid_dim, feature_dim)
        )

        self.to(self.device)

    @property
    def projector(self):
        # Backward-compatible accessor for older code paths.
        return self.projection_head

    def forward(self, x):
        x = x.to(self.device)
        N = x.shape[0] // 2


        features = self.backbone(x)
        features = torch.flatten(features, 1)


        projections = self.projection_head(features)

        projections = F.normalize(projections, dim=1)

        z_i = projections[:N]
        z_j = projections[N:]

        loss = self.nt_xent_loss(z_i, z_j)

        return loss, z_i, z_j

    def nt_xent_loss(self, z_i, z_j):
        N = z_i.shape[0]


        z = torch.cat([z_i, z_j], dim=0)  # [2N, feature_dim]

        similarity_matrix = torch.matmul(z, z.T)  # [2N, 2N]


        similarity_matrix = similarity_matrix / self.temperature  # [2N, 2N]

        pos_mask = torch.zeros(2 * N, 2 * N, dtype=torch.bool).to(self.device)
        pos_mask[:N, N:] = torch.eye(N, dtype=torch.bool).to(self.device)
        pos_mask[N:, :N] = torch.eye(N, dtype=torch.bool).to(self.device)

        diag_mask = torch.eye(2 * N, dtype=torch.bool).to(self.device)

        pos_logits = similarity_matrix[pos_mask].view(2 * N, 1)

        neg_mask = ~(diag_mask | pos_mask)
        neg_logits_list = []
        for i in range(2 * N):
            neg_logits_list.append(similarity_matrix[i, neg_mask[i]])
        neg_logits = torch.stack(neg_logits_list, dim=0)

        logits = torch.cat([pos_logits, neg_logits], dim=1)

        labels = torch.zeros(2 * N, dtype=torch.long).to(self.device)

        loss = F.cross_entropy(logits, labels)

        return loss

    @torch.no_grad()
    def get_features(self, x):
        x = x.to(self.device)
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        features = self.projection_head(features)
        features = F.normalize(features, dim=1)
        return features

    def get_backbone_features(self, x):
        x = x.to(self.device)
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        return features
