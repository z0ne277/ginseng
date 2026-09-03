"""Model heads for the image-only self-supervised baselines."""

from __future__ import annotations

import copy

import torch
from torch import nn


class ProjectionMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        final_batch_norm_affine: bool = True,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=False),
            nn.BatchNorm1d(output_dim, affine=final_batch_norm_affine),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class PredictionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, projections: torch.Tensor) -> torch.Tensor:
        return self.layers(projections)


class SimSiamModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        *,
        feature_dim: int,
        projection_dim: int = 2048,
        projection_hidden_dim: int = 2048,
        prediction_hidden_dim: int = 512,
    ):
        super().__init__()
        self.encoder = encoder
        self.projector = ProjectionMLP(
            feature_dim,
            projection_hidden_dim,
            projection_dim,
            final_batch_norm_affine=False,
        )
        self.predictor = PredictionMLP(projection_dim, prediction_hidden_dim)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def forward(self, first: torch.Tensor, second: torch.Tensor):
        projection_first = self.projector(self.encode(first))
        projection_second = self.projector(self.encode(second))
        prediction_first = self.predictor(projection_first)
        prediction_second = self.predictor(projection_second)
        return (
            prediction_first,
            prediction_second,
            projection_first.detach(),
            projection_second.detach(),
        )


class VICRegModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        *,
        feature_dim: int,
        projection_dim: int = 2048,
        projection_hidden_dim: int = 4096,
    ):
        super().__init__()
        self.encoder = encoder
        self.projector = ProjectionMLP(
            feature_dim,
            projection_hidden_dim,
            projection_dim,
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def forward(self, first: torch.Tensor, second: torch.Tensor):
        return (
            self.projector(self.encode(first)),
            self.projector(self.encode(second)),
        )


class DinoHead(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.Linear(bottleneck_dim, output_dim, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected = nn.functional.normalize(self.mlp(features), dim=-1)
        return self.last_layer(projected)


@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, momentum: float) -> None:
    if not 0.0 <= momentum <= 1.0:
        raise ValueError("EMA momentum must be in [0, 1]")
    target_parameters = dict(target.named_parameters())
    online_parameters = dict(online.named_parameters())
    if target_parameters.keys() != online_parameters.keys():
        raise ValueError("EMA modules must have matching parameters")
    for name, target_parameter in target_parameters.items():
        target_parameter.mul_(momentum).add_(
            online_parameters[name].detach(),
            alpha=1.0 - momentum,
        )
    target_buffers = dict(target.named_buffers())
    online_buffers = dict(online.named_buffers())
    if target_buffers.keys() != online_buffers.keys():
        raise ValueError("EMA modules must have matching buffers")
    for name, target_buffer in target_buffers.items():
        target_buffer.copy_(online_buffers[name])


@torch.no_grad()
def update_dino_center(
    center: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    momentum: float,
) -> torch.Tensor:
    if center.ndim != 2 or center.shape[0] != 1:
        raise ValueError("DINO center must have shape [1, output_dim]")
    if teacher_logits.ndim != 2 or teacher_logits.shape[1] != center.shape[1]:
        raise ValueError("teacher logits and DINO center dimensions do not match")
    batch_center = teacher_logits.detach().mean(dim=0, keepdim=True)
    return center.detach().mul(momentum).add(batch_center, alpha=1.0 - momentum)


def clone_frozen(module: nn.Module) -> nn.Module:
    """Create the initial DINO teacher without sharing trainable parameters."""
    target = copy.deepcopy(module)
    target.requires_grad_(False)
    target.eval()
    return target


class DinoModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        *,
        feature_dim: int,
        output_dim: int = 16384,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
    ):
        super().__init__()
        self.student_encoder = encoder
        self.student_head = DinoHead(
            input_dim=feature_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
        )
        self.teacher_encoder = clone_frozen(self.student_encoder)
        self.teacher_head = clone_frozen(self.student_head)
        self.register_buffer("center", torch.zeros(1, output_dim))

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.student_encoder(images)

    def forward(self, views):
        if len(views) < 2:
            raise ValueError("DINO requires at least two global views")
        student_logits = [
            self.student_head(self.student_encoder(view)) for view in views
        ]
        with torch.no_grad():
            teacher_logits = [
                self.teacher_head(self.teacher_encoder(view)) for view in views[:2]
            ]
        return student_logits, teacher_logits

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        ema_update(self.teacher_encoder, self.student_encoder, momentum)
        ema_update(self.teacher_head, self.student_head, momentum)

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher_encoder.eval()
        self.teacher_head.eval()
        return self
