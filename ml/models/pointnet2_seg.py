"""Segmentação de nuvens de pontos no estilo PointNet++.

Esta implementação usa somente operações de PyTorch (``cdist``, amostragem
hierárquica e propagação por interpolação). Assim, PyTorch Geometric continua
sendo opcional. A arquitetura não tenta ser um clone bit a bit do repositório
original: ela preserva a ideia central do PointNet++ e é simples de integrar
ao pipeline deste projeto.
"""

from __future__ import annotations

from typing import Any

from ..config import PointNet2Config
from ..dependencies import MissingOptionalDependency, require_torch, torch as _torch


if _torch is not None:  # pragma: no cover - covered in environments with torch
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F

    def _index_points(points: Tensor, indices: Tensor) -> Tensor:
        """Gather ``points[B,N,C]`` with indices ``[B,S]`` or ``[B,S,K]``."""

        batch = torch.arange(points.shape[0], device=points.device).view(-1, 1, 1)
        if indices.ndim == 2:
            batch = batch[:, :, 0]
        return points[batch, indices]


    def _farthest_point_sample(xyz: Tensor, count: int) -> Tensor:
        """Deterministic farthest-point sampling for ``xyz[B,N,3]``.

        The loop is intentionally small (the default uses 256 and 64 points),
        and the implementation has no custom CUDA extension, which keeps CPU
        smoke tests and packaged applications portable.
        """

        batch_size, num_points, _ = xyz.shape
        count = max(1, min(int(count), num_points))
        if count == num_points:
            return torch.arange(num_points, device=xyz.device).expand(batch_size, -1)

        sampled = torch.zeros((batch_size, count), dtype=torch.long, device=xyz.device)
        distances = torch.full((batch_size, num_points), float("inf"), device=xyz.device)
        # Use the point nearest to the centroid as a stable starting point.
        centroid = xyz.mean(dim=1, keepdim=True)
        farthest = ((xyz - centroid) ** 2).sum(dim=-1).argmin(dim=1)
        batch_index = torch.arange(batch_size, device=xyz.device)
        for step in range(count):
            sampled[:, step] = farthest
            current = xyz[batch_index, farthest].unsqueeze(1)
            distances = torch.minimum(distances, ((xyz - current) ** 2).sum(dim=-1))
            farthest = distances.max(dim=1).indices
        return sampled


    class _SharedMLP(nn.Module):
        def __init__(self, in_channels: int, channels: tuple[int, ...]) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            previous = in_channels
            for channel in channels:
                layers.extend(
                    [
                        nn.Conv2d(previous, channel, kernel_size=1, bias=False),
                        nn.BatchNorm2d(channel),
                        nn.ReLU(inplace=True),
                    ]
                )
                previous = channel
            self.layers = nn.Sequential(*layers)

        def forward(self, values: Tensor) -> Tensor:
            return self.layers(values)


    class _SetAbstraction(nn.Module):
        """Sampling + local PointNet block used by the encoder."""

        def __init__(
            self,
            in_channels: int,
            out_channels: tuple[int, ...],
            points: int,
            neighbors: int,
        ) -> None:
            super().__init__()
            self.points = int(points)
            self.neighbors = int(neighbors)
            # Relative xyz is always appended to point features.
            self.mlp = _SharedMLP(in_channels + 3, out_channels)

        def forward(self, xyz: Tensor, features: Tensor | None) -> tuple[Tensor, Tensor]:
            # xyz [B,N,3], features [B,C,N]
            indices = _farthest_point_sample(xyz, self.points)
            sampled_xyz = _index_points(xyz, indices)
            count = min(self.neighbors, xyz.shape[1])
            distances = torch.cdist(sampled_xyz, xyz)
            neighbor_indices = distances.topk(count, dim=-1, largest=False).indices
            grouped_xyz = _index_points(xyz, neighbor_indices) - sampled_xyz.unsqueeze(2)
            grouped_xyz = grouped_xyz.permute(0, 3, 1, 2).contiguous()
            if features is None:
                grouped = grouped_xyz
            else:
                # Convert [B,C,N] -> [B,N,C] before gathering.
                grouped_features = _index_points(features.transpose(1, 2), neighbor_indices)
                grouped_features = grouped_features.permute(0, 3, 1, 2).contiguous()
                grouped = torch.cat([grouped_xyz, grouped_features], dim=1)
            encoded = self.mlp(grouped).max(dim=-1).values
            return sampled_xyz, encoded


    def _interpolate_features(
        source_xyz: Tensor,
        source_features: Tensor,
        target_xyz: Tensor,
    ) -> Tensor:
        """Three-nearest-neighbour inverse-distance interpolation."""

        count = min(3, source_xyz.shape[1])
        distances = torch.cdist(target_xyz, source_xyz)
        nearest_distance, nearest_index = distances.topk(count, dim=-1, largest=False)
        weights = 1.0 / (nearest_distance + 1e-8)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        grouped = _index_points(source_features.transpose(1, 2), nearest_index)
        grouped = grouped * weights.unsqueeze(-1)
        return grouped.sum(dim=2).transpose(1, 2).contiguous()


    class PointNet2Segmentation(nn.Module):
        """PointNet++-style semantic segmentation network.

        Args:
            config: :class:`~ml.config.PointNet2Config` or mapping accepted by
                ``PointNet2Config.from_mapping``.

        Input accepts either ``[B, N, C]`` or ``[B, C, N]``. The first three
        channels must be XYZ; additional channels (for example intensity) are
        passed through the local PointNet blocks. Output is ``[B, N, classes]``.
        """

        def __init__(self, config: PointNet2Config | dict[str, Any] | None = None, **kwargs: Any) -> None:
            super().__init__()
            if config is None:
                self.config = PointNet2Config.from_mapping(kwargs)
            elif isinstance(config, PointNet2Config):
                if kwargs:
                    self.config = PointNet2Config.from_mapping(config.to_dict(), **kwargs)
                else:
                    self.config = config
            else:
                self.config = PointNet2Config.from_mapping(config, **kwargs)
            width = self.config.hidden_channels
            self.sa1 = _SetAbstraction(
                self.config.in_channels,
                (width, width, width * 2),
                self.config.sa1_points,
                self.config.neighbors,
            )
            self.sa2 = _SetAbstraction(
                width * 2,
                (width * 2, width * 2, width * 4),
                self.config.sa2_points,
                self.config.neighbors,
            )
            self.fp2 = nn.Sequential(
                nn.Conv1d(width * 4 + width * 2, width * 2, 1, bias=False),
                nn.BatchNorm1d(width * 2),
                nn.ReLU(inplace=True),
            )
            self.fp1 = nn.Sequential(
                nn.Conv1d(width * 2 + self.config.in_channels, width, 1, bias=False),
                nn.BatchNorm1d(width),
                nn.ReLU(inplace=True),
            )
            self.head = nn.Sequential(
                nn.Conv1d(width + self.config.in_channels, width, 1, bias=False),
                nn.BatchNorm1d(width),
                nn.ReLU(inplace=True),
                nn.Dropout(self.config.dropout),
                nn.Conv1d(width, self.config.num_classes, 1),
            )

        def forward(self, points: Tensor) -> Tensor:
            if points.ndim != 3:
                raise ValueError("A entrada deve ter formato [B,N,C] ou [B,C,N]")
            if points.shape[-1] == self.config.in_channels:
                values = points
            elif points.shape[1] == self.config.in_channels:
                values = points.transpose(1, 2)
            else:
                raise ValueError(
                    f"A entrada deve conter {self.config.in_channels} canais; "
                    f"formato recebido: {tuple(points.shape)}"
                )
            if values.shape[-1] < 3:
                raise ValueError("São necessários pelo menos os canais XYZ")
            xyz = values[..., :3].contiguous()
            features = values.transpose(1, 2).contiguous()
            xyz1, f1 = self.sa1(xyz, features)
            xyz2, f2 = self.sa2(xyz1, f1)
            f2_up = _interpolate_features(xyz2, f2, xyz1)
            f2_up = self.fp2(torch.cat([f2_up, f1], dim=1))
            f1_up = _interpolate_features(xyz1, f2_up, xyz)
            f1_up = self.fp1(torch.cat([f1_up, features], dim=1))
            return self.head(torch.cat([f1_up, features], dim=1)).transpose(1, 2).contiguous()

        def predict(self, points: Tensor) -> tuple[Tensor, Tensor]:
            """Return class IDs and confidence per point in evaluation mode."""

            was_training = self.training
            self.eval()
            with torch.no_grad():
                logits = self(points)
                probabilities = logits.softmax(dim=-1)
                confidence, labels = probabilities.max(dim=-1)
            if was_training:
                self.train()
            return labels, confidence

else:

    class PointNet2Segmentation:  # pragma: no cover - exercised without torch
        """Placeholder that keeps imports safe in geometry-only installs."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise MissingOptionalDependency("torch", "instanciar o modelo PointNet++")
