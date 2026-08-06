from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class Encoder3D(nn.Module):
    def __init__(self, latent_channels: int):
        super().__init__()
        widths = (16, 24, latent_channels)
        self.net = nn.Sequential(
            nn.Conv3d(2, widths[0], 3, padding=1),
            nn.InstanceNorm3d(widths[0], affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(widths[0], widths[1], 4, stride=2, padding=1),
            nn.InstanceNorm3d(widths[1], affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(widths[1], widths[2], 4, stride=2, padding=1),
            nn.InstanceNorm3d(widths[2], affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(widths[2], widths[2], 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class ConvLSTMCell3D(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv3d(input_channels + hidden_channels, hidden_channels * 4, 3, padding=1)

    def forward(
        self, value: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            shape = (value.shape[0], self.hidden_channels, *value.shape[2:])
            hidden = value.new_zeros(shape)
            cell = value.new_zeros(shape)
        else:
            hidden, cell = state
        input_gate, forget_gate, output_gate, candidate = self.gates(
            torch.cat([value, hidden], dim=1)
        ).chunk(4, dim=1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        output_gate = torch.sigmoid(output_gate)
        candidate = torch.tanh(candidate)
        cell = forget_gate * cell + input_gate * candidate
        hidden = output_gate * torch.tanh(cell)
        return hidden, cell


class LocalNeuralField(nn.Module):
    def __init__(self, latent_channels: int, hidden: int = 64, layers: int = 5):
        super().__init__()
        modules: list[nn.Module] = []
        input_width = latent_channels + 3
        for layer in range(layers):
            output_width = 1 if layer == layers - 1 else hidden
            linear = nn.Linear(input_width, output_width)
            if layer == 0:
                nn.init.uniform_(linear.weight, -1.0 / input_width, 1.0 / input_width)
            modules.append(linear)
            input_width = hidden
        self.layers = nn.ModuleList(modules)

    def forward(self, latent: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        grid = coords[:, :, None, None, :]
        local = F.grid_sample(latent, grid, mode="bilinear", padding_mode="border", align_corners=True)
        local = local[:, :, :, 0, 0].transpose(1, 2)
        value = torch.cat([coords, local], dim=-1)
        for index, layer in enumerate(self.layers):
            value = layer(value)
            if index != len(self.layers) - 1:
                value = torch.sin(30.0 * value)
        return torch.tanh(value[..., 0])


def temporal_encoding(interval: torch.Tensor, order: int, scale_days: float) -> torch.Tensor:
    normalized = interval / scale_days
    frequencies = (2.0 ** torch.arange(order, device=interval.device, dtype=interval.dtype)) * math.pi
    angles = normalized[:, None] * frequencies[None, :]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


class DeepGrowthLite(nn.Module):
    """Memory-bounded reproduction of the released method's core components."""

    def __init__(
        self,
        latent_channels: int = 24,
        temporal_order: int = 6,
        decoder_hidden: int = 64,
        decoder_layers: int = 5,
        time_scale_days: float = 2190.0,
    ):
        super().__init__()
        self.temporal_order = temporal_order
        self.time_scale_days = time_scale_days
        self.encoder = Encoder3D(latent_channels)
        self.recurrent = ConvLSTMCell3D(latent_channels + temporal_order * 2, latent_channels)
        self.decoder = LocalNeuralField(latent_channels, decoder_hidden, decoder_layers)

    def _with_time(self, latent: torch.Tensor, interval: torch.Tensor) -> torch.Tensor:
        encoded = temporal_encoding(interval, self.temporal_order, self.time_scale_days)
        encoded = encoded[:, :, None, None, None].expand(-1, -1, *latent.shape[2:])
        return torch.cat([latent, encoded], dim=1)

    def encode(self, observed: torch.Tensor, days: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        first = self.encoder(observed[:, 0])
        second = self.encoder(observed[:, 1])
        state = self.recurrent(self._with_time(first, days[:, 1] - days[:, 0]), None)
        future, _ = self.recurrent(self._with_time(second, days[:, 2] - days[:, 1]), state)
        return first, second, future

    def forward(
        self, observed: torch.Tensor, days: torch.Tensor, coords: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        first, second, future = self.encode(observed, days)
        reconstructed = torch.stack(
            [self.decoder(first, coords), self.decoder(second, coords)], dim=1
        )
        predicted = self.decoder(future, coords)
        regularization = torch.stack(
            [first.square().mean(), second.square().mean(), future.square().mean()]
        ).mean()
        return reconstructed, predicted, regularization

    def predict_sdf_grid(
        self, observed: torch.Tensor, days: torch.Tensor, shape: tuple[int, int, int], chunk: int
    ) -> torch.Tensor:
        _, _, future = self.encode(observed, days)
        depth, height, width = shape
        z, y, x = torch.meshgrid(
            torch.linspace(-1, 1, depth, device=observed.device),
            torch.linspace(-1, 1, height, device=observed.device),
            torch.linspace(-1, 1, width, device=observed.device),
            indexing="ij",
        )
        coords = torch.stack([x, y, z], dim=-1).reshape(1, -1, 3).expand(observed.shape[0], -1, -1)
        outputs = [self.decoder(future, coords[:, start : start + chunk]) for start in range(0, coords.shape[1], chunk)]
        return torch.cat(outputs, dim=1).reshape(observed.shape[0], depth, height, width)
