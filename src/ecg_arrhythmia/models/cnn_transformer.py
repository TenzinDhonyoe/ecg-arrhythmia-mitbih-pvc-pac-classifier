"""CNN-Transformer hybrid for ECG beat classification (experimental).

Architecture:

  morph (B, 1, 180)
    │
    ▼ SE-ResNet-1D stem + 2 stages   →   (B, 64, 23)
    │
    ▼ project to d_model=64           →   (B, 23, 64)
    │
    ▼ + sinusoidal positional encoding
    │
    ▼ 2-layer Transformer encoder     →   (B, 23, 64)
    │
    ▼ CLS token / mean-pool           →   (B, 64)
    │
    ▼ concat with LayerNorm(rr)       →   (B, 66)
    │
    ▼ MLP head                        →   (B, n_classes)

Why scoped down? With only ~70k MIT-BIH beats (and AAMI F/Q classes <1k each),
a deeper or wider transformer overfits and underperforms SE-ResNet. This is a
secondary, **experimental** model — useful as an architecture showcase, not as
the headline. Same ``(morph, rr) -> logits`` contract as :class:`ResNet1D`,
so it slots into ``ECGClassifier`` and ONNX export unchanged.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .resnet1d import ResidualBlock1D


class CNNTransformer1D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        samples_per_lead: int = 180,
        rr_features: int = 2,
        n_classes: int = 5,
        *,
        cnn_channels: list[int] | None = None,
        cnn_blocks_per_stage: int = 2,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
        use_se: bool = True,
        use_cls_token: bool = True,
    ) -> None:
        super().__init__()
        cnn_channels = cnn_channels or [32, 64]
        self.in_channels = in_channels
        self.samples_per_lead = samples_per_lead
        self.use_cls_token = bool(use_cls_token)

        # CNN stem identical in spirit to ResNet1D's first two stages.
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, cnn_channels[0], kernel_size=15,
                      stride=2, padding=7, bias=False),
            nn.BatchNorm1d(cnn_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        stages: list[nn.Module] = []
        prev = cnn_channels[0]
        for stage_i, ch in enumerate(cnn_channels):
            for block_i in range(cnn_blocks_per_stage):
                stride = 2 if (stage_i > 0 and block_i == 0) else 1
                stages.append(ResidualBlock1D(prev, ch, stride=stride, use_se=use_se))
                prev = ch
        self.stages = nn.Sequential(*stages)

        self.proj = nn.Conv1d(cnn_channels[-1], d_model, kernel_size=1, bias=False)

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.rr_norm = nn.LayerNorm(rr_features)
        self.head = nn.Sequential(
            nn.Linear(d_model + rr_features, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    @staticmethod
    def _sinusoidal_pe(length: int, dim: int, device, dtype) -> torch.Tensor:
        position = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32, device=device)
            * (-math.log(10000.0) / dim)
        )
        pe = torch.zeros(length, dim, dtype=torch.float32, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.to(dtype=dtype).unsqueeze(0)  # (1, T, D)

    def forward(self, morph: torch.Tensor, rr: torch.Tensor) -> torch.Tensor:
        x = self.stem(morph)
        x = self.stages(x)              # (B, C, T)
        x = self.proj(x)                # (B, d_model, T)
        x = x.transpose(1, 2)           # (B, T, d_model)

        if self.use_cls_token:
            cls = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1)        # (B, T+1, d_model)

        pe = self._sinusoidal_pe(x.shape[1], x.shape[2], x.device, x.dtype)
        x = x + pe

        x = self.encoder(x)
        # CLS token (when present) sits at position 0; else mean-pool the sequence.
        pooled = x[:, 0] if self.use_cls_token else x.mean(dim=1)

        rr_n = self.rr_norm(rr)
        z = torch.cat([pooled, rr_n], dim=1)
        return self.head(z)


__all__ = ["CNNTransformer1D"]
