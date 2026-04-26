"""ResNet-1D + RR-feature head for ECG beat classification.

Small enough to run on CPU comfortably (<300k params, <5 ms/beat on a laptop).
The morphology branch is a stack of residual 1D conv blocks with global average
pooling; the RR features (pre/post RR ratios) are concatenated to the pooled
embedding before the final fully-connected classifier head — same feature mix
as the LR baseline, just with a learned representation instead of raw windows.
"""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)


class ResNet1D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        samples_per_lead: int = 180,
        rr_features: int = 2,
        n_classes: int = 3,
        channels: list[int] | None = None,
        blocks_per_stage: int = 2,
    ) -> None:
        super().__init__()
        channels = channels or [32, 64, 128]
        self.in_channels = in_channels
        self.samples_per_lead = samples_per_lead

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, channels[0], kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        stages: list[nn.Module] = []
        prev = channels[0]
        for stage_i, ch in enumerate(channels):
            for block_i in range(blocks_per_stage):
                stride = 2 if (stage_i > 0 and block_i == 0) else 1
                stages.append(ResidualBlock1D(prev, ch, stride=stride))
                prev = ch
        self.stages = nn.Sequential(*stages)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.rr_norm = nn.BatchNorm1d(rr_features)
        self.head = nn.Sequential(
            nn.Linear(channels[-1] + rr_features, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, morph: torch.Tensor, rr: torch.Tensor) -> torch.Tensor:
        x = self.stem(morph)
        x = self.stages(x)
        x = self.gap(x).squeeze(-1)
        rr = self.rr_norm(rr)
        x = torch.cat([x, rr], dim=1)
        return self.head(x)
