"""ResNet-1D + RR-feature head for ECG beat classification.

Small enough to run on CPU comfortably (<300k params, <5 ms/beat on a laptop).
The morphology branch is a stack of residual 1D conv blocks with global average
pooling; the RR features (pre/post RR ratios) are concatenated to the pooled
embedding before the final fully-connected classifier head — same feature mix
as the LR baseline, just with a learned representation instead of raw windows.

v0.4 adds two opt-in upgrades:

- :class:`SEBlock1D` — Squeeze-and-Excitation channel attention. Enable via
  ``ResNet1D(use_se=True)``. Bit-identical to v0.3 when ``use_se=False``.
- ``stem="inception"`` — multi-scale 1D conv stem (parallel kernels 3/7/15).
  Helps capture P-wave / QRS / T-wave at their native scales. Default
  ``stem="conv"`` is unchanged.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

StemKind = Literal["conv", "inception"]


class SEBlock1D(nn.Module):
    """Squeeze-and-Excitation channel attention for 1D conv features.

    Reference: Hu, Shen & Sun, "Squeeze-and-Excitation Networks," CVPR 2018.
    Adapted to 1D: global-average-pool over the time axis to produce a
    per-channel descriptor, then a 2-layer FC bottleneck (with reduction) +
    sigmoid produces per-channel scaling weights that re-weight the
    feature map. Adds ~few-percent param overhead.
    """

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        s = x.mean(dim=2)                 # squeeze: (B, C)
        s = torch.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))    # excitation: (B, C) in (0, 1)
        return x * s.unsqueeze(-1)        # scale: (B, C, T)


class InceptionStem1D(nn.Module):
    """Multi-scale conv stem with parallel kernels 3/7/15, concatenated to ``out_ch``.

    Each branch contributes ``out_ch // 3`` channels; any leftover goes to the
    longest-kernel branch. Output is BN + ReLU, with a stride-2 MaxPool to
    match the receptive-field reduction of the default conv stem.
    """

    def __init__(self, in_channels: int, out_ch: int) -> None:
        super().__init__()
        per = out_ch // 3
        leftover = out_ch - 2 * per
        self.b3 = nn.Conv1d(in_channels, per, kernel_size=3, stride=2, padding=1, bias=False)
        self.b7 = nn.Conv1d(in_channels, per, kernel_size=7, stride=2, padding=3, bias=False)
        self.b15 = nn.Conv1d(in_channels, leftover, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.b3(x), self.b7(x), self.b15(x)], dim=1)
        return self.pool(self.act(self.bn(x)))


class ResidualBlock1D(nn.Module):
    """1D residual block with optional Squeeze-Excitation attention.

    ``use_se=False`` (the default) keeps the v0.3 numerics exactly: the SE
    submodule is not constructed, so old state dicts continue to load.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, *, use_se: bool = False) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)
        self.se: nn.Module | None = SEBlock1D(out_ch) if use_se else None
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
        if self.se is not None:
            out = self.se(out)
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
        *,
        use_se: bool = False,
        stem: StemKind = "conv",
    ) -> None:
        super().__init__()
        channels = channels or [32, 64, 128]
        self.in_channels = in_channels
        self.samples_per_lead = samples_per_lead
        self.use_se = bool(use_se)
        self.stem_kind: StemKind = stem

        if stem == "inception":
            self.stem: nn.Module = InceptionStem1D(in_channels, channels[0])
        elif stem == "conv":
            self.stem = nn.Sequential(
                nn.Conv1d(in_channels, channels[0], kernel_size=15, stride=2, padding=7, bias=False),
                nn.BatchNorm1d(channels[0]),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            )
        else:
            raise ValueError(f"Unknown stem kind {stem!r}; expected 'conv' or 'inception'")

        stages: list[nn.Module] = []
        prev = channels[0]
        for stage_i, ch in enumerate(channels):
            for block_i in range(blocks_per_stage):
                stride = 2 if (stage_i > 0 and block_i == 0) else 1
                stages.append(ResidualBlock1D(prev, ch, stride=stride, use_se=use_se))
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

    # Convenience for explainability + uncertainty downstream — return the
    # penultimate (post-GAP, pre-head) embedding plus the head logits.
    def forward_embedding(self, morph: torch.Tensor, rr: torch.Tensor):
        x = self.stem(morph)
        x = self.stages(x)
        emb = self.gap(x).squeeze(-1)
        rr_n = self.rr_norm(rr)
        z = torch.cat([emb, rr_n], dim=1)
        logits = self.head(z)
        return emb, logits
