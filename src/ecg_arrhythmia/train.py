"""CLI for training ECG arrhythmia models."""

from __future__ import annotations

import argparse
from pathlib import Path

from .labels import get_scheme
from .training import train_baseline_lr, train_baseline_quick_smoke, train_resnet_1d


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ECG arrhythmia models on MIT-BIH.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to MIT-BIH dataset directory containing RECORDS.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/baseline"),
        help="Output directory for model artifacts (default: artifacts/baseline).",
    )
    parser.add_argument(
        "--model",
        choices=["baseline", "resnet"],
        default="baseline",
    )
    parser.add_argument(
        "--scheme",
        choices=["mitbih3", "aami5"],
        default="mitbih3",
        help="Label scheme. mitbih3 keeps the legacy 3-class N/V/a; aami5 is "
        "AAMI EC57 5-class (N/S/V/F/Q).",
    )
    parser.add_argument(
        "--leads",
        nargs="+",
        default=["MLII"],
        help="Lead name(s) to train on (default: MLII). Multi-lead concatenates morphology windows.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quick-smoke",
        action="store_true",
        help="Run a synthetic smoke-training path (no MIT-BIH data required).",
    )
    parser.add_argument("--epochs", type=int, default=30, help="ResNet only: epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="ResNet only: batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="ResNet only: learning rate.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="ResNet only: training device.",
    )

    # ----- v0.4 ResNet upgrades -----
    parser.add_argument(
        "--architecture",
        choices=["resnet", "cnn_transformer"],
        default="resnet",
        help="ResNet (default) or experimental CNN-Transformer hybrid.",
    )
    parser.add_argument("--use-se", action="store_true", help="Enable SE channel attention.")
    parser.add_argument(
        "--stem",
        choices=["conv", "inception"],
        default="conv",
        help="Conv stem (v0.3 default) or multi-scale Inception stem.",
    )
    parser.add_argument("--augment", action="store_true", help="Enable signal augmentation.")
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=0.0,
        help="Focal-loss γ (0 reproduces weighted CE).",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--balanced-sampler",
        action="store_true",
        help="Use a WeightedRandomSampler for class-balanced batches.",
    )
    parser.add_argument(
        "--two-stage",
        action="store_true",
        help="Two-stage schedule: balanced sampler + focal in stage 1, "
        "natural distribution + label-smoothing in stage 2.",
    )
    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=0.0,
        help="Mixup α (0 disables mixup; common values 0.1–0.4).",
    )
    parser.add_argument("--ema-decay", type=float, default=0.0, help="EMA decay (e.g. 0.999).")
    parser.add_argument("--grad-clip", type=float, default=0.0, help="Gradient-norm clip.")
    parser.add_argument("--warmup-epochs", type=int, default=0, help="Linear LR warmup epochs.")
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early-stopping patience (epochs of no val-F1 improvement).",
    )
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP even on CUDA.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--split",
        choices=["random", "ds1ds2"],
        default="random",
        help="Train/test split. ds1ds2 is the AAMI-canonical de Chazal split.",
    )
    parser.add_argument(
        "--exclude-paced-records",
        action="store_true",
        help="Drop MIT-BIH records 102/104/107/217 (paced). "
        "Standard practice for AAMI evaluation.",
    )

    args = parser.parse_args()
    scheme = get_scheme(args.scheme)

    if args.model == "baseline":
        if args.quick_smoke:
            out_dir = args.out_dir if "smoke" in str(args.out_dir) else Path("artifacts/smoke")
            artifacts = train_baseline_quick_smoke(out_dir, seed=args.seed)
        else:
            if args.data_dir is None:
                raise SystemExit("--data-dir is required unless --quick-smoke is set.")
            artifacts = train_baseline_lr(
                args.data_dir,
                args.out_dir,
                seed=args.seed,
                leads=tuple(args.leads),
                scheme=scheme,
                split_strategy=args.split,
                exclude_paced=args.exclude_paced_records,
            )
    else:  # resnet
        common_kwargs = dict(
            seed=args.seed,
            leads=tuple(args.leads),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            scheme=scheme,
            architecture=args.architecture,
            use_se=args.use_se,
            stem_kind=args.stem,
            augment=args.augment,
            focal_gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing,
            balanced_sampler=args.balanced_sampler,
            two_stage=args.two_stage,
            mixup_alpha=args.mixup_alpha,
            ema_decay=args.ema_decay,
            grad_clip=args.grad_clip,
            warmup_epochs=args.warmup_epochs,
            patience=args.patience,
            use_amp=False if args.no_amp else None,
            num_workers=args.num_workers,
            split_strategy=args.split,
            exclude_paced=args.exclude_paced_records,
        )
        if args.quick_smoke:
            out_dir = args.out_dir if "smoke" in str(args.out_dir) else Path("artifacts/resnet_smoke")
            artifacts = train_resnet_1d(
                Path("."),
                out_dir,
                quick_smoke=True,
                **common_kwargs,
            )
        else:
            if args.data_dir is None:
                raise SystemExit("--data-dir is required unless --quick-smoke is set.")
            out_dir = args.out_dir if args.out_dir != Path("artifacts/baseline") else Path("artifacts/resnet")
            artifacts = train_resnet_1d(
                args.data_dir,
                out_dir,
                quick_smoke=False,
                **common_kwargs,
            )

    print(f"Saved model:   {artifacts.model_path}")
    print(f"Saved scaler:  {artifacts.scaler_path}")
    print(f"Saved metrics: {artifacts.metrics_path}")
    print(f"Saved split:   {artifacts.split_path}")


if __name__ == "__main__":
    main()
