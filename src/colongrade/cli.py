"""The ``colongrade`` command."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Settings, load_settings
from .data.images import CONTRAST_MODES
from .experiment import MODELS, ExperimentConfig, predict_file, run


def _device(settings: Settings) -> str:
    if settings.device != "auto":
        return settings.device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def cmd_synth(args, settings) -> int:
    from .data.synthetic import write_dataset

    out = Path(args.out or settings.data_dir / "synthetic_limuc")
    write_dataset(out, n_patients=args.patients, images_per_patient=args.per_patient, size=args.size,
                  seed=settings.seed)
    print(f"wrote synthetic images to {out}")
    return 0


def cmd_manifest(args, settings) -> int:
    from .data.manifest import assign_folds, build_manifest, check_manifest, save_manifest

    root = Path(args.root)
    df = assign_folds(build_manifest(root), n_folds=settings.folds, seed=settings.seed)
    out = Path(args.out or root / "manifest.csv")
    save_manifest(df, out)
    print(json.dumps(check_manifest(df), indent=2, default=str))
    print(f"wrote {out}")
    return 0


def _config(args, settings) -> ExperimentConfig:
    return ExperimentConfig(model=args.model, contrast=args.contrast, image_size=args.size or settings.image_size,
                            folds=settings.folds, seed=settings.seed, gan_per_grade=args.gan_per_grade,
                            gan_epochs=args.gan_epochs, cnn_epochs=args.epochs, pretrained=args.pretrained,
                            device=_device(settings), n_boot=args.n_boot)


def cmd_train(args, settings) -> int:
    from .data.manifest import load_manifest

    root = Path(args.root)
    manifest = load_manifest(args.manifest or root / "manifest.csv")
    cfg = _config(args, settings)
    name = args.name or f"{cfg.model}_{cfg.contrast}{'_gan' if cfg.gan_per_grade else ''}"
    result = run(manifest, root, cfg, Path(settings.out_dir) / name)
    print(json.dumps({"cv": {k: v for k, v in result["cv"].items() if k != "folds"},
                      "test": {k: result["test"][k] for k in ("qwk", "qwk_ci95", "macro_f1", "accuracy",
                                                               "per_grade_recall", "mae_grades", "ece")}}, indent=2))
    return 0


def cmd_predict(args, settings) -> int:
    for image in args.image:
        print(json.dumps(predict_file(args.run, image)))
    return 0


def cmd_demo(args, settings) -> int:
    from .data.manifest import assign_folds, build_manifest
    from .data.synthetic import write_dataset

    out = Path(args.out or settings.out_dir / "demo")
    root = write_dataset(out / "images", n_patients=40, images_per_patient=12, size=64, seed=settings.seed)
    manifest = assign_folds(build_manifest(root), n_folds=5, seed=settings.seed)
    print(f"synthetic images: {len(manifest)} ({(manifest.split == 'test').sum()} test)")
    rows = []
    variants = [("features-multinomial", "none", 0), ("features-ordinal", "none", 0),
                ("features-ordinal", "clahe", 0)]
    if args.neural:
        variants += [("cnn-small", "none", 0), ("cnn-small", "none", 25)]
    for model, contrast, gan in variants:
        cfg = ExperimentConfig(model=model, contrast=contrast, image_size=64, folds=5, seed=settings.seed,
                               gan_per_grade=gan, gan_epochs=args.gan_epochs, cnn_epochs=args.epochs,
                               device=_device(settings), n_boot=500)
        res = run(manifest, root, cfg, out / f"{model}_{contrast}{'_gan' if gan else ''}", log=lambda m: None)
        t = res["test"]
        rows.append({"model": model, "contrast": contrast, "gan_per_grade": gan,
                     "cv_qwk": f"{res['cv']['qwk_mean']:.3f} ± {res['cv']['qwk_sd']:.3f}",
                     "test_qwk": f"{t['qwk']:.3f} [{t['qwk_ci95'][0]:.3f}, {t['qwk_ci95'][1]:.3f}]",
                     "test_macro_f1": round(t["macro_f1"], 3), "test_acc": round(t["accuracy"], 3),
                     "active_auc": round(t.get("remission_vs_active_auc", float("nan")), 3),
                     "seconds": res["seconds"]})
        print(json.dumps(rows[-1], ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="colongrade", description="Mayo endoscopic score grading (research only).")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("synth", help="write a synthetic LIMUC-like image folder")
    p.add_argument("--out")
    p.add_argument("--patients", type=int, default=40)
    p.add_argument("--per-patient", type=int, default=12)
    p.add_argument("--size", type=int, default=64)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("manifest", help="list ALL images, read grades and splits, assign grouped folds")
    p.add_argument("root")
    p.add_argument("--out")
    p.set_defaults(func=cmd_manifest)

    p = sub.add_parser("train", help="cross-validate on trainval, then score the test set once")
    p.add_argument("root")
    p.add_argument("--manifest")
    p.add_argument("--model", choices=MODELS, default="features-ordinal")
    p.add_argument("--contrast", choices=CONTRAST_MODES, default="none")
    p.add_argument("--size", type=int)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--pretrained", action="store_true", help="ImageNet weights for resnet backbones")
    p.add_argument("--gan-per-grade", type=int, default=0, help="generated images per grade (0 = off)")
    p.add_argument("--gan-epochs", type=int, default=30)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--name")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("predict", help="grade images with a trained run (research output only)")
    p.add_argument("run")
    p.add_argument("image", nargs="+")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("demo", help="offline demo on synthetic images")
    p.add_argument("--out")
    p.add_argument("--neural", action="store_true", help="add the small CNN and the GAN ablation (torch)")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--gan-epochs", type=int, default=20)
    p.set_defaults(func=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    return args.func(args, load_settings())


if __name__ == "__main__":
    raise SystemExit(main())
