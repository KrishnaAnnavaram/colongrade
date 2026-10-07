"""Cross-validation on the trainval folds, then ONE evaluation on the official test set.

Order of events:

1. For each fold k: train on the other folds, validate on fold k (never on the training images).
2. Summarize the folds (mean and SD of QWK and macro-F1).
3. Train the final model on all trainval images. A CNN trains for the median best epoch of the folds.
4. Score the official test set one time. Write the run folder and a model card.

The optional GAN ablation trains the conditional GAN on the training folds of each step only.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data.images import ImageConfig, load_image
from .evaluate import qwk, report
from .features import FEATURE_NAMES, extract, extract_many
from .models.baseline import GRADERS

MODELS = ("features-multinomial", "features-ordinal", "cnn-small", "cnn-resnet18", "cnn-resnet50")


@dataclass
class ExperimentConfig:
    model: str = "features-ordinal"
    contrast: str = "none"
    image_size: int = 224
    folds: int = 5
    seed: int = 13
    gan_per_grade: int = 0  # 0 = no GAN augmentation
    gan_epochs: int = 30
    cnn_epochs: int = 20
    pretrained: bool = False
    device: str = "cpu"
    n_boot: int = 1000

    def __post_init__(self):
        if self.model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}")


def load_images(paths, root: Path, img_cfg: ImageConfig) -> np.ndarray:
    return np.stack([load_image(root / p, img_cfg) for p in paths])


def _gan_extra(cfg: ExperimentConfig, images: np.ndarray, grades: np.ndarray):
    from .models.cgan import ConditionalGAN

    gan = ConditionalGAN(size=images.shape[1], epochs=cfg.gan_epochs, seed=cfg.seed, device=cfg.device)
    gan.fit(images, grades)
    fake_grades = np.repeat(np.arange(4), cfg.gan_per_grade)
    return gan.sample(fake_grades).astype(np.float32) / 255.0, fake_grades


def _grader(cfg: ExperimentConfig, epochs: int | None = None):
    if cfg.model.startswith("features-"):
        return GRADERS[cfg.model](seed=cfg.seed)
    from .models.cnn import CNNGrader

    return CNNGrader(backbone=cfg.model.split("-", 1)[1], pretrained=cfg.pretrained,
                     epochs=epochs or cfg.cnn_epochs, patience=4, seed=cfg.seed,
                     device=cfg.device)


def _fit_predict(cfg, train_x, train_y, eval_x, val_x=None, val_y=None, epochs=None):
    """Fit on (train_x, train_y), return (model, probabilities on eval_x). ``*_x`` are images."""
    if cfg.gan_per_grade:
        extra_x, extra_y = _gan_extra(cfg, train_x, train_y)
        train_x, train_y = np.concatenate([train_x, extra_x]), np.concatenate([train_y, extra_y])
    model = _grader(cfg, epochs)
    if cfg.model.startswith("features-"):
        model.fit(extract_many(train_x), train_y)
        return model, model.predict_proba(extract_many(eval_x))
    model.fit(train_x, train_y, val_x, val_y)  # final model: val_x is None, so a fixed number of epochs
    return model, model.predict_proba(eval_x)


def permutation_importance(model, x_feats: np.ndarray, y: np.ndarray, repeats: int = 3, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    base = qwk(y, model.predict_proba(x_feats).argmax(1))
    out = []
    for j, name in enumerate(FEATURE_NAMES):
        drops = []
        for _ in range(repeats):
            x = x_feats.copy()
            x[:, j] = rng.permutation(x[:, j])
            drops.append(base - qwk(y, model.predict_proba(x).argmax(1)))
        out.append({"feature": name, "qwk_drop": round(float(np.mean(drops)), 4)})
    return sorted(out, key=lambda r: -r["qwk_drop"])


def run(manifest: pd.DataFrame, root: str | Path, cfg: ExperimentConfig, out_dir: str | Path, log=print) -> dict:
    root, out = Path(root), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    img_cfg = ImageConfig(size=cfg.image_size, contrast=cfg.contrast)
    tv = manifest[manifest["split"] == "trainval"].reset_index(drop=True)
    test = manifest[manifest["split"] == "test"].reset_index(drop=True)
    if (tv["fold"] < 0).any():
        raise ValueError("assign folds first (colongrade manifest)")
    start = time.time()
    x_tv = load_images(tv["path"], root, img_cfg)
    y_tv = tv["mayo"].to_numpy()
    fold_reports, best_epochs = [], []
    for k in sorted(tv["fold"].unique()):
        tr, va = (tv["fold"] != k).to_numpy(), (tv["fold"] == k).to_numpy()
        model, proba = _fit_predict(cfg, x_tv[tr], y_tv[tr], x_tv[va], x_tv[va], y_tv[va])
        rep = report(y_tv[va], proba, n_boot=200, seed=cfg.seed)
        fold_reports.append({"fold": int(k), "n_val": int(va.sum()), "qwk": rep["qwk"], "macro_f1": rep["macro_f1"],
                             "accuracy": rep["accuracy"]})
        if hasattr(model, "best_epoch"):
            best_epochs.append(model.best_epoch + 1)
        log(f"fold {k}: QWK {rep['qwk']:.3f}  macro-F1 {rep['macro_f1']:.3f}  (n = {int(va.sum())})")
    qwks = [f["qwk"] for f in fold_reports]
    f1s = [f["macro_f1"] for f in fold_reports]
    cv = {"folds": fold_reports, "qwk_mean": float(np.mean(qwks)), "qwk_sd": float(np.std(qwks, ddof=1)),
          "macro_f1_mean": float(np.mean(f1s)), "macro_f1_sd": float(np.std(f1s, ddof=1))}
    final_epochs = int(np.median(best_epochs)) if best_epochs else None
    x_test = load_images(test["path"], root, img_cfg)
    y_test = test["mayo"].to_numpy()
    model, proba = _fit_predict(cfg, x_tv, y_tv, x_test, epochs=final_epochs)
    test_report = report(y_test, proba, n_boot=cfg.n_boot, seed=cfg.seed)
    result = {"config": asdict(cfg), "cv": cv, "test": test_report, "final_epochs": final_epochs,
              "images": {"trainval": len(tv), "test": len(test)}, "seconds": round(time.time() - start, 1)}
    if cfg.model.startswith("features-"):
        result["permutation_importance_test"] = permutation_importance(model, extract_many(x_test), y_test)[:10]
        joblib.dump(model, out / "model.joblib")
    else:
        import torch

        torch.save(model.model.state_dict(), out / "model.pt")
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "model_card.md").write_text(model_card(result), encoding="utf-8")
    pd.DataFrame({"path": test["path"], "mayo": y_test, "pred": proba.argmax(1),
                  **{f"p_mayo{k}": proba[:, k] for k in range(4)}}).to_csv(out / "test_predictions.csv", index=False)
    return result


def predict_file(run_dir: str | Path, image_path: str | Path) -> dict:
    run_dir = Path(run_dir)
    cfg = ExperimentConfig(**json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
    img = load_image(image_path, ImageConfig(size=cfg.image_size, contrast=cfg.contrast))
    if cfg.model.startswith("features-"):
        proba = joblib.load(run_dir / "model.joblib").predict_proba(extract(img)[None])[0]
    else:
        import torch

        from .models.cnn import CNNGrader, OrdinalCNN, build_backbone

        grader = CNNGrader(backbone=cfg.model.split("-", 1)[1], device="cpu")
        grader.model = OrdinalCNN(build_backbone(grader.backbone_name, False))
        grader.model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True))
        proba = grader.predict_proba(img[None])[0]
    return {"image": str(image_path), "predicted_mayo": int(proba.argmax()),
            "probabilities": [round(float(p), 4) for p in proba],
            "p_active_disease": round(float(proba[2:].sum()), 4),
            "note": "Research output only. Not a diagnosis. A clinician must review the image."}


def model_card(result: dict) -> str:
    cfg, cv, t = result["config"], result["cv"], result["test"]
    lines = [
        f"# Model card: {cfg['model']} (contrast {cfg['contrast']}, GAN images per grade {cfg['gan_per_grade']})", "",
        "## Intended use",
        "Research on automatic Mayo endoscopic subscore grading. NOT a medical device. NOT for diagnosis or",
        "treatment decisions. A qualified clinician must review every image and every grade.", "",
        "## Data",
        f"Trainval images: {result['images']['trainval']}, test images: {result['images']['test']}.",
        "The grade labels come from the dataset annotators. Rater disagreement between neighbour grades is common.", "",
        "## Results",
        f"Cross-validation QWK {cv['qwk_mean']:.3f} (SD {cv['qwk_sd']:.3f}), macro-F1 {cv['macro_f1_mean']:.3f}.",
        f"Test QWK {t['qwk']:.3f} (95 % CI {t['qwk_ci95'][0]:.3f} - {t['qwk_ci95'][1]:.3f}), macro-F1 "
        f"{t['macro_f1']:.3f}, accuracy {t['accuracy']:.3f}, MAE {t['mae_grades']:.3f} grades, ECE {t['ece']:.3f}.",
        f"Remission vs active AUC: {t.get('remission_vs_active_auc', float('nan')):.3f}.",
        f"Per-grade recall: {t['per_grade_recall']}.", "",
        "## Limitations",
        "- One dataset from one centre. Performance on other endoscopes, centres and populations is unknown.",
        "- Frame-level grading. Clinical Mayo scoring uses the most severe segment of the whole colon.",
        "- Calibration is reported but not corrected.",
    ]
    return "\n".join(lines) + "\n"
