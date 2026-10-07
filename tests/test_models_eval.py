import json

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

from colongrade.cli import main
from colongrade.evaluate import ece, qwk, report
from colongrade.experiment import ExperimentConfig, predict_file, run
from colongrade.features import FEATURE_NAMES, extract
from colongrade.models.baseline import MultinomialGrader, OrdinalGrader


def test_qwk_matches_sklearn_and_punishes_far_errors():
    # reference problem 10: accuracy treats 0->3 like 0->1
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 4, 200), rng.integers(0, 4, 200)
    assert qwk(y, p) == pytest.approx(cohen_kappa_score(y, p, weights="quadratic"))
    truth = np.array([0, 0, 1, 2, 3, 3])
    near = np.array([1, 0, 1, 2, 3, 3])
    far = np.array([3, 0, 1, 2, 3, 3])
    assert (near == truth).mean() == (far == truth).mean()
    assert qwk(truth, near) > qwk(truth, far)


def test_report_fields():
    y = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    proba = np.eye(4)[y] * 0.7 + 0.075
    r = report(y, proba, n_boot=50)
    assert r["qwk"] == pytest.approx(1.0) and r["accuracy"] == 1.0
    assert r["confusion"][0] == [2, 0, 0, 0] and r["per_grade_recall"] == [1.0] * 4
    assert r["remission_vs_active_auc"] == pytest.approx(1.0) and r["mae_grades"] == 0
    assert ece(y, proba) == pytest.approx(1 - 0.775, abs=1e-6)  # all correct at confidence 0.775


def test_features_follow_the_mayo_cues():
    pale = np.ones((32, 32, 3)) * np.array([0.8, 0.55, 0.5])
    red = np.ones((32, 32, 3)) * np.array([0.85, 0.35, 0.33])
    f_pale, f_red = extract(pale), extract(red)
    i = FEATURE_NAMES.index("redness_mean")
    assert f_red[i] > f_pale[i] and len(f_pale) == len(FEATURE_NAMES)


@pytest.mark.parametrize("cls", [MultinomialGrader, OrdinalGrader])
def test_graders_handle_imbalance_and_give_probabilities(cls):
    # reference problem 7: class weights instead of SMOTE on pixels
    rng = np.random.default_rng(1)
    y = np.repeat([0, 1, 2, 3], [120, 60, 20, 10])
    x = y[:, None] + rng.normal(0, 0.6, (len(y), 3))
    proba = cls().fit(x, y).predict_proba(x)
    assert proba.shape == (len(y), 4) and np.allclose(proba.sum(1), 1)
    assert qwk(y, proba.argmax(1)) > 0.7
    assert (proba.argmax(1) == 3).sum() > 0  # the rare grade is still predicted


def test_experiment_uses_held_out_folds_and_test_once(dataset, tmp_path):
    # reference problems 1, 2 and 6: all images, validation on held-out folds, one transform path
    root, manifest = dataset
    cfg = ExperimentConfig(model="features-ordinal", contrast="clahe", image_size=32, folds=3, n_boot=50)
    res = run(manifest, root, cfg, tmp_path / "run", log=lambda m: None)
    assert res["images"] == {"trainval": int((manifest.split == "trainval").sum()),
                             "test": int((manifest.split == "test").sum())}
    assert sum(f["n_val"] for f in res["cv"]["folds"]) == res["images"]["trainval"]
    assert res["test"]["n"] == res["images"]["test"]
    for name in ("config.json", "result.json", "model_card.md", "test_predictions.csv", "model.joblib"):
        assert (tmp_path / "run" / name).exists()
    assert "NOT a medical device" in (tmp_path / "run" / "model_card.md").read_text(encoding="utf-8")
    assert len(res["permutation_importance_test"]) == 10
    out = predict_file(tmp_path / "run", root / manifest["path"].iloc[0])
    assert out["predicted_mayo"] in range(4) and "Not a diagnosis" in out["note"]


def test_bad_config_is_refused():
    with pytest.raises(ValueError):
        ExperimentConfig(model="xgboost")


def test_cli_manifest_train_predict(dataset, tmp_path, monkeypatch, capsys):
    root, _ = dataset
    monkeypatch.setenv("COLONGRADE_OUT_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("COLONGRADE_FOLDS", "3")
    assert main(["manifest", str(root), "--out", str(tmp_path / "m.csv")]) == 0
    assert main(["train", str(root), "--manifest", str(tmp_path / "m.csv"), "--model", "features-multinomial",
                 "--size", "32", "--n-boot", "20", "--name", "r1"]) == 0
    capsys.readouterr()
    image = next(root.rglob("*.png"))
    assert main(["predict", str(tmp_path / "runs" / "r1"), str(image)]) == 0
    assert json.loads(capsys.readouterr().out)["predicted_mayo"] in range(4)
