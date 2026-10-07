"""CNN and GAN tests. They skip when torch is not installed (CI installs only the dev extra)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from colongrade.experiment import ExperimentConfig, run  # noqa: E402
from colongrade.models.cgan import ConditionalGAN, Generator  # noqa: E402
from colongrade.models.cnn import CNNGrader, coral_proba, coral_targets  # noqa: E402


def test_coral_targets_and_probabilities():
    assert coral_targets(torch.tensor([0, 2, 3])).tolist() == [[0, 0, 0], [1, 1, 0], [1, 1, 1]]
    p = coral_proba(torch.tensor([[5.0, 0.0, -5.0], [-1.0, 2.0, 0.0]]))
    assert torch.allclose(p.sum(1), torch.ones(2)) and (p >= 0).all()
    assert p[0].argmax().item() in (1, 2)


def test_cnn_trains_and_keeps_the_best_epoch(dataset):
    root, manifest = dataset
    from colongrade.data.images import ImageConfig
    from colongrade.experiment import load_images

    tv = manifest[manifest.split == "trainval"]
    x = load_images(tv["path"], root, ImageConfig(size=32))
    y = tv["mayo"].to_numpy()
    tr, va = (tv["fold"] != 0).to_numpy(), (tv["fold"] == 0).to_numpy()
    model = CNNGrader(epochs=3, patience=1).fit(x[tr], y[tr], x[va], y[va])
    assert model.best_epoch == int(np.argmax(model.history))
    proba = model.predict_proba(x[va])
    assert proba.shape == (va.sum(), 4) and np.allclose(proba.sum(1), 1, atol=1e-5)
    cam = model.grad_cam(x[0])
    assert cam.shape == (32, 32) and 0 <= cam.min() and cam.max() <= 1


def test_conditional_gan_uses_real_grades_and_correct_range():
    # reference problems 3 and 4: a fifth "synthetic" class and wrapped pixel values
    g = Generator(z_dim=8, size=32, width=8)
    with pytest.raises(IndexError):
        g(torch.randn(1, 8), torch.tensor([4]))  # no fifth class exists
    images = np.random.default_rng(0).random((16, 32, 32, 3)).astype(np.float32)
    gan = ConditionalGAN(size=32, z_dim=8, epochs=2, batch_size=8)
    gan.fit(images, np.repeat([0, 1, 2, 3], 4))
    assert len(gan.losses) == 2  # epochs are passes over the data
    out = gan.sample([0, 1, 2, 3])
    assert out.shape == (4, 32, 32, 3) and out.dtype == np.uint8


def test_gan_ablation_runs_inside_the_training_folds(dataset, tmp_path):
    root, manifest = dataset
    cfg = ExperimentConfig(model="features-ordinal", image_size=32, folds=3, gan_per_grade=3, gan_epochs=1, n_boot=20)
    res = run(manifest, root, cfg, tmp_path / "gan", log=lambda m: None)
    assert res["test"]["n"] == int((manifest.split == "test").sum())  # generated images never reach the test set


def test_final_cnn_without_validation_trains_the_given_epochs(dataset):
    root, manifest = dataset
    from colongrade.data.images import ImageConfig
    from colongrade.experiment import load_images

    tv = manifest[manifest.split == "trainval"].head(24)
    x = load_images(tv["path"], root, ImageConfig(size=32))
    model = CNNGrader(epochs=2).fit(x, tv["mayo"].to_numpy())
    assert model.best_epoch == 1 and model.history == []
