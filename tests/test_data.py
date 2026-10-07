import numpy as np
import pytest
from PIL import Image

from colongrade.config import load_settings
from colongrade.data.images import ImageConfig, clahe, equalize, load_image, tanh_to_uint8
from colongrade.data.manifest import ManifestError, assign_folds, build_manifest, check_manifest, load_manifest
from colongrade.labels import LabelError, active, parse_grade


def test_grade_parsing_and_active_split():
    assert [parse_grade(x) for x in ("Mayo 0", "mayo_1", "MAYO-2", "3")] == [0, 1, 2, 3]
    with pytest.raises(LabelError):
        parse_grade("Mayo 4")
    assert [active(g) for g in range(4)] == [0, 0, 1, 1]


def test_manifest_lists_every_image(dataset):
    # reference problem 1: the prototype used 5 images per class
    root, manifest = dataset
    n_files = sum(1 for _ in root.rglob("*.png"))
    assert len(manifest) == n_files == 160
    assert set(manifest["split"]) == {"trainval", "test"}
    assert (manifest[manifest.split == "test"]["fold"] == -1).all()
    assert set(manifest[manifest.split == "trainval"]["fold"]) == {0, 1, 2}


def test_folds_are_grouped_by_patient_and_test_is_separate(dataset):
    _, manifest = dataset
    tv = manifest[manifest.split == "trainval"]
    assert (tv.groupby("patient")["fold"].nunique() == 1).all()
    assert not set(tv["patient"]) & set(manifest[manifest.split == "test"]["patient"])
    stats = check_manifest(manifest)
    assert stats["trainval"] + stats["test"] == 160


def test_leaks_are_detected(dataset, tmp_path):
    _, manifest = dataset
    leak = manifest.copy()
    leak.loc[leak[leak.split == "trainval"].index[0], "patient"] = leak[leak.split == "test"]["patient"].iloc[0]
    with pytest.raises(ManifestError):
        check_manifest(leak)
    dup = manifest.copy()
    first_test = dup[dup.split == "test"].index[0]
    dup.loc[first_test, "content_hash"] = dup[dup.split == "trainval"]["content_hash"].iloc[0]
    with pytest.raises(ManifestError):
        check_manifest(dup)
    manifest.to_csv(tmp_path / "m.csv", index=False)
    assert len(load_manifest(tmp_path / "m.csv")) == len(manifest)


def test_manifest_without_patients_uses_plain_stratified_folds(tmp_path):
    for split in ("train_and_validation_sets", "test_set"):
        for g in range(4):
            folder = tmp_path / split / f"Mayo {g}"
            folder.mkdir(parents=True)
            for i in range(6):
                Image.fromarray(np.full((8, 8, 3), 10 * g + i + (100 if split == "test_set" else 0),
                                        np.uint8)).save(folder / f"{i}.png")
    (tmp_path / "notes").mkdir()
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(tmp_path / "notes" / "ungraded.png")
    df = assign_folds(build_manifest(tmp_path), n_folds=3)
    assert len(df) == 48 and (df["patient"] == "").all()
    with pytest.raises(ManifestError):
        build_manifest(tmp_path / "notes")


def test_images_stay_rgb_and_contrast_keeps_hue(dataset):
    # reference problem 8: grayscale removed the redness signal
    root, manifest = dataset
    img = load_image(root / manifest["path"].iloc[0], ImageConfig(size=24))
    assert img.shape == (24, 24, 3) and img.dtype == np.float32 and 0 <= img.min() and img.max() <= 1
    dark = np.clip(img * 0.4, 0, 1)
    for out in (equalize(dark), clahe(dark, tiles=4)):
        assert out.shape == dark.shape and out.mean() > dark.mean()
        ratio_in = dark[..., 0].sum() / dark.sum()
        ratio_out = out[..., 0].sum() / out.sum()
        assert ratio_out == pytest.approx(ratio_in, abs=0.03)  # redness share is kept
    with pytest.raises(ValueError):
        ImageConfig(contrast="sharpen")


def test_tanh_rescaling_has_no_wrap_around():
    # reference problem 4: tanh output * 255 cast to uint8 wrapped negative values
    out = tanh_to_uint8(np.array([-1.0, -0.5, 0.0, 1.0, 1.2]))
    assert out.tolist() == [0, 64, 128, 255, 255]


def test_settings():
    s = load_settings({"COLONGRADE_IMAGE_SIZE": "64", "COLONGRADE_FOLDS": "3", "COLONGRADE_DEVICE": "CPU"})
    assert s.image_size == 64 and s.folds == 3 and s.device == "cpu"
    with pytest.raises(ValueError):
        load_settings({"COLONGRADE_FOLDS": "1"})
