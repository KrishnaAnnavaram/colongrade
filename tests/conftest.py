import pytest

from colongrade.data.manifest import assign_folds, build_manifest
from colongrade.data.synthetic import write_dataset


@pytest.fixture(scope="session")
def dataset(tmp_path_factory):
    root = write_dataset(tmp_path_factory.mktemp("limuc"), n_patients=20, images_per_patient=8, test_patients=5,
                         size=32, seed=5)
    manifest = assign_folds(build_manifest(root), n_folds=3, seed=5)
    return root, manifest
