# data/

Git ignores everything in this folder except this file. Do not commit images, manifests or models.
The offline demo and the tests use synthetic drawings (`colongrade synth`) and need no download.

## Source

**LIMUC** (Labeled Images for Ulcerative Colitis), Zenodo record 5827695,
<https://zenodo.org/records/5827695>, license CC BY 4.0. About 11 000 colonoscopy frames graded
Mayo 0 to 3 by experts, with a train-and-validation part and a separate test part. Cite the dataset
authors in any publication.

## Download

1. Download the archive from the Zenodo page.
2. Extract it and the inner archives under `data/LIMUC/`.
3. Check that each image sits below a folder named for its grade (`Mayo 0` ... `Mayo 3`).

## Expected layout

```
data/LIMUC/
  train_and_validation_sets/ ... /Mayo 0/*.bmp   (any depth, one grade folder above each image)
  test_set/ ... /Mayo 3/*.bmp                    (a top folder with "test" in its name is the test set)
```

If a parent folder is a patient number (`12` or `patient_12`), `manifest` reads it and the folds are
grouped by patient. Without patient folders, the folds are stratified by grade only (see the README,
known problems).

## Files that `colongrade manifest` writes

| File | Columns |
|---|---|
| `data/LIMUC/manifest.csv` | `image_id, path, mayo, split, patient, content_hash, fold` (`fold` = -1 for the test set) |

The manifest checks that no patient and no identical image file occurs in both trainval and test.
