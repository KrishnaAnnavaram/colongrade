"""Feature baselines (scikit-learn): a class-weighted multinomial model and an ordinal threshold model.

Imbalance is handled with class weights, not with SMOTE on raw pixels (interpolated pixel vectors are
not realistic images). The scaler is inside the pipeline, so it is fit on training data only.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

K = 4


class MultinomialGrader:
    name = "features-multinomial"

    def __init__(self, c: float = 1.0, seed: int = 13):
        self.pipe = Pipeline([("scale", StandardScaler()),
                              ("clf", LogisticRegression(C=c, class_weight="balanced", max_iter=3000,
                                                         random_state=seed))])

    def fit(self, x, y) -> "MultinomialGrader":
        self.pipe.fit(x, y)
        return self

    def predict_proba(self, x) -> np.ndarray:
        raw = self.pipe.predict_proba(x)
        out = np.zeros((len(x), K))
        out[:, self.pipe.classes_.astype(int)] = raw
        return out


class OrdinalGrader:
    """Frank-and-Hall ordinal model: K-1 binary models for P(grade > k), turned into grade probabilities."""

    name = "features-ordinal"

    def __init__(self, c: float = 1.0, seed: int = 13):
        self.c, self.seed = c, seed
        self.scaler = StandardScaler()
        self.models: list[LogisticRegression] = []

    def fit(self, x, y) -> "OrdinalGrader":
        y = np.asarray(y)
        xs = self.scaler.fit_transform(x)
        weights = compute_sample_weight("balanced", y)
        self.models = []
        for k in range(K - 1):
            target = (y > k).astype(int)
            m = LogisticRegression(C=self.c, max_iter=3000, random_state=self.seed)
            m.fit(xs, target, sample_weight=weights)
            self.models.append(m)
        return self

    def predict_proba(self, x) -> np.ndarray:
        xs = self.scaler.transform(x)
        gt = np.column_stack([m.predict_proba(xs)[:, 1] for m in self.models])  # P(y > k)
        gt = np.minimum.accumulate(gt, axis=1)  # keep P(y>0) >= P(y>1) >= P(y>2)
        probs = np.column_stack([1 - gt[:, 0], gt[:, 0] - gt[:, 1], gt[:, 1] - gt[:, 2], gt[:, 2]])
        probs = np.clip(probs, 0, None)
        return probs / probs.sum(1, keepdims=True)


GRADERS = {"features-multinomial": MultinomialGrader, "features-ordinal": OrdinalGrader}
