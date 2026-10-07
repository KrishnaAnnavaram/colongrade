"""CNN grader with an ordinal CORAL head (requires torch; a pretrained backbone needs torchvision).

* RGB input, the same ``load_image`` path as every other model.
* Backbones: ``small`` (a compact residual CNN, trained from scratch, used offline) or ``resnet18`` /
  ``resnet50`` from torchvision with optional ImageNet weights (transfer learning).
* CORAL head: one shared score and three ordered thresholds give P(grade > k).
* Class-weighted loss for the imbalance, flips and 90-degree turns for augmentation (no hue change,
  because hue carries the redness signal).
* Early stopping on the VALIDATION fold QWK, with a deep copy of the best weights.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from ..evaluate import qwk

K = 4


class _Block(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(),
                                  nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout))
        self.skip = nn.Sequential() if stride == 1 and cin == cout else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        return torch.relu(self.body(x) + self.skip(x))


class SmallBackbone(nn.Module):
    def __init__(self, width: int = 16):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, 1, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU())
        self.stages = nn.Sequential(_Block(width, width, 1), _Block(width, 2 * width, 2),
                                    _Block(2 * width, 4 * width, 2), _Block(4 * width, 8 * width, 2))
        self.out_features = 8 * width

    def forward(self, x):
        return self.stages(self.stem(x))


def build_backbone(name: str, pretrained: bool = False) -> nn.Module:
    if name == "small":
        return SmallBackbone()
    if name in ("resnet18", "resnet50"):
        try:
            import torchvision
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise ImportError("install the 'vision' extra: pip install -e '.[vision]'") from exc
        weights = "DEFAULT" if pretrained else None
        net = getattr(torchvision.models, name)(weights=weights)
        feats = net.fc.in_features
        body = nn.Sequential(*list(net.children())[:-2])
        body.out_features = feats
        return body
    raise ValueError(f"unknown backbone {name!r}")


class CoralHead(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.score = nn.Linear(features, 1, bias=False)
        self.thresholds = nn.Parameter(torch.tensor([-1.0, 0.0, 1.0]))

    def forward(self, f):
        return self.score(f) - self.thresholds  # (B, K-1) logits of P(y > k)


class OrdinalCNN(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = CoralHead(backbone.out_features)

    def forward(self, x):
        return self.head(self.pool(self.backbone(x)).flatten(1))


def coral_targets(y: torch.Tensor) -> torch.Tensor:
    return (y.unsqueeze(1) > torch.arange(K - 1, device=y.device)).float()


def coral_proba(logits: torch.Tensor) -> torch.Tensor:
    gt = torch.sigmoid(logits)
    gt = torch.cummin(gt, dim=1).values
    probs = torch.cat([1 - gt[:, :1], gt[:, :-1] - gt[:, 1:], gt[:, -1:]], dim=1).clamp(min=0)
    return probs / probs.sum(1, keepdim=True)


def _augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    if torch.rand(1, generator=gen).item() < 0.5:
        x = x.flip(-1)
    if torch.rand(1, generator=gen).item() < 0.5:
        x = x.flip(-2)
    k = int(torch.randint(0, 4, (1,), generator=gen).item())
    return torch.rot90(x, k, dims=(-2, -1))


_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_tensor(images: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float()
    return (x - _MEAN) / _STD


class CNNGrader:
    def __init__(self, backbone: str = "small", pretrained: bool = False, epochs: int = 20, patience: int = 4,
                 lr: float = 1e-3, batch_size: int = 32, seed: int = 13, device: str = "cpu"):
        self.name = f"cnn-{backbone}"
        self.backbone_name, self.pretrained = backbone, pretrained
        self.epochs, self.patience, self.lr, self.batch_size = epochs, patience, lr, batch_size
        self.seed, self.device = seed, device
        self.model: OrdinalCNN | None = None
        self.history: list[float] = []
        self.best_epoch = -1

    def _predict(self, images: np.ndarray) -> np.ndarray:
        self.model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(images), 64):
                logits = self.model(to_tensor(images[i : i + 64]).to(self.device))
                out.append(coral_proba(logits).cpu().numpy())
        return np.concatenate(out)

    def fit(self, images: np.ndarray, y, val_images: np.ndarray | None = None, val_y=None) -> "CNNGrader":
        """With a validation set: early stopping on its QWK. Without one: ``epochs`` epochs, last weights."""
        torch.manual_seed(self.seed)
        gen = torch.Generator().manual_seed(self.seed)
        y = np.asarray(y)
        self.model = OrdinalCNN(build_backbone(self.backbone_name, self.pretrained)).to(self.device)
        counts = np.bincount(y, minlength=K).astype(float)
        weight_per_grade = torch.tensor(len(y) / (K * np.maximum(counts, 1)), dtype=torch.float32)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        rng = np.random.default_rng(self.seed)
        best, best_state, bad = -np.inf, None, 0
        for epoch in range(self.epochs):
            self.model.train()
            order = rng.permutation(len(y))
            for i in range(0, len(order), self.batch_size):
                idx = order[i : i + self.batch_size]
                x = _augment(to_tensor(images[idx]), gen).to(self.device)
                yt = torch.tensor(y[idx], device=self.device)
                logits = self.model(x)
                w = weight_per_grade.to(self.device)[yt].unsqueeze(1)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, coral_targets(yt), weight=w)
                opt.zero_grad()
                loss.backward()
                opt.step()
            if val_images is None:
                self.best_epoch = epoch
                continue
            score = qwk(np.asarray(val_y), self._predict(val_images).argmax(1))
            self.history.append(score)
            if score > best:
                best, best_state, bad, self.best_epoch = score, copy.deepcopy(self.model.state_dict()), 0, epoch
            else:
                bad += 1
                if bad > self.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict_proba(self, images: np.ndarray) -> np.ndarray:
        return self._predict(images)

    def grad_cam(self, image: np.ndarray) -> np.ndarray:
        """Grad-CAM map (H, W) in [0, 1] for the expected-grade score of one image."""
        self.model.eval()
        feats = {}
        handle = self.model.backbone.register_forward_hook(lambda m, i, o: feats.setdefault("a", o))
        x = to_tensor(image[None]).to(self.device)
        logits = self.model(x)
        handle.remove()
        act = feats["a"]
        score = torch.sigmoid(logits).sum()
        grads = torch.autograd.grad(score, act)[0]
        cam = torch.relu((grads.mean(dim=(2, 3), keepdim=True) * act).sum(1))[0]
        cam = nn.functional.interpolate(cam[None, None], size=image.shape[:2], mode="bilinear",
                                        align_corners=False)[0, 0]
        cam = cam - cam.min()
        return (cam / cam.max().clamp(min=1e-8)).detach().cpu().numpy()
