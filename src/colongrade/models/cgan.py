"""Class-conditional DCGAN for the augmentation ABLATION (requires torch).

Fixes against the prototype:

* The condition is the real Mayo grade (0-3). A generated image gets that grade as its label, never
  a fifth "synthetic" class that the classifier has no output for.
* ``tanh`` output in [-1, 1] maps to [0, 255] with ``tanh_to_uint8`` (no wrap-around).
* ``epochs`` counts passes over the training data, not single batches.
* The GAN trains on the TRAINING folds only. Generated images never enter validation or test.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..data.images import tanh_to_uint8

K = 4


class Generator(nn.Module):
    def __init__(self, z_dim: int = 64, size: int = 64, width: int = 32):
        super().__init__()
        if size % 16:
            raise ValueError("size must be a multiple of 16")
        self.z_dim, self.size = z_dim, size
        self.embed = nn.Embedding(K, z_dim)
        self.start = size // 16
        self.fc = nn.Linear(2 * z_dim, 8 * width * self.start * self.start)
        layers, ch = [], 8 * width
        for _ in range(4):
            layers += [nn.ConvTranspose2d(ch, ch // 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ch // 2), nn.ReLU()]
            ch //= 2
        self.up = nn.Sequential(*layers, nn.Conv2d(ch, 3, 3, 1, 1), nn.Tanh())
        self.width = width

    def forward(self, z, grade):
        h = self.fc(torch.cat([z, self.embed(grade)], 1)).view(len(z), 8 * self.width, self.start, self.start)
        return self.up(h)


class Discriminator(nn.Module):
    def __init__(self, size: int = 64, width: int = 32):
        super().__init__()
        layers, ch = [], 3
        for i in range(4):
            out = width * 2**i
            layers += [nn.Conv2d(ch, out, 4, 2, 1), nn.LeakyReLU(0.2)]
            ch = out
        self.body = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.out = nn.Linear(ch, 1)
        self.embed = nn.Embedding(K, ch)  # projection discriminator: the grade enters as an inner product

    def forward(self, x, grade):
        h = self.body(x)
        return self.out(h).squeeze(1) + (self.embed(grade) * h).sum(1)


class ConditionalGAN:
    def __init__(self, size: int = 64, z_dim: int = 64, epochs: int = 30, batch_size: int = 32, lr: float = 2e-4,
                 seed: int = 13, device: str = "cpu"):
        self.size, self.z_dim, self.epochs, self.batch_size = size, z_dim, epochs, batch_size
        self.lr, self.seed, self.device = lr, seed, device
        self.g = Generator(z_dim, size).to(device)
        self.d = Discriminator(size).to(device)
        self.losses: list[tuple[float, float]] = []

    def fit(self, images: np.ndarray, grades) -> "ConditionalGAN":
        """``images``: float RGB (N, size, size, 3) in [0, 1] from the training folds only."""
        torch.manual_seed(self.seed)
        x_all = torch.from_numpy(images).permute(0, 3, 1, 2).float() * 2 - 1
        y_all = torch.as_tensor(np.asarray(grades), dtype=torch.long)
        opt_g = torch.optim.Adam(self.g.parameters(), lr=self.lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.d.parameters(), lr=self.lr, betas=(0.5, 0.999))
        bce = nn.BCEWithLogitsLoss()
        rng = np.random.default_rng(self.seed)
        for _ in range(self.epochs):  # one epoch = one pass over all training images
            order = rng.permutation(len(x_all))
            for i in range(0, len(order), self.batch_size):
                idx = torch.as_tensor(order[i : i + self.batch_size])
                real, y = x_all[idx].to(self.device), y_all[idx].to(self.device)
                z = torch.randn(len(idx), self.z_dim, device=self.device)
                fake = self.g(z, y)
                d_loss = bce(self.d(real, y), torch.full((len(idx),), 0.9, device=self.device)) + \
                    bce(self.d(fake.detach(), y), torch.zeros(len(idx), device=self.device))
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()
                g_loss = bce(self.d(fake, y), torch.ones(len(idx), device=self.device))
                opt_g.zero_grad()
                g_loss.backward()
                opt_g.step()
            self.losses.append((float(d_loss.detach()), float(g_loss.detach())))
        return self

    @torch.no_grad()
    def sample(self, grades) -> np.ndarray:
        """uint8 RGB images (N, size, size, 3) for the given grades."""
        self.g.eval()
        y = torch.as_tensor(np.asarray(grades), dtype=torch.long, device=self.device)
        z = torch.randn(len(y), self.z_dim, device=self.device)
        out = self.g(z, y).permute(0, 2, 3, 1).cpu().numpy()
        self.g.train()
        return tanh_to_uint8(out)
