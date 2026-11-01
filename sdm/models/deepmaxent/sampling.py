"""Training batch samplers for DeepMaxent.

Balanced batches (optional, off by default)
------------------------------------------
When ``balanced_batches`` is on, each mini-batch includes a fixed number of
presence rows for **every** output (sampled with replacement when a species is
rarer than that count), with the remainder filled by a fresh draw of background
rows (sites with no presence for any output).

DeepMaxent loss (paper eq. 4) soft-maxes intensities over **all** sites in the
batch. Oversampled presence rows for output *j* therefore also enter every other
output's partition function as non-presence sites (their ``y`` for those outputs
is zero). That is the intended quadrature for those heads — no extra correction
is applied; the count-weighted softmax already ignores them in the numerator.
"""

from __future__ import annotations

from typing import Iterator, List, Optional

import numpy as np

from sdm.models.deepmaxent._torch_util import require_torch


def presence_indices_by_output(Y: np.ndarray) -> List[np.ndarray]:
    """Row indices with a positive label for each output column."""
    Y = np.asarray(Y)
    if Y.ndim != 2:
        raise ValueError(f"Y must be rank-2, got shape {Y.shape}")
    return [np.flatnonzero(Y[:, j] > 0) for j in range(Y.shape[1])]


def background_indices(Y: np.ndarray) -> np.ndarray:
    """Rows with no presence on any output (shared TGB / background sites)."""
    Y = np.asarray(Y)
    return np.flatnonzero(Y.sum(axis=1) <= 0)


def n_batches_for_epoch(n_rows: int, batch_size: int) -> int:
    """Match default DataLoader length with ``drop_last=False``."""
    bs = max(2, min(int(batch_size), max(int(n_rows), 2)))
    return int(np.ceil(n_rows / bs)) if n_rows > 0 else 0


class BalancedPresenceBatchSampler:
    """Yield fixed-size batches with ``k`` presence slots per output.

    Uses NumPy RNG for presence/background draws (with-replacement for rare
    outputs). Implements the torch ``BatchSampler`` protocol so it plugs into
    ``DataLoader(batch_sampler=...)``.
    """

    def __init__(
        self,
        Y: np.ndarray,
        *,
        batch_size: int,
        presences_per_output: int = 4,
        seed: int = 0,
        n_batches: Optional[int] = None,
    ) -> None:
        torch, _nn = require_torch()  # ensure dl extra is present
        del torch  # sampler itself is index-only

        self.Y = np.asarray(Y)
        if self.Y.ndim != 2:
            raise ValueError(f"Y must be rank-2, got shape {self.Y.shape}")
        self.n_rows = int(self.Y.shape[0])
        self.n_outputs = int(self.Y.shape[1])
        self.batch_size = int(batch_size)
        self.presences_per_output = int(presences_per_output)
        self.seed = int(seed)
        self._epoch = 0

        if self.presences_per_output < 1:
            raise ValueError("presences_per_output must be >= 1")
        if self.batch_size < 2:
            raise ValueError("batch_size must be >= 2")

        self._presence = presence_indices_by_output(self.Y)
        for j, idx in enumerate(self._presence):
            if idx.size == 0:
                raise ValueError(
                    f"Output column {j} has no presence rows; "
                    "cannot build balanced batches"
                )

        needed = self.n_outputs * self.presences_per_output
        if needed >= self.batch_size:
            raise ValueError(
                f"presences_per_output={self.presences_per_output} × "
                f"n_outputs={self.n_outputs} = {needed} leaves no room for "
                f"background in batch_size={self.batch_size}"
            )

        self._background = background_indices(self.Y)
        if self._background.size == 0:
            # Fall back to any row; still allows training on tiny synthetics.
            self._background = np.arange(self.n_rows)

        self.n_batches = (
            int(n_batches)
            if n_batches is not None
            else n_batches_for_epoch(self.n_rows, self.batch_size)
        )

    def set_epoch(self, epoch: int) -> None:
        """Reseed draws each epoch (Lightning-compatible)."""
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + 1_000_003 * self._epoch)
        n_bg = self.batch_size - self.n_outputs * self.presences_per_output
        for _ in range(self.n_batches):
            parts: List[np.ndarray] = []
            for idx in self._presence:
                # With-replacement when rarer than k (and always for a simple API).
                chosen = rng.choice(idx, size=self.presences_per_output, replace=True)
                parts.append(chosen)
            bg = rng.choice(self._background, size=n_bg, replace=True)
            parts.append(bg)
            batch = np.concatenate(parts)
            rng.shuffle(batch)
            yield batch.astype(int).tolist()


def build_train_dataloader(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    batch_size: int,
    seed: int,
    balanced_batches: bool = False,
    presences_per_output: int = 4,
):
    """Train DataLoader: shuffled default path, or balanced presence batches.

    When ``balanced_batches`` is False the loader matches the historical
    shuffle + ``drop_last=False`` behaviour (same length and RNG seeding).
    """
    torch, _nn = require_torch()
    from torch.utils.data import DataLoader, TensorDataset

    ds = TensorDataset(
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(Y, dtype=torch.float32),
    )
    n = X.shape[0]
    bs = max(2, min(int(batch_size), max(n, 2)))

    if not balanced_batches:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        return DataLoader(
            ds,
            batch_size=bs,
            shuffle=True,
            generator=generator,
            drop_last=False,
        )

    batch_sampler = BalancedPresenceBatchSampler(
        Y,
        batch_size=bs,
        presences_per_output=presences_per_output,
        seed=seed,
        n_batches=n_batches_for_epoch(n, bs),
    )
    return DataLoader(ds, batch_sampler=batch_sampler)
