from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def normalize_log1p(values: np.ndarray, target_sum: float = 1e4) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    totals = values.sum(axis=1, keepdims=True)
    scaled = values * (target_sum / np.maximum(totals, 1.0))
    return np.log1p(scaled).astype(np.float32)


@dataclass(frozen=True)
class PseudospotData:
    train_x: np.ndarray
    train_p: np.ndarray
    validation_x: np.ndarray
    validation_p: np.ndarray
    prototype_x: np.ndarray
    class_names: np.ndarray


def build_pseudospot_data(
    marker_raw_counts: np.ndarray,
    labels: np.ndarray,
    *,
    n_train: int = 4096,
    n_validation: int = 1024,
    min_cells: int = 8,
    max_cells: int = 24,
    max_types: int = 5,
    equalize_cell_library: bool = True,
    seed: int = 42,
) -> PseudospotData:
    """Generate pseudo-spots with known cell-count proportions."""
    counts = np.asarray(marker_raw_counts, dtype=np.float32)
    labels = np.asarray(labels).astype(str)
    if counts.ndim != 2 or len(counts) != len(labels):
        raise ValueError("marker_raw_counts and labels must have aligned rows")
    if n_train < 1 or n_validation < 1:
        raise ValueError("pseudo-spot train and validation sizes must be positive")
    if min_cells < 1 or max_cells < min_cells or max_types < 1:
        raise ValueError("invalid pseudo-spot mixture limits")
    if equalize_cell_library:
        # Targets are cell-count fractions, not RNA-contribution fractions.
        cell_totals = counts.sum(axis=1, keepdims=True)
        counts = counts * (1e4 / np.maximum(cell_totals, 1.0))

    class_names, encoded = np.unique(labels, return_inverse=True)
    indices_by_type = [
        np.flatnonzero(encoded == idx) for idx in range(len(class_names))
    ]
    if not len(class_names) or any(len(indices) == 0 for indices in indices_by_type):
        raise ValueError("at least one observed cell is required per class")
    rng = np.random.default_rng(seed)
    prototypes = np.vstack(
        [counts[indices].mean(axis=0) for indices in indices_by_type]
    ).astype(np.float32)

    def generate(n_spots: int) -> tuple[np.ndarray, np.ndarray]:
        pseudo = np.zeros((n_spots, counts.shape[1]), dtype=np.float32)
        proportions = np.zeros((n_spots, len(class_names)), dtype=np.float32)
        for spot_idx in range(n_spots):
            n_present = int(rng.integers(1, min(max_types, len(class_names)) + 1))
            present = rng.choice(len(class_names), size=n_present, replace=False)
            n_cells = int(rng.integers(min_cells, max_cells + 1))
            allocation = rng.multinomial(
                n_cells, rng.dirichlet(np.full(n_present, 0.5))
            )
            for missing_idx in np.flatnonzero(allocation == 0):
                donor = int(np.argmax(allocation))
                if allocation[donor] > 1:
                    allocation[donor] -= 1
                    allocation[missing_idx] += 1
            selected: list[int] = []
            for class_id, amount in zip(present, allocation):
                if amount <= 0:
                    continue
                pool = indices_by_type[int(class_id)]
                selected.extend(
                    rng.choice(
                        pool, size=int(amount), replace=len(pool) < amount
                    ).tolist()
                )
                proportions[spot_idx, int(class_id)] = float(amount) / n_cells
            pseudo[spot_idx] = counts[selected].sum(axis=0)
        return normalize_log1p(pseudo), proportions

    train_x, train_p = generate(n_train)
    validation_x, validation_p = generate(n_validation)
    return PseudospotData(
        train_x=train_x,
        train_p=train_p,
        validation_x=validation_x,
        validation_p=validation_p,
        prototype_x=normalize_log1p(prototypes),
        class_names=class_names,
    )
