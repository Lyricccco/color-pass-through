#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit PCA in camera-null coordinates (paper Option A)"
    )
    parser.add_argument("--hsi-dir", required=True)
    parser.add_argument("--camera-sensitivity", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--components", type=int, default=3)
    parser.add_argument("--chunk-pixels", type=int, default=1_000_000)
    return parser


def _update(
    mean: "np.ndarray", scatter: "np.ndarray", count: int, batch: "np.ndarray"
) -> tuple["np.ndarray", "np.ndarray", int]:
    import numpy as np

    size = len(batch)
    if size == 0:
        return mean, scatter, count
    batch_mean = batch.mean(axis=0)
    centered = batch - batch_mean
    batch_scatter = centered.T @ centered
    if count == 0:
        return batch_mean, batch_scatter, size
    total = count + size
    delta = batch_mean - mean
    new_mean = mean + delta * (size / total)
    new_scatter = (
        scatter
        + batch_scatter
        + np.outer(delta, delta) * (count * size / total)
    )
    return new_mean, new_scatter, total


def main() -> None:
    args = build_parser().parse_args()
    import numpy as np

    from color_pass_through.models.camera_null import load_camera_sensitivity

    files = sorted(Path(args.hsi_dir).glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No NPY cubes in {args.hsi_dir}")
    first = np.load(files[0], mmap_mode="r")
    if first.ndim != 3:
        raise ValueError(f"Expected H×W×B, got {first.shape}")
    bands = int(first.shape[-1])
    wavelengths = np.arange(400, 400 + 10 * bands, 10)
    sensitivity = load_camera_sensitivity(args.camera_sensitivity, wavelengths).astype(
        np.float64
    )
    left, singular_values, _ = np.linalg.svd(sensitivity, full_matrices=True)
    rank = int(
        np.sum(singular_values > 1e-10 * max(float(singular_values[0]), 1.0))
    )
    null_basis = left[:, rank:]
    dimension = null_basis.shape[1]
    mean = np.zeros(dimension, dtype=np.float64)
    scatter = np.zeros((dimension, dimension), dtype=np.float64)
    count = 0
    for path in files:
        cube = np.load(path, mmap_mode="r")
        if cube.shape[-1] != bands:
            raise ValueError(f"Band mismatch in {path}: {cube.shape}")
        pixels = cube.reshape(-1, bands)
        for start in range(0, len(pixels), args.chunk_pixels):
            coordinates = (
                np.asarray(
                    pixels[start : start + args.chunk_pixels], dtype=np.float64
                )
                @ null_basis
            )
            mean, scatter, count = _update(mean, scatter, count, coordinates)
    covariance = scatter / max(count - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    number = min(args.components, dimension)
    components = (null_basis @ eigenvectors[:, :number]).T
    for index in range(number):
        maximum = int(np.argmax(np.abs(components[index])))
        if components[index, maximum] < 0:
            components[index] *= -1
    ratio = eigenvalues / max(float(eigenvalues.sum()), 1e-12)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / f"pca_components_n{number}.npy", components.astype(np.float32))
    np.save(output / "wavelengths_400_700_10nm.npy", wavelengths.astype(np.int32))
    summary = {
        "method": "OptionA: PCA in null coordinates z=x@N",
        "files": len(files),
        "pixels": count,
        "bands": bands,
        "camera_rank": rank,
        "null_dimension": dimension,
        "components": number,
        "explained_variance_ratio": ratio[:number].tolist(),
        "cumulative_explained_variance": np.cumsum(ratio[:number]).tolist(),
    }
    (output / "pca_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
