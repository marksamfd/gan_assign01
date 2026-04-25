from pathlib import Path

import matplotlib.pyplot as plt
from loguru import logger
import numpy as np
import tensorflow as tf
import typer

from gan_assign01.config import LATENT_DIM, MODELS_DIR, PROCESSED_DATA_DIR
from gan_assign01.dataset import get_dataset_regions, load_region_dataset
from gan_assign01.modeling.models.vae import build_vae

app = typer.Typer()


def _save_reconstruction_grid(original, reconstructed, output_path, title):
    n_samples = original.shape[0]
    fig = plt.figure(figsize=(max(8, n_samples * 1.8), 4))

    for i in range(n_samples):
        plt.subplot(2, n_samples, i + 1)
        plt.imshow(original[i].squeeze(), cmap="gray")
        plt.axis("off")

        plt.subplot(2, n_samples, i + n_samples + 1)
        plt.imshow(reconstructed[i].squeeze(), cmap="gray")
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_generated_grid(generated, output_path, title):
    n_samples = generated.shape[0]
    cols = int(np.ceil(np.sqrt(n_samples)))
    rows = int(np.ceil(n_samples / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.array(axes).reshape(-1)

    for i in range(rows * cols):
        axes[i].axis("off")
        if i < n_samples:
            axes[i].imshow(generated[i].squeeze(), cmap="gray")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def _get_input_batch(dataset, n_samples):
    batch = next(iter(dataset.take(1)))
    x = batch[0] if isinstance(batch, tuple) else batch
    return x[:n_samples].numpy()


@app.command()
def main(
    region: str = typer.Option(None, help="Dataset region/folder name to run inference on."),
    model_type: str = typer.Option("vae", help="Model type: 'ae' or 'vae'."),
    n_samples: int = typer.Option(8, min=1, max=64, help="Number of samples for outputs."),
    predictions_dir: Path = typer.Option(
        PROCESSED_DATA_DIR / "predictions",
        help="Directory to store generated predictions and plots.",
    ),
    model_dir: Path = typer.Option(MODELS_DIR, help="Directory containing trained models."),
    seed: int = typer.Option(42, help="Random seed used for VAE sample generation."),
):
    predictions_dir.mkdir(parents=True, exist_ok=True)

    available_regions = get_dataset_regions()
    if not available_regions:
        raise typer.BadParameter("No dataset regions found in data/raw.")

    if region is None:
        region = available_regions[0]
        logger.info(f"No region provided. Using first available region: {region}")

    if region not in available_regions:
        raise typer.BadParameter(
            f"Unknown region '{region}'. Available regions: {', '.join(available_regions)}"
        )

    model_type = model_type.lower().strip()
    if model_type not in {"ae", "vae"}:
        raise typer.BadParameter("model_type must be either 'ae' or 'vae'.")

    logger.info(f"Loading data for region: {region}")
    ds = load_region_dataset(region)
    x_input = _get_input_batch(ds, n_samples=n_samples)

    if model_type == "ae":
        ae_path = model_dir / f"{region}_AE.h5"
        if not ae_path.exists():
            raise typer.BadParameter(f"AE model not found: {ae_path}")

        logger.info(f"Loading AE model: {ae_path}")
        ae = tf.keras.models.load_model(ae_path, compile=False)
        reconstruction = ae.predict(x_input, verbose=0)

        np.save(predictions_dir / f"{region}_ae_input.npy", x_input)
        np.save(predictions_dir / f"{region}_ae_reconstruction.npy", reconstruction)

        _save_reconstruction_grid(
            x_input,
            reconstruction,
            predictions_dir / f"{region}_ae_prediction_grid.png",
            f"{region} - AE Predictions",
        )

    if model_type == "vae":
        vae_weights = model_dir / f"{region}_VAE.weights.h5"
        if not vae_weights.exists():
            raise typer.BadParameter(f"VAE weights not found: {vae_weights}")

        logger.info(f"Loading VAE weights: {vae_weights}")
        vae, encoder, decoder = build_vae()
        vae.load_weights(vae_weights)

        z_mean, _, z = encoder.predict(x_input, verbose=0)
        reconstruction = decoder.predict(z, verbose=0)

        rng = np.random.default_rng(seed)
        z_random = rng.normal(size=(n_samples, LATENT_DIM)).astype(np.float32)
        generated = decoder.predict(z_random, verbose=0)

        np.save(predictions_dir / f"{region}_vae_input.npy", x_input)
        np.save(predictions_dir / f"{region}_vae_latent.npy", z_mean)
        np.save(predictions_dir / f"{region}_vae_reconstruction.npy", reconstruction)
        np.save(predictions_dir / f"{region}_vae_generated.npy", generated)

        _save_reconstruction_grid(
            x_input,
            reconstruction,
            predictions_dir / f"{region}_vae_prediction_grid.png",
            f"{region} - VAE Reconstructions",
        )
        _save_generated_grid(
            generated,
            predictions_dir / f"{region}_vae_generated_grid.png",
            f"{region} - VAE Generated Samples",
        )

    logger.success(f"Inference complete. Outputs written to: {predictions_dir}")


if __name__ == "__main__":
    app()
