from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Model

from gan_assign01.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def _ensure_figures_dir():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _flatten_dataset_samples(dataset, max_points=1000):
    samples = []
    collected = 0

    for batch in dataset:
        x = batch[0] if isinstance(batch, tuple) else batch
        x_np = x.numpy()
        samples.append(x_np)
        collected += x_np.shape[0]
        if collected >= max_points:
            break

    if not samples:
        raise ValueError("Dataset is empty, cannot build latent visualization.")

    x_data = np.concatenate(samples, axis=0)
    return x_data[:max_points]


def _pca_projection(latent_vectors, n_components):
    z = latent_vectors - latent_vectors.mean(axis=0, keepdims=True)
    cov = np.cov(z, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    basis = eigvecs[:, order[:n_components]]
    return z @ basis


def _project_for_plot(latent_vectors, dims):
    if latent_vectors.shape[1] >= dims:
        return latent_vectors[:, :dims]
    return _pca_projection(latent_vectors, dims)


def _save_2d_scatter(points, title, output_path):
    plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 0], points[:, 1], s=10, alpha=0.7)
    plt.title(title)
    plt.xlabel("Latent 1")
    plt.ylabel("Latent 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _save_3d_scatter(points, title, output_path):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=10, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Latent 1")
    ax.set_ylabel("Latent 2")
    ax.set_zlabel("Latent 3")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def show_latent_space_ae(autoencoder, dataset, region, max_points=1000):
    _ensure_figures_dir()
    x_data = _flatten_dataset_samples(dataset, max_points=max_points)

    latent_layer = autoencoder.get_layer("latent_vector")
    encoder = Model(inputs=autoencoder.input, outputs=latent_layer.output)
    latent_vectors = encoder.predict(x_data, verbose=0)

    points_2d = _project_for_plot(latent_vectors, dims=2)
    points_3d = _project_for_plot(latent_vectors, dims=3)

    output_2d = FIGURES_DIR / f"{region}-ae-latent-2d.png"
    output_3d = FIGURES_DIR / f"{region}-ae-latent-3d.png"

    _save_2d_scatter(points_2d, f"{region} - AE Latent Space (2D)", output_2d)
    _save_3d_scatter(points_3d, f"{region} - AE Latent Space (3D)", output_3d)


def show_latent_space_vae(encoder, dataset, region, max_points=1000):
    _ensure_figures_dir()
    x_data = _flatten_dataset_samples(dataset, max_points=max_points)

    z_mean, _, _ = encoder.predict(x_data, verbose=0)

    points_2d = _project_for_plot(z_mean, dims=2)
    points_3d = _project_for_plot(z_mean, dims=3)

    output_2d = FIGURES_DIR / f"{region}-vae-latent-2d.png"
    output_3d = FIGURES_DIR / f"{region}-vae-latent-3d.png"

    _save_2d_scatter(points_2d, f"{region} - VAE Latent Space (2D)", output_2d)
    _save_3d_scatter(points_3d, f"{region} - VAE Latent Space (3D)", output_3d)


def show_generated_samples_vae(decoder, region, n_samples=25, seed=42):
    _ensure_figures_dir()

    latent_dim = decoder.input_shape[-1]
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n_samples, latent_dim)).astype(np.float32)
    generated = decoder.predict(z, verbose=0)

    cols = int(np.ceil(np.sqrt(n_samples)))
    rows = int(np.ceil(n_samples / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.array(axes).reshape(-1)

    for i in range(rows * cols):
        axes[i].axis("off")
        if i < n_samples:
            axes[i].imshow(generated[i].squeeze(), cmap="gray")

    plt.suptitle(f"{region} - VAE Generated Samples")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{region}-vae-generated.png", dpi=150)
    plt.close(fig)


def show_reconstruction_ae(model, dataset, region):

    _ensure_figures_dir()

    batch = next(iter(dataset.take(1)))

    # if dataset gives (x,y)
    if isinstance(batch, tuple):
        sample = batch[0]
    else:
        sample = batch

    sample = sample[:5]

    recon = model.predict(sample, verbose=0)

    plt.figure(figsize=(12,4))

    for i in range(5):

        plt.subplot(2,5,i+1)
        plt.imshow(sample[i].numpy().squeeze(), cmap="gray")
        plt.axis("off")

        plt.subplot(2,5,i+6)
        plt.imshow(recon[i].squeeze(), cmap="gray")
        plt.axis("off")

    plt.suptitle(f"{region} - AE Reconstruction")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{region}-ae.png", dpi=150)
    plt.close()

def show_reconstruction_vae(encoder, decoder, dataset, region):
    _ensure_figures_dir()
    batch = next(iter(dataset))
    if isinstance(batch, tuple):
          sample = batch[0]
    else:
          sample = batch

    sample = sample[:5]
    z_mean, _, z = encoder(sample)
    recon = decoder(z)

    plt.figure(figsize=(12,4))

    for i in range(5):
        plt.subplot(2,5,i+1)
        plt.imshow(sample[i].numpy().squeeze(), cmap='gray')
        plt.axis('off')

        plt.subplot(2,5,i+6)
        plt.imshow(recon[i].numpy().squeeze(), cmap='gray')
        plt.axis('off')

    plt.suptitle(f"{region} - VAE Reconstruction")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{region}-vae.png", dpi=150)
    plt.close()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating plot from data...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Plot generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
