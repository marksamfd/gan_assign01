from json import encoder
from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import matplotlib.pyplot as plt

from gan_assign01.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def show_reconstruction_ae(model, dataset, region):

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
    plt.savefig(f"{region}-ae.png")
    plt.show()

def show_reconstruction_vae(encoder, decoder, dataset, region):
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
    plt.show()
    plt.savefig(f"{region}-vae.png")


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
