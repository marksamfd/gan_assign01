import os
from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import tensorflow as tf

from gan_assign01.config import BATCH_SIZE, IMG_SIZE, PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


def download_dataset():
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Downloading dataset...")
    import kagglehub

    # Download latest version
    kaggle_path = kagglehub.dataset_download(
        "andrewmvd/medical-mnist", path=RAW_DATA_DIR
    )

    print("Path to dataset files:", kaggle_path)
    logger.success("Dataset download complete.")
    # -----------------------------------------


def get_dataset_regions():
    return sorted(
        [
            folder
            for folder in os.listdir(RAW_DATA_DIR)
            if os.path.isdir(os.path.join(RAW_DATA_DIR, folder))
        ]
    )


def load_region_dataset(region):

    path = os.path.join(RAW_DATA_DIR, region)

    ds = tf.keras.utils.image_dataset_from_directory(
        path,
        labels=None,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        image_size=(IMG_SIZE, IMG_SIZE),
        shuffle=True,
    )

    # normalize
    ds = ds.map(lambda x: x / 255.0)

    # for Autoencoder target = input
    ds = ds.map(lambda x: (x, x))

    return ds


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Downloading dataset...")
    download_dataset()
    # -----------------------------------------


if __name__ == "__main__":
    app()
