from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer


from tensorflow.keras import layers, Model
from tensorflow.keras.losses import binary_crossentropy

from gan_assign01.config import EPOCHS, IMG_SIZE, LATENT_DIM, MODELS_DIR, PROCESSED_DATA_DIR
from gan_assign01.dataset import get_dataset_regions, load_region_dataset
from gan_assign01.modeling.models.ae import build_autoencoder
from gan_assign01.modeling.models.vae import build_vae
from gan_assign01.plots import show_reconstruction_ae

app = typer.Typer()



@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    regions = get_dataset_regions()
    results = {region: {"ae":False, "vae":False} for region in regions}

    for region in regions():

        print("="*60)
        print("Training Region:", region)

        ds = load_region_dataset(region)

        # ---------------------------
        # AUTOENCODER
        # ---------------------------
        if not results[region]["ae"]:
            print("="*15, "AutoEncoder", "="*15)
            
            ae = build_autoencoder()

            ae.fit(
                ds,
                epochs=EPOCHS,
                verbose=1
            )

            show_reconstruction_ae(ae, ds, region)
        # Save models
            ae.save(f"{region}_AE.h5")
            results[region]["ae"] = True
        # ---------------------------
        # VAE
        # ---------------------------
        if not results[region]["vae"]:
            print("="*15, "Variational AutoEncoder", "="*15)
            vae, encoder, decoder = build_vae()

            vae.fit(
                ds,
                epochs=EPOCHS,
                verbose=1
            )

            
            results[region]["vae"] = True


if __name__ == "__main__":
    app()
