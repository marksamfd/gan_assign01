from pathlib import Path

import typer

from gan_assign01.config import EPOCHS, MODELS_DIR, PROCESSED_DATA_DIR
from gan_assign01.dataset import get_dataset_regions, load_region_dataset
from gan_assign01.modeling.models.ae import build_autoencoder
from gan_assign01.modeling.models.vae import build_vae
from gan_assign01.plots import (
    show_generated_samples_vae,
    show_latent_space_ae,
    show_latent_space_vae,
    show_reconstruction_ae,
    show_reconstruction_vae,
)

app = typer.Typer()



@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    regions = get_dataset_regions()
    results = {region: {"ae":False, "vae":False} for region in regions}

    for region in regions:

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
            show_latent_space_ae(ae, ds, region)
        # Save models
            ae.save(MODELS_DIR / f"{region}_AE.h5")
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

            show_reconstruction_vae(encoder, decoder, ds, region)
            show_latent_space_vae(encoder, ds, region)
            show_generated_samples_vae(decoder, region)
            vae.save_weights(MODELS_DIR / f"{region}_VAE.weights.h5")
            
            results[region]["vae"] = True


if __name__ == "__main__":
    app()
