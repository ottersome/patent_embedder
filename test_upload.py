import os

import lightning as L
from transformers import AutoConfig, AutoModelForMaskedLM, BertConfig

from src.lightningMods import LightningMod_DocEmbedder
from src.utils.general import upload_checkpoint_to_gcs

# import lightning model
print("Loading model from checkpoint")
model = LightningMod_DocEmbedder.load_from_checkpoint(
    "./checkpoints/model.ckpt", lang_head_name="allenai/scibert_scivocab_uncased"
)
# Create OS environment variable for GCS credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./gcs_service_key.json"

print("Uploading to cgs")
upload_checkpoint_to_gcs(
    model=model, bucketname="llmbi-models", blob_name="checkpoints/v0.ckpt"
)
print("Done")
