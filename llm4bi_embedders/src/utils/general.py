import argparse
import logging
import os

import lightning as L
from google.cloud import storage


def getargs():
    argparser = argparse.ArgumentParser(
        description="PyTorch implementation of Document Embeddings"
    )

    # General Arguments
    argparser.add_argument(
        "--data-path",
        type=str,
        default="dataset/",
        help="Location of the data corpus, already stored as a parquet file",
    )
    argparser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    argparser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    argparser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    argparser.add_argument("--seed", type=int, default=420, help="Random seed")
    argparser.add_argument("--cuda", action="store_true", help="use CUDA")
    argparser.add_argument("--wandb", action="store_true", help="Report to wandb")
    argparser.add_argument(
        "--save", type=str, default="model.pt", help="Path to save the final model"
    )
    argparser.add_argument(
        "--load", type=str, default="", help="Path to load the model"
    )
    argparser.add_argument(
        "--log_interval", type=int, default=200, help="report interval"
    )
    argparser.add_argument(
        "--encoder_base",
        type=str,
        default="allenai/scibert_scivocab_uncased",
        help="Encoder base",
    )
    argparser.add_argument(
        "--encoder_max_length",
        type=int,
        default=512,
        help="Encoder max length",
    )
    # For CGS
    argparser.add_argument(
        "--gcs_credentials",
        default="gcs_service_key.json",
        help="Credentials for Google Cloud Storage",
    )
    argparser.add_argument("--gcs_bucketname", default="llmbi-models")
    argparser.add_argument("--gcs_blobname", default="checkpoints/model.ckpt")

    # Arguments for psql
    argparser.add_argument("--psql_host", required=True)
    argparser.add_argument("--psql_database", required=True)
    argparser.add_argument("--psql_user", required=True)
    argparser.add_argument("--psql_password", required=True)
    argparser.add_argument("--psql_port", required=True)

    # Table Specific
    argparser.add_argument("--psql_relationships_tablename", default="samples_large")
    argparser.add_argument("--psql_reltexts_tablename", default="samples_large_text")

    return argparser.parse_args()


def setup_logger(name, file="main.log", level=logging.INFO):
    os.makedirs("~/.llm4bi/logs", exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    expand_user = os.path.join(os.path.expanduser("~"), ".llm4bi/logs/")
    os.makedirs(expand_user, exist_ok=True)
    fh = logging.FileHandler(os.path.join(expand_user, file), "w")
    sh = logging.StreamHandler()
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    fh.setLevel(level)
    sh.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def upload_checkpoint_to_gcs(model: L.LightningModule, bucketname, blob_name):
    local_path = "checkpoints/model.ckpt"

    # Initialize GCS clien
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucketname)
    blob = bucket.blob(blob_name)

    # Upload checkpoint to CGS
    blob.upload_from_filename(local_path)
