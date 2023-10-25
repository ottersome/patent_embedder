import logging
import os

import torch
from google.cloud import storage

from src.lightningMods import LightningMod_DocEmbedder
from src.utils.general import setup_logger


class DocRep:
    def __init__(self, title, abstract):
        self.title = title
        self.abstract = abstract
        self.data = "<title>" + self.title + "[SEP]<abstract>" + self.abstract


class DocEmbedder:
    BUCKET_NAME: str = "llmbi-models"
    TARGET_PATH: str = "~/.llm4bi/"

    def __init__(self, blob_name: str):
        self.logger = setup_logger("DocEmbWrapper", "main.log", logging.INFO)
        target_file: str = os.path.join(self.TARGET_PATH, blob_name)
        target_dir = os.path.dirname(target_file)
        os.makedirs(target_dir, exist_ok=True)
        self.blob_name = blob_name
        # Load from CGS
        self.logger.info(
            f"Cache not found. Downloading your model {blob_name} from GCS into dir f{target_dir}"
        )
        self._dload_from_gcs(target_file)
        # Load into lightining
        self.logger.info("Dowloaded. Now Loading into Lightning")
        self.model = LightningMod_DocEmbedder.load_from_checkpoint(target_file)

    def _dload_from_gcs(self, target_file: str) -> None:
        if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") is None:
            # Ask for path from user
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = input(
                "Enter path to service key: "
            )
        storage_client = storage.Client()
        bucket = storage_client.bucket(self.BUCKET_NAME)
        blob = bucket.blob(self.blob_name)
        blob.download_to_filename(target_file)

    def __call__(self, doc: DocRep) -> torch.Tensor:
        return self.model(doc.data)

    def doc_dist(self, doc1: DocRep, doc2: DocRep) -> torch.Tensor:
        return torch.norm(
            self.model(doc1.data).squeeze() - self.model(doc2.data).squeeze(),
            p=2,
            dim=1,
            keepdim=True,
        )

    def dist(self, doc: DocRep, docRep):
        return
