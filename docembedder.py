import logging
import os

import torch
from google.cloud import storage
from transformers import AutoModel, AutoTokenizer

from src.lightningMods import LightningMod_DocEmbedder
from src.utils.general import setup_logger


class DocRep:
    def __init__(self, title, abstract):
        self.title = title
        self.abstract = abstract
        self.data = "<title>" + self.title + "[SEP]<abstract>" + self.abstract


class DocEmbedder:
    BUCKET_NAME: str = "llmbi-models"
    TARGET_PATH: str = ".llm4bi/"
    FOUNDATION_MODEL = "allenai/scibert_scivocab_uncased"
    ENCODER_MAX_LENGTH = 512

    def __init__(self, blob_name: str):
        self.logger = setup_logger("DocEmbWrapper", "main.log", logging.INFO)
        home_path = os.path.expanduser("~")
        target_dir = os.path.join(home_path, self.TARGET_PATH)
        target_file: str = os.path.join(target_dir, blob_name)
        full_dir = os.path.dirname(target_file)

        self.tokenizer = AutoTokenizer.from_pretrained(self.FOUNDATION_MODEL)

        os.makedirs(full_dir, exist_ok=True)
        self.blob_name = blob_name

        # Check for cache file
        if os.path.exists(target_file):
            self.logger.info(
                f"Cache found. Loading your model {blob_name} from dir {target_dir}"
            )
        else:
            # Load from CGS
            self.logger.info(
                f"Cache not found. Downloading your model {blob_name} from GCS into dir {target_dir}"
            )
            self._dload_from_gcs(target_file)
            self.logger.info("Download complete")
        # Load into lightining
        self.model = LightningMod_DocEmbedder.load_from_checkpoint(
            target_file,
            lang_head_name=self.FOUNDATION_MODEL,
            load_init_weights=False,
            strict=False,
        )

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
        self.model.eval()
        embed1 = (
            self.model(
                torch.Tensor(
                    self.tokenizer.encode(
                        doc1.data,
                        max_length=self.ENCODER_MAX_LENGTH,
                        truncation=True,
                        padding="max_length",
                    )
                )
                .to(torch.long)
                .view(1, -1)
            )
            .hidden_states[-1][:, 0, :]
            .detach()
            .squeeze()
        )
        embed2 = (
            self.model(
                torch.Tensor(
                    self.tokenizer.encode(
                        doc2.data,
                        max_length=self.ENCODER_MAX_LENGTH,
                        truncation=True,
                        padding="max_length",
                    )
                )
                .to(torch.long)
                .view(1, -1)
            )
            .hidden_states[-1][:, 0, :]
            .detach()
            .squeeze()
        )
        # self.logger.info(f"embed1: {embed1}\n embed2: {embed2})")

        return torch.norm(
            embed1 - embed2,
            p=2,
            dim=0,
            keepdim=True,
        )

    def dist(self, doc: DocRep, docRep):
        return
