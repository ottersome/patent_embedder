import logging
from typing import List

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision as tv
from torch import Tensor
from transformers import AutoConfig, AutoModel

from .loss import MarginilizedLoss
from .utils.general import setup_logger


class LightningMod_DocEmbedder(L.LightningModule):
    def __init__(self, lang_head_name, load_init_weights=True):
        super().__init__()
        self.mylogger = setup_logger(
            "LightningMod_DocEmbedder",
            "LightningMod_DocEmbedder.log",
            level=logging.DEBUG,
        )
        self.lang_head_config = AutoConfig.from_pretrained(
            lang_head_name, output_hidden_states=True
        )
        if load_init_weights:  # Actually downloads the weights (from HF likely)
            self.mylogger.info("⏬Downloading Model")
            self.lang_head = AutoModel.from_pretrained(lang_head_name)
        else:  # This model has the correct architecture but randomly initalized weights
            self.lang_head = AutoModel.from_config(self.lang_head_config)
        self.batch_size = 4

        self.loss = MarginilizedLoss()

    def forward(self, x, batch_idx):
        return self.lang_head(x)

    def training_step(self, ref_batches: List[Tensor], batch_idx):
        # Get embeddings
        # self.mylogger.debug(f"ref_batches: {ref_batches}")

        contextualized_embeddings_batches = []
        for batch in ref_batches:
            cont_embeddings = self.lang_head(batch).hidden_states[-1]
            contextualized_embeddings_batches.append(cont_embeddings[:, 0, :])

        # Once We have embeddings we can compute the loss
        loss = self.loss(contextualized_embeddings_batches)
        loss_avg = loss.mean()  # Average across batches
        self.log("train/loss", loss_avg.item())
        return loss_avg

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # Warmup scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10, eta_min=0
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_descripion("Validation")
        return bar

    def validation_step(self, ref_batches: List[Tensor], batch_idx):
        # Get embeddings
        # self.mylogger.debug(f"ref_batches: {ref_batches}")
        cont_embeddings = []
        for batch in ref_batches:
            docs_embeds = self.lang_head(batch).hidden_states[-1]
            cont_embeddings.append(docs_embeds[:, 0, :])
        # Once We have embeddings we can compute the loss
        loss = self.loss(cont_embeddings)
        loss_avg = loss.mean()
        self.log("val/loss", loss_avg.item())
        return loss_avg
