import logging
from typing import List

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch import Tensor
from transformers import AutoConfig, AutoModel, PreTrainedTokenizer

from .loss import MarginilizedLoss
from .utils.general import setup_logger


class LightningMod_DocEmbedder(L.LightningModule):
    def __init__(
        self,
        lang_head_name,
        tokenizer: PreTrainedTokenizer,
        load_init_weights=True,
    ):
        super().__init__()
        self.mylogger = setup_logger(
            "LightningMod_DocEmbedder",
            "LightningMod_DocEmbedder.log",
            level=logging.INFO,
        )
        self.lang_head_config = AutoConfig.from_pretrained(
            lang_head_name, output_hidden_states=True
        )
        # self.my_head = nn.Linear(
        # self.lang_head_config.hidden_size, self.lang_head_config.hidden_size
        # )
        # For batch_size finding
        if load_init_weights:  # Actually downloads the weights (from HF likely)
            self.mylogger.info("⏬Downloading Model")
            # Create config so we can get hidden ouputs
            config = AutoConfig.from_pretrained(
                lang_head_name, output_hidden_states=True
            )
            self.lang_head = AutoModel.from_pretrained(
                lang_head_name
            )  # , config=config)
        else:  # This model has the correct architecture but randomly initalized weights
            self.mylogger.info("🛠️ Configuring Model Architecture")
            self.lang_head = AutoModel.from_config(self.lang_head_config)

        # TODO: remove this from loss. Its was only for bug huntin 🐛
        self.loss = MarginilizedLoss(tokenizer)

    def forward(self, x, attention_mask):
        return self.lang_head(x, attention_mask=attention_mask)

    def training_step(self, ref_batches, batch_idx):
        # Get embeddings
        # self.mylogger.debug(f"ref_batches: {ref_batches}")
        total_norm = 0.0
        for p in self.lang_head.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)

        self.mylogger.debug(f"Total nurm at {batch_idx} is {total_norm}")
        # self.mylogger.debug(f"reg_batches: looks like {ref_batches})")
        contextualized_embeddings_batches = []
        self.mylogger.debug(f"Number of batches: {len(ref_batches)}")
        for batch in ref_batches:
            # Create attention mask online for the batch
            attention_mask = torch.ones_like(batch)
            attention_mask[batch == 0] = 0
            # cont_embeddings = self.lang_head(
            # batch, attention_mask=attention_mask
            # ).hidden_states[-1]
            # TODO: trying type_ids
            cont_embeddings = self.lang_head(batch, attention_mask=attention_mask)
            # lm_output = cont_embeddings.pooler_output
            # lm_output = self.my_head(cont_embeddings.last_hidden_state[:, 0, :])
            lm_output = cont_embeddings.last_hidden_state[:, 0, :]
            self.mylogger.debug(
                f"The output of the network looks like : {lm_output.shape}"
            )
            # cont_embeddings = self.lang_head(batch).hidden_states[-1]
            contextualized_embeddings_batches.append(lm_output)

        # Once We have embeddings we can compute the loss
        loss = self.loss(contextualized_embeddings_batches)
        loss_avg = loss.mean()  # Average across batches
        self.log(
            "train_loss", loss_avg.item(), prog_bar=True, on_step=True, on_epoch=True
        )
        return loss_avg

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        # Warmup scheduler
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        # optimizer, T_max=10, eta_min=0
        # )
        # return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        return optimizer

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_descripion("Validation")
        return bar

        # def validation_step(self, ref_batches: List[Tensor], batch_idx):
        # Get embedding
        # self.mylogger.debug(f"ref_batches: {ref_batches}")
        cont_embeddings = []
        # for batch in ref_batches:
        #    attention_mask = torch.ones_like(batch)
        #    attention_mask[batch == 0] = 0
        #    docs_embeds = self.lang_head(
        #        batch, attention_mask=attention_mask
        #    )  # .hidden_states[-1]
        #    # cont_embeddings.append(docs_embeds[:, 0, :])
        ## Once We have embeddings we can compute the loss
        # loss = self.loss(0.1)
        # loss_avg = loss.mean()
        # self.log("val/loss", loss_avg.item())
        # return loss_avg
