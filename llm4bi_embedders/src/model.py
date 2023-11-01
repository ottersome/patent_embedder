"""
Written by @ottersome - Luis Garcia
Will create embeddings in vector space out of documents
"""
import logging
from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from .utils.general import setup_logger


class DocumentEmbeddor(nn.Module):
    def __init__(self, embedding_dim, lh_model: nn.Module):
        super(DocumentEmbeddor, self).__init__()

        self.mylogger = setup_logger(
            "DocumentEmbeddor", "DocumentEmbeddor.log", level=logging.INFO
        )

        self.language_head = lh_model
        self.embedding_dim = embedding_dim

    def forward(self, token_batches: List[Tensor]):
        """
        For convenience we will have the forward pass take a tuple of papers
        and return their emebeddings. We will handle loss outside of the model
        args:
            tokens: List[Tensor] where each element might be a batched tensor
        """
        contextualized_embeddings_batches = []
        for batch in token_batches:
            # self.mylogger.debug(f"batch: {batch}")
            contextualized_batch = self.language_head(batch).hidden_states[
                -1
            ]  # Last hidden layer is at end of tuple

            # print(f"contextualized_batch: {contextualized_batch}")
            # TODO: we might want to add a pooling layer here or a linear layer to reduce the dimensionality

            # HACK : we have to be careful not to be averaging over all-zero vectors
            # Each element in the batch will be embedding_dim*num_embeddings. We want to average over num_embeddings:
            # contextualized_embeddings_batches.append(
            #    torch.mean(contextualized_batch, dim=1)
            # )
            # Get the first token ([CLS]) of each batch
            contextualized_embeddings_batches.append(contextualized_batch[:, 0, :])
        return contextualized_embeddings_batches
