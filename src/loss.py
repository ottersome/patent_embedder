import logging
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .utils.general import setup_logger


class MarginilizedLoss(nn.Module):
    def __init__(self, margin=0.1, hyp1=1, hkyp2=2, hyp3=3):
        super(MarginilizedLoss, self).__init__()
        self.logger = setup_logger(
            "MarginilizedLoss", "MarginilizedLoss.log", level=logging.DEBUG
        )

    def forward(self, x: List[Tensor]):
        """
        Will receive three different samples and will compute their loss
        based on the distance.
        args:
            x : List[Tensor]. The first n-1 tensors are positive samples whereas the last one is the negative sample.
                ref x batch_size x embedding_dim
        """
        # self.logger.debug(f"input x for loss is x: {x}")
        intra_distance = torch.sum(
            torch.stack([self._distance(x[0], neighbor) for neighbor in x]), dim=0
        )
        # self.logger.debug(f"intra_distance: {intra_distance}")
        inter_distance = self._distance(x[0], x[-1])
        # self.logger.debug(f"inter_distance: {inter_distance}")
        difference = intra_distance - inter_distance + 1 * len(x[:-1])
        # self.logger.debug(f"difference: {difference}")
        zeros_like = torch.zeros_like(difference)
        return torch.max(torch.stack([difference, zeros_like]), dim=0)[0]

    def _distance(self, x: Tensor, y: Tensor) -> Tensor:
        epsilon = 1e-10
        return torch.sqrt(
            torch.sum(torch.pow(x - y, 2), dim=-1) + epsilon
        )  # Returns unit [x] tensor
