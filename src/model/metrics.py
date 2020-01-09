import torch
import torch.nn as nn
import numpy as np
from skimage.morphology import label


class EPE(nn.Module):
    """The End Point Error.
    """
    def __init__(self):
        super().__init__()

    def forward(self, output, target, mask=None):
        """
        Args:
            output (torch.Tensor) (N, C, *): The model output.
            target (torch.LongTensor) (N, 1, *): The data target.
        Returns:
            metric (torch.Tensor) (C): The dice scores for each class.
        """
        # Get the one-hot encoding of the prediction and the ground truth label.

        # maxi = torch.max(target)
        # mini = torch.min(target)
        # output = (output - mini) / (maxi - mini)
        # target = (target - mini) / (maxi - mini)
        flow = target['flow']
        flow_pred = output['flow']
        if mask is not None:
            epe = torch.norm(flow - flow_pred, p=2, dim=1)
            epe = epe * mask
            epe = torch.sum(epe) / torch.sum(mask)
        else:
            epe = torch.norm(flow - flow_pred, p=2, dim=1).mean()

        return epe


class F1Score(nn.Module):
    """The accuracy
    """
    def __init__(self, threshold):
        super().__init__()
        self.th = threshold

    def forward(self, output, target, mask=None):
        """
        Args:
            output (torch.Tensor) (N, C, *): The model output.
            target (torch.LongTensor) (N, 1, *): The data target.
        Returns:
            metric (torch.Tensor) (C): The dice scores for each class.
        """
        # Get the one-hot encoding of the prediction and the ground truth label.
        flow = target['flow']
        flow_pred = output['flow']
        err = torch.norm(flow_pred-flow, p=2, dim=1)
        flow_len = torch.norm(flow, p=2, dim=1)

        # print(error.shape)
        # print((error <= self.th).float().shape)
        # print(mask.shape)
        # print(((error/gtflow_len <= self.th)*mask))

        if mask is not None:
            f1 = ((err/flow_len <= self.th) * mask.byte())
            f1 = torch.sum(f1.float())
            f1 = f1 / (torch.sum(mask) + 1e-20)
        else:
            f1 = (err/flow_len <= self.th).byte()
            f1 = torch.mean(f1.float())

        return f1

