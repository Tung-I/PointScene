import torch
import torch.nn as nn


class MyL2Loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, target, mask=None):
        flow = target['flow']
        flow_pred = output['flow']

        # minimum = torch.min(flow)
        # maximum = torch.max(flow)
        # output = (output - minimum) / (maximum - minimum)
        # flow = (flow - minimum) / (maximum - minimum)
        if mask is not None:
            err = torch.norm(flow - flow_pred, p=2, dim=1)
            err = err * mask
            loss = torch.sum(err) / torch.sum(mask)
        else:
            loss = torch.norm(flow - flow_pred, p=2, dim=1).mean()

        return loss
