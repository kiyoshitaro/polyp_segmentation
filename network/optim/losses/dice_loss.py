import torch
import torch.nn as nn
import torch.nn.functional as F


class dice_loss(nn.Module):
    def __init__(self, n_classes=2):
        super(dice_loss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        pred = torch.sigmoid(pred)
        inter = ((pred * mask)).sum(dim=(2, 3))
        union = ((pred + mask)).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return wiou.mean()
