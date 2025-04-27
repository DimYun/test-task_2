"""Module for self-written Loss functions"""

import torch
from torch import nn


class BoundaryDoULoss(nn.Module):
    """Boundary Do-U Loss function, based on paper: https://arxiv.org/abs/2308.00220"""
    def __init__(self, n_classes: int = 2) -> None:
        super(BoundaryDoULoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor: torch.Tensor) -> torch.Tensor:
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim = 1)
        return output_tensor.float()

    def _adaptive_size(
            self,
            score: torch.Tensor,
            target: torch.Tensor,
    ) -> torch.Tensor:
        kernel = torch.Tensor(
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        )
        padding_out = torch.zeros(
            (target.shape[0], target.shape[-2] + 2, target.shape[-1] + 2)
        )
        padding_out[:, 1:-1, 1:-1] = target
        h_pad, w_pad = 3, 3

        Y_matrix = torch.zeros(
            (
                padding_out.shape[0],
                padding_out.shape[1] - h_pad + 1,
                padding_out.shape[2] - w_pad + 1
            )
        ).cuda()
        for i in range(Y_matrix.shape[0]):
            Y_matrix[i, :, :] = torch.conv2d(
                target[i].unsqueeze(0).unsqueeze(0),
                kernel.unsqueeze(0).unsqueeze(0).cuda(),
                padding=1
            )
        Y_matrix = Y_matrix * target
        Y_matrix[Y_matrix == 5] = 0
        C_score = torch.count_nonzero(Y_matrix)
        S_score = torch.count_nonzero(target)
        smooth = 1e-5
        alpha = 1 - (C_score + smooth) / (S_score + smooth)
        alpha = 2 * alpha - 1

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        alpha = min(
            alpha,
            0.8
        )  ## We recommend using a truncated alpha of 0.8,
        # as using truncation gives better results on some datasets and has rarely effect on others.
        loss = (
                (z_sum + y_sum - 2 * intersect + smooth) /
                (z_sum + y_sum - (1 + alpha) * intersect + smooth)
        )
        return loss

    def forward(
            self,
            inputs: torch.Tensor,
            target: torch.Tensor,
    ) -> torch.Tensor:
        inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)

        assert (
                inputs.size() == target.size()
        ), 'predict {} & target {} shape do not match'.format(
            inputs.size(),
            target.size()
        )

        loss = 0.0
        for i in range(0, self.n_classes):
            loss += self._adaptive_size(inputs[:, i], target[:, i])
        return loss / self.n_classes
