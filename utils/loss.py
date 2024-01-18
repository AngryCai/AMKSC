import torch
import torch.nn as nn


# class TVLoss(nn.Module):
#     def __init__(self, TVLoss_weight=1.):
#         super(TVLoss, self).__init__()
#         self.TVLoss_weight = TVLoss_weight
#
#     def forward(self, x):
#         batch_size = x.size()[0]
#         h_x = x.size()[2]
#         w_x = x.size()[3]
#         count_h = self._tensor_size(x[:, :, 1:, :])
#         count_w = self._tensor_size(x[:, :, :, 1:])
#         h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
#         w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
#         # tv = self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size
#         tv = self.TVLoss_weight * (h_tv / count_h + w_tv / count_w) / batch_size
#         return tv
#
#     def _tensor_size(self, t):
#         return t.size()[1] * t.size()[2] * t.size()[3]

class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1.):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, img):
        """
        Compute total variation loss.
        Inputs:
        - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
        - tv_weight: Scalar giving the weight w_t to use for the TV loss.
        Returns:
        - loss: PyTorch Variable holding a scalar giving the total variation loss
          for img weighted by tv_weight.
        """
        w_variance = torch.sum(torch.pow(img[:, :, :, :-1] - img[:, :, :, 1:], 2))
        h_variance = torch.sum(torch.pow(img[:, :, :-1, :] - img[:, :, 1:, :], 2))
        loss = self.TVLoss_weight * (h_variance + w_variance)
        return loss


class net(nn.Module):

    def __init__(self, spatial_size, num_anchor, kernel='rbf', smooth_coef=1.2, lambda_tv=1., init_z=None):
        """
        :param spatial_size: (width, height), n_samples = width*height
        :param num_anchor:
        :param kernel:
        :param smooth_coef:
        :param lambda_tv:
        :param init_z:
        """
        super(net, self).__init__()
        self.spatial_size = spatial_size
        self.num_anchor = num_anchor
        self.init_z = init_z
        self.kernel = kernel
        self.smooth_coef = smooth_coef
        self.lambda_tv = lambda_tv
        self.criterion = nn.MSELoss()
        self.tv_loss = TVLoss(self.lambda_tv)
        self.Z = nn.Parameter(torch.ones(num_anchor, spatial_size[0] * spatial_size[1]), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_z is not None:
            if not isinstance(self.init_z, torch.Tensor):
                self.Z = torch.from_numpy(self.init_z).float()
            self.Z.data = self.init_z
        else:
            nn.init.xavier_uniform(self.Z.data, gain=1)

    def forward(self, x_list, anchors, modality_weights):
        sum_ = 0.
        for a_, x_, mu_ in zip(anchors, x_list, modality_weights):
            # sum_ += 0.5 * self.criterion(x_, a_.matmul(self.Z)) * mu_.pow(self.smooth_coef)
            sum_ += 0.5 * torch.norm(x_ - a_.matmul(self.Z), p='fro').pow(2) * mu_.pow(self.smooth_coef)
        tv_term = 0.5 * self.tv_loss(self.Z.reshape((1, self.num_anchor, self.spatial_size[0], self.spatial_size[1])))
        return sum_ + tv_term
