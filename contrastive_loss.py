import torch
from torch import nn


class contrastiveLoss(nn.Module):
    def __init__(self):
        super(contrastiveLoss, self).__init__()

    def sim(
            self, 
            z1, 
            z2
            ):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        product_matrix = torch.mm(z1, z2.t())
        norm_product_matrix = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(product_matrix / (norm_product_matrix * 0.1))
        return sim_matrix

    def forward(
            self, 
            z_r, 
            z_a
            ):
        matrix = self.sim(z_r, z_a)
        matrix = matrix / (torch.sum(matrix, dim=1, keepdim=True) + 1e-8)
        positive_loss = -torch.log(matrix.diag()).mean()

        # negative samples (non-diagonal elements) to pull them apart
        batch_size = z_r.shape[0]
        negative_loss = -torch.log(1 - matrix + 1e-8) 
        negative_loss = negative_loss.sum() - negative_loss.diag().sum()  
        negative_loss /= (batch_size * (batch_size - 1))  # normalization

        return positive_loss+0.5*negative_loss
