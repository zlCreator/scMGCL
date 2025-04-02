import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class ImprovedGCNEncoder(nn.Module):
    def __init__(
            self, 
            input_dim, 
            hidden_dim, 
            output_dim
            ):
        super(ImprovedGCNEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)

    def forward(
            self, 
            data
                ):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)

        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        x = self.conv3(x, edge_index)
        x = torch.relu(x)
        return x


class ContrastiveGNN(nn.Module):
    def __init__(
            self, 
            input_dim, 
            hidden_dim, 
            output_dim
            ):
        super(ContrastiveGNN, self).__init__()
        self.encoder = ImprovedGCNEncoder(input_dim, hidden_dim, output_dim)
        self.projector = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(
            self, 
            data
            ):
        h = self.encoder(data)
        z = self.projector(h)
        return z
