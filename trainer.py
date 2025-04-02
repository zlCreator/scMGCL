import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from model import ContrastiveGNN
from contrastive_loss import contrastiveLoss
import numpy as np

class Trainer:
    def __init__(
            self, 
                 graphs, 
                 input_dim, 
                 hidden_dim, 
                 output_dim, 
                 lr, 
                 batch_size, 
                 epochs, 
                 device
                 ):
        self.graphs = graphs
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device

        self.model = ContrastiveGNN(input_dim, hidden_dim, output_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr)
        self.contrastive_loss_fn = ContrastiveLoss()
        self.dataloader = DataLoader(graphs, batch_size=batch_size, shuffle=True)

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for data in self.dataloader:
                data = data.to(self.device)
                self.optimizer.zero_grad()
                z_rna = self.model(data[0])
                z_atac = self.model(data[1])
                cons_loss = self.contrastive_loss_fn(z_rna, z_atac)
                loss = cons_loss
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f'Epoch {epoch+1}, Loss: {total_loss/len(self.dataloader)}')

    def get_embeddings(self):
        self.model.eval()
        embeddings = []
        with torch.no_grad():
            rna_graph = self.graphs[0]
            atac_graph = self.graphs[1]
            rna_graph = rna_graph.to(self.device)
            atac_graph = atac_graph.to(self.device)
            z_rna = self.model(rna_graph)
            z_atac = self.model(atac_graph)
            z_rna = z_rna.cpu().numpy()
            z_atac = z_atac.cpu().numpy()
            # stitching embedding results
            embeddings = np.concatenate((z_atac, z_rna), axis=0)
        return embeddings
