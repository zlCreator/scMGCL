from sklearn.metrics import pairwise_distances
import numpy as np
from torch_geometric.data import Data
import torch

def create_graph_cosine(
        data, 
        k
        ):
    # calculate Euclidean distance
    dist_matrix = pairwise_distances(data, metric='euclidean')

    # csonvert the distance to similarity (weight), the smaller the distance
    sim_matrix = 1 / (1 + dist_matrix)  

    # set a value of 0 except for the first k nearest neighbors
    sorted_indices = dist_matrix.argsort(axis=1)[:, 1:k+1]  # take the k with the smallest distance (excluding yourself)
    adj = np.zeros_like(dist_matrix)
    for i in range(adj.shape[0]):
        adj[i, sorted_indices[i]] = sim_matrix[i, sorted_indices[i]]  # the similarity of K-nearest neighbors is retained as a weight

    # gets the index of non-zero elements
    edge_index = np.array(np.nonzero(adj)).T
    edge_attr = adj[edge_index[:, 0], edge_index[:, 1]]
    return edge_index, edge_attr


def create_graphs(
        rna_pca, 
        atac_pca, 
        k
        ):
    rna_edge_index, rna_edge_attr = create_graph_cosine(rna_pca, k)
    atac_edge_index, atac_edge_attr = create_graph_cosine(atac_pca, k)

    rna_graph = Data(x=torch.tensor(rna_pca, dtype=torch.float),
                     edge_index=torch.tensor(rna_edge_index.T, dtype=torch.long))
    atac_graph = Data(x=torch.tensor(atac_pca, dtype=torch.float),
                      edge_index=torch.tensor(atac_edge_index.T, dtype=torch.long))

    return [rna_graph, atac_graph]