import torch
import numpy as np
from preprocess import preprocess_data
from graph_construct import create_graphs
from trainer import Trainer

def scMGCL(
        adata,
        n_components=30,
        hidden_dim=300,
        nk=20,
        lr=0.0006,
        batch_size=64,
        epochs=800,
        seed=124
):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set up random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load and preprocess data
    rna_pca, atac_pca  = preprocess_data(adata, n_components)

    # Create graph structure
    graphs = create_graphs(rna_pca, atac_pca, nk)

    # model training
    trainer = Trainer(graphs, n_components, hidden_dim, n_components, lr, batch_size, epochs, device)
    trainer.train()

    # Store the results in adata 
    adata.obsm['integrated_embeddings'] = trainer.get_embeddings()

    return adata