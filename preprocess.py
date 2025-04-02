import scanpy as sc
from sklearn.decomposition import PCA
import anndata

def preprocess_data(
        adata, 
        n_top=2000, 
        max_scale=10, 
        n_components=40
        ):
    # load data
    adata_combined = adata
    rna_data = adata_combined[adata_combined.obs['source'] == 'RNA']
    atac_data = adata_combined[adata_combined.obs['source'] == 'ATAC']

    # Preprocessing of RNA
    sc.pp.normalize_total(rna_data)
    sc.pp.log1p(rna_data)
    sc.pp.highly_variable_genes(rna_data, n_top_genes=n_top, inplace=False, subset=True)
    sc.pp.scale(rna_data, max_value=max_scale)

    # Preprocessing of ATAC's gene activity matrix
    sc.pp.normalize_total(atac_data)
    sc.pp.log1p(atac_data)
    sc.pp.highly_variable_genes(atac_data, n_top_genes=n_top, inplace=False, subset=True)
    sc.pp.scale(atac_data, max_value=max_scale)

    # Data dimensionality reduction
    pca = PCA(n_components)
    rna_pca = pca.fit_transform(rna_data.X)
    atac_pca = pca.fit_transform(atac_data.X)

    return rna_pca, atac_pca
