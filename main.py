import anndata
from scMGCL import scMGCL
from evaluate_metrics import evaluate_model
import scanpy as sc


# load data
adata_atac = anndata.read_h5ad('ATAC.h5ad')
adata_rna = anndata.read_h5ad('RNA.h5ad')

adata_atac.obs['source'] == 'ATAC'
adata_rna.obs['source'] == 'RNA'


adata = adata_atac.concatenate(adata_rna,join='inner')

cell_type='cell_type'
adata.obs['cell_type']=adata.obs[f'{cell_type}']

# the scMGCL function is called to return the trained adata directly
integrated = scMGCL(adata)

RNA=integrated[integrated.obs['source'] == 'RNA']
ATAC=integrated[integrated.obs['source'] == 'ATAC']

z_rna=RNA.obsm['integrated_embeddings']
z_atac=ATAC.obsm['integrated_embeddings']

rna_cell_types=RNA.obs['cell_type']
atac_cell_types=ATAC.obs['cell_type']

results = evaluate_model(z_rna, z_atac, rna_cell_types, atac_cell_types, integrated)

# UMAP visualization
sc.set_figure_params(dpi=400, fontsize=10)
sc.pp.neighbors(integrated,n_neighbors=30)
sc.tl.umap(integrated,min_dist=0.1)

sc.pl.umap(integrated, color=['dataset','cell_type'],title=['',''],wspace=0.3, legend_fontsize=10)

# save results
integrated.write('scMGCL_integrated.h5ad')


