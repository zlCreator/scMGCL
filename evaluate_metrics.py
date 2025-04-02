import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, f1_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
import scipy

def batch_entropy_mixing_score(
        data, 
        batches, 
        n_neighbors=50, 
        n_pools=50, 
        n_samples_per_pool=100
        ):
    """
    Calculate the batch mixing entropy score.
    """
    def entropy(batches):
        p = np.zeros(N_batches)
        adapt_p = np.zeros(N_batches)
        a = 0
        for i in range(N_batches):
            p[i] = np.mean(batches == batches_[i])
            a = a + p[i] / P[i]
        entropy = 0
        for i in range(N_batches):
            adapt_p[i] = (p[i] / P[i]) / a
            entropy = entropy - adapt_p[i] * np.log(adapt_p[i] + 10 ** -8)
        return entropy

    n_neighbors = min(n_neighbors, len(data) - 1)
    nne = NearestNeighbors(n_neighbors=1 + n_neighbors, n_jobs=8)
    nne.fit(data)
    kmatrix = nne.kneighbors_graph(data) - scipy.sparse.identity(data.shape[0])

    score = 0
    batches_ = np.unique(batches)
    N_batches = len(batches_)
    if N_batches < 2:
        raise ValueError("Should be more than one cluster for batch mixing")
    P = np.zeros(N_batches)
    for i in range(N_batches):
        P[i] = np.mean(batches == batches_[i])
    for t in range(n_pools):
        indices = np.random.choice(np.arange(data.shape[0]), size=n_samples_per_pool)
        score += np.mean([entropy(batches[kmatrix[indices].nonzero()[1]
        [kmatrix[indices].nonzero()[0] == i]])
                          for i in range(n_samples_per_pool)])
    Score = score / float(n_pools)
    return Score / float(np.log2(N_batches))

def evaluate_model(
        z_rna, 
        z_atac, 
        rna_cell_types, 
        atac_cell_types, 
        adata_combined
        ):
    """
    Evaluate the model using various metrics.
    """
    # ATAC cell types were predicted using the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(z_rna, rna_cell_types)
    pred_atac_cell_types = knn.predict(z_atac)


    # ARI
    ari = adjusted_rand_score(atac_cell_types, pred_atac_cell_types)
    # NMI
    nmi = normalized_mutual_info_score(atac_cell_types, pred_atac_cell_types)
    # F1-score
    f1 = f1_score(atac_cell_types, pred_atac_cell_types, average='micro')

    # Batch Entropy
    batch_entropy = batch_entropy_mixing_score(adata_combined.obsm['embedding'], adata_combined.obs['source'])

    # Silhouette Score
    silhouette = silhouette_score(adata_combined.obsm['embedding'], adata_combined.obs['cell_type'])


    # total metrics
    sum_evl = f1 + ari + nmi

    # 打印结果
    print(f'ARI: {ari}')
    print(f'NMI: {nmi}')
    print(f'F1: {f1}')
    print(f'SUM: {sum_evl}')
    print(f'Batch Entropy: {batch_entropy}')
    print(f'Silhouette: {silhouette}')

    return {
        'ARI': ari,
        'NMI': nmi,
        'F1': f1,
        'SUM': sum_evl,
        'Batch Entropy': batch_entropy,
        'Silhouette': silhouette
    }