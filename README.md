# scMGCL
Single-cell multi-omics data integration is essential for understanding cellular states and disease mechanisms, yet integrating heterogeneous data modalities remains a challenge. We present
scMGCL, a graph contrastive learning framework for robust integration of single-cell ATAC-seq and RNA-seq data. Our approach leverages self-supervised learning on cell-cell similarity
graphs, where each modality’s graph structure serves as an augmentation for the other. This cross-modality contrastive paradigm enables the learning of biologically meaningful, shared representations while preserving modality-specific features. Benchmarking against state-of-the-art
methods demonstrates that scMGCL outperforms others in cell-type clustering, label transfer accuracy, and preservation of marker gene correlations. Additionally, scMGCL significantly improves computational efficiency, reducing runtime and memory usage.
The method’s effectiveness is further validated through extensive analyses of cell-type similarity and functional consistency, providing a powerful tool for multi-omics data exploration.

![](./Method-fig.pdf)

# Author
Zhenglong Cheng, Risheng Lu, Shixiong Zhang

School of Computer Science and Technology

Xidian University

Xi’an, Shaanxi 710071, China

# Contact
24031212242@stu.xidian.edu.cn
