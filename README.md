# GSSP


## GSSP: Eliminating Stragglers through Grouping Synchronous for Distributed Deep Learning in Heterogeneous Cluster

### Introduction
we propose Grouping Stale Synchronous Parallel (GSSP) scheme, which groups edge nodes with similar performance together. Group servers coordinate intra-group workers using Stale Synchronous Parallel while they communicate with each other asynchronously to eliminate stragglers in the heterogeneous edge environment and refine the model weights. To save network bandwidth, we further propose Grouping Dynamic Tok-K Sparsification (GDTopK), which dynamically adjusts the upload ratio for each group to make communication volume differentiated and mitigate the inter-group iteration speed gap. We evaluate GSSP on a cluster with 16 workers and 4 PS nodes as shown in Fig.1.

![image](https://github.com/guizhiyi/GSSP/blob/main/imgs/fig1.001.jpeg)

### Code Setup
1. `pip install -r requirements.txt`

2. Run `python gssp.py`
   You can change your configurations in python codes.

