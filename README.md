# GSSP


## GSSP: Eliminating Stragglers through Grouping Synchronous for Distributed Deep Learning in Heterogeneous Cluster

Codes are implemented with Python 3.6.2 and Pytorch 1.5.2.

### Introduction
we propose Grouping Stale Synchronous Parallel (GSSP) scheme, which groups edge nodes with similar performance together. Group servers coordinate intra-group workers using Stale Synchronous Parallel while they communicate with each other asynchronously to eliminate stragglers in the heterogeneous edge environment and refine the model weights. To save network bandwidth, we further propose Grouping Dynamic Tok-K Sparsification (GDTopK), which dynamically adjusts the upload ratio for each group to make communication volume differentiated and mitigate the inter-group iteration speed gap. Our work is on image classification and machine translation tasks. We run GSSP on a cluster with 16 workers and 4 PS nodes as shown in the figure.

![image](https://github.com/guizhiyi/GSSP/blob/main/imgs/fig1.001.jpeg)

### Code Setup
1. `pip install -r requirements.txt`

2. Run `python gssp.py` for simple GSSP. And run `python gdtopk.py` for GDTopK on top of GSSP.
   For image classification，the codes are in folder `cv`. For machine translation, the codes are in `nlp`. You can change your configurations in python codes.

