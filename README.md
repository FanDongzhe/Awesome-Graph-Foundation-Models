# Awesome-Graph-Foundation-Models [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A list of existing efforts on Graph Foundation Models (Graph FMs) based on our survey paper.

## Contents

- [Awesome-Graph-Foundation-Models ](#awesome-graph-foundation-models-)
  - [Contents](#contents)
  - [What is Graph Foundation Model ?](#what-is-graph-foundation-model-)
  - [Graph Embedding Foundation Models (GraphEFMs)](#graph-embedding-foundation-models-graphefms)
    - [GNN-based Models](#gnn-based-models)
    - [Transformer-based Models](#transformer-based-models)
    - [GNN + Transformer Models](#gnn--transformer-models)
  - [Graph Predictive Foundation Models](#graph-predictive-foundation-models)
    - [GNN-based Models](#gnn-based-models-1)
    - [LLM-based Models](#llm-based-models)
  - [Graph Generative Foundation Models](#graph-generative-foundation-models)
    - [One-Time Graph Generation](#one-time-graph-generation)
    - [Graph-related Data Generation](#graph-related-data-generation)
  - [Citation](#citation)





## What is Graph Foundation Model ?
Graph Foundation Models (Graph FMs) are AI models trained on vast datasets, often using self-supervision, and contain tens of billions of parameters, making them versatile across a wide range of tasks.They exhibit two core properties: **Data Generalization** and **Task Generalization**. In this survey we categorize existing efforts in Graph FMs based on their learning objectives: **Embedding Foundation Model (GraphEFM)**, **Predictive Foundation Model (GraphPFM)** and **Generative Foundation Model (GraphGFM)**.

<p align="center">
    <img src="./Figures/GFM.jpg" width="90%" style="align:center;"/>
</p>


## Graph Embedding Foundation Models (GraphEFMs)

GraphEmbedding Foundation Model focuses on learning representations of graph structures and nodes. These embeddings capture essential patterns and relationships within the graph, enabling efficient use in downstream tasks such as clustering, classification, or link prediction. 

<p align="center">
    <img src="./Figures/GEFM.jpg" height="480" width="380" style="align:center;"/>
</p>

### GNN-based Models
- (**NeurIPS'16**) Variational Graph Auto-Encoders [[paper](https://arxiv.org/abs/1611.07308)][[code](https://github.com/tkipf/gae)]
- (**NeurIPS'20**) Graph contrastive learning with augmentations [[paper](https://arxiv.org/abs/2010.13902)][[code](https://github.com/Shen-Lab/GraphCL)]
- (**ICLR'21**) Large-Scale Representation Learning on Graphs via Bootstrapping [[paper](https://arxiv.org/abs/2102.06514)][[code](https://github.com/Namkyeong/BGRL_Pytorch)]
- (**WWW'21**) Graph contrastive learning with adaptive augmentation [[paper](https://arxiv.org/abs/2010.14945)][[code](https://github.com/CRIPAC-DIG/GCA)]
- (**AAAI'22**) Augmentation-free self-supervised learning on graphs [[paper](https://arxiv.org/abs/2112.02472)][[code](https://github.com/Namkyeong/AFGRL)]
- (**KBS'22**) Graph Barlow Twins: A self-supervised representation learning framework for graphs [[paper](https://arxiv.org/abs/2106.02466)][[code](https://github.com/pbielak/graph-barlow-twins)]
- (**KDD'22**) GraphMAE: Self-Supervised Masked Graph Autoencoders [[paper](https://arxiv.org/abs/2205.10803)][[code](https://github.com/THUDM/GraphMAE)]
- (**WSDM'23**) S2GAE: Self-Supervised Graph Autoencoders are Generalizable Learners with Graph Masking [[paper](https://dl.acm.org/doi/10.1145/3539597.3570404)][[code](https://github.com/qiaoyu-tan/S2GAE)]
- (**WWW'23**) GraphMAE2: A Decoding-Enhanced Masked Self-Supervised Graph Learner [[paper](https://arxiv.org/abs/2304.04779)][[code](https://github.com/THUDM/GraphMAE2)]
- (**CIKM'23**) GiGaMAE: Generalizable Graph Masked Autoencoder via Collaborative Latent Space Reconstruction [[paper](https://arxiv.org/abs/2308.09663)][[code](https://github.com/sycny/GiGaMAE)]
- (**NeurIPS'23**) PRODIGY: Enabling In-context Learning Over Graphs [[paper](https://arxiv.org/abs/2305.12600)][[code](https://github.com/snap-stanford/prodigy)]
- (**KDD'24**) Gaugllm: Improving graph contrastive learning for text-attributed graphs with large language models [[paper](https://arxiv.org/abs/2406.11945)][[code](https://github.com/NYUSHCS/GAugLLM)]
- (**arXiv 2024.08**) AnyGraph: Graph Foundation Model in the Wild [[paper](https://arxiv.org/pdf/2408.10700)][[code](https://github.com/HKUDS/AnyGraph)]
### Transformer-based Models
### GNN + Transformer Models

## Graph Predictive Foundation Models
The Graph Predictive Foundation Model is aimed at developing models that can directly predict outcomes for various graph-related tasks related to forecasting, classification, or inference based on existing data. By capturing intricate relationships and dependencies within graphs, these models offer strong predictive capabilities across different domains.
<p align="center">
    <img src="./Figures/GPFM.jpg" height="380" width="450" style="align:center;"/>
</p>

### GNN-based Models
### LLM-based Models

## Graph Generative Foundation Models
Graph Generation Foundation Model emphasizes the ability to generate new graph structures and nodes based on learned patterns from existing graphs.
<p align="center">
    <img src="./Figures/GGFM.jpg" height="460" width="390" style="align:center;"/>
</p>

### One-Time Graph Generation
### Graph-related Data Generation

## Citation
