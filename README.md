# EigenMol

## Introduction

This project is aimed at analysis of efficacy and meaningfulness of molecular embedding algorithms followed by uncovering interpretation of eigenvectors.
Inspired by NLP algorithms. For instance, [Mol2vec](https://pubs.acs.org/doi/10.1021/acs.jcim.7b00616) learns vector representations of molecular substructures that point in similar directions for chemically related substructures. 
Our goal is to test the applicability of PCA for molecular embeddings - whether this procedure provide meaningful insigts for cheminformatics application.
In our research, we used several datasets:

1. Small molecules for drug design https://www.kaggle.com/datasets/art3mis/chembl22/data
2. Peptides for drug design https://www.sciencedirect.com/science/article/pii/S235234091930962X

All initial molecule representations are in [SMILES](https://en.wikipedia.org/wiki/Simplified_Molecular_Input_Line_Entry_System) format

## Pipeline of EigenMol

1. Get embeddings of molecules dataset (via Mol2vec or Molformer) 
2. Construct covariance matrix, calculate PCA (get eigenvectors)
3. Determine the closest molecule representations to eigenvectors with $l_2$ norms (KDTree was used)
4. Compare the results with processing Morgan Fingerprint

## Requirements

Mol2vec from [here](https://github.com/mmkuznecov/mol2vec)

Download it via:

```shell
pip install git+https://github.com/mmkuznecov/mol2vec
```

## Code usage

### Peptides example

1. Transform dataset into parquet format 
```shell
py process_peptides.py
```
2. Get molecular embeddings and store them
```shell
py m2v_emb_generation_peptides.py
```
3. Get PCA results and molecule representations: check `StorageUsage.ipynb`

## Interpretation

As a result, we get eigenvectors assigned to molecular embeddings - "eigenmolecules". What is the meaning of these eigenmolecules? We suggest, from a chemical perspective and understanding of LLM technique, that the eigenmolecule corresponding to $\approx 96\%$ of explained variance is the molecule with largest amount of common substructures in the dataset. In other words, all other molecules have close substructures to this eigenmolecule. This gives possibility of understanding the dataset nature and a brief screening of structure-related properties 

## Distance metric comparison

We used [RDkit](https://github.com/rdkit/rdkit) tools for implementation of classical approach to vector representation of molecule structures - Morgan Fingerprints (as binary strings) and Tanimoto similarity.

## Embeddings quality assessment

There is a nice [paper](https://arxiv.org/abs/2305.16562) about different metrics for embedding quality, following were implemented:
1. RankME. Given a matrix $\mathbf{M} \in \mathbb{R}^{n_1 \times n_2}$ with SVD $\mathbf{M} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^{*}$:
```math
\text{RankMe}(\mathbf{M}) = -\sum\limits_{i}p_i \log p_i, \quad p_i = \dfrac{\sigma_i}{\Vert \Sigma \Vert_1}
```
3. NESUM. Given a matrix $\mathbf{M} \in \mathbb{R}^{n_1 \times n_2}$ with its covariance matrix eigendecomposition $\mathbf{C} = \mathbf{U\Lambda U}^{\top}$:
```math
\text{NESum}(\mathbf{M}) = \sum\limits_{i}\dfrac{\lambda_i}{\lambda}
```
4. $\mu_0$-incoherence. For matrix $\mathbf{M} \in \mathbb{R}^{n_1 \times n_2}$ with rank- $r$ and SVD $\mathbf{M} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^{*}$ its said to satisfy $\textit{standard incoherence}$ condition with parameter $\mu_0$, if:
```math
 \max_{1 \leq i \leq n_1} \Vert \mathbf{U}^{\top}e_i \Vert_2 \leq \sqrt{\dfrac{\mu_0r}{n_1}}, \quad \max_{1 \leq i \leq n_2} \Vert \mathbf{V}^{\top}e_j \Vert_2 \leq \sqrt{\dfrac{\mu_0r}{n_2}}
 ```

## References
Jaeger, Sabrina, Simone Fulle, and Samo Turk. "Mol2vec: unsupervised machine learning approach with chemical intuition." Journal of chemical information and modeling 58.1 (2018): 27-35. https://pubs.acs.org/doi/10.1021/acs.jcim.7b00616

A. Tsitsulin, M. Munkhoeva, B. Perozzi. "Unsupervised Embedding Quality Evaluation". (2023) https://arxiv.org/abs/2305.16562
