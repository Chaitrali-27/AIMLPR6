# Practical Assignment 6 – Clustering Algorithms

**TY B.Tech. – Artificial Intelligence and Machine Learning**

Unsupervised clustering of the real-world **UCI Wine Recognition** dataset (178 wines, 13 chemical measurements)
using **K-Means, Hierarchical (Ward), DBSCAN and Gaussian Mixture Models**. Cluster quality is judged with
**Silhouette score, Davies-Bouldin index, Calinski-Harabasz index** (internal) and **ARI / NMI / homogeneity /
completeness** (external, using the known cultivar only for validation – never for training), plus visual inspection.

## Project structure

```
clustering-assignment/
├── README.md                    <- this file
├── requirements.txt             <- dependencies
├── src/
│   └── clustering_analysis.py   <- complete, commented source code
├── data/
│   ├── wine_features.csv        <- dataset (13 features)
│   └── wine_true_labels.csv     <- cultivar labels (validation only)
├── outputs/
│   ├── figures/                 <- 11 result plots (PNG)
│   └── results/                 <- metric tables (CSV), run_log.json, console_output.txt
└── docs/
    └── AIML_Practical6_Clustering_Report.docx   <- filled assignment template
```

## How to run

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/clustering_analysis.py
```
Runtime is under one minute. Everything is reproducible (`random_state = 42`). The dataset ships with
scikit-learn, so no download is required.

## Pipeline

1. **EDA** – shape, missing values, duplicates, correlation heat-map, distributions
2. **Preprocessing** – `StandardScaler` (essential for distance-based methods)
3. **Choosing k** – elbow, silhouette, Davies-Bouldin, Calinski-Harabasz for k = 2…10; GMM BIC/AIC
4. **Algorithms** – K-Means, Agglomerative-Ward (+ dendrogram), DBSCAN (grid search + k-distance plot), GMM
5. **Evaluation** – internal + external metrics, silhouette plot, PCA visualisation, cluster profiling
6. **Improvement / robustness** – effect of scaling, PCA dimensionality, stability over seeds and 80 % sub-samples

## Key results

| Algorithm | Clusters | Noise | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ | ARI ↑ | NMI ↑ |
|---|---|---|---|---|---|---|---|
| **K-Means (k=3)** | 3 | 0 | 0.285 | **1.389** | **70.94** | **0.897** | **0.876** |
| Hierarchical Ward (k=3) | 3 | 0 | 0.277 | 1.419 | 67.65 | 0.790 | 0.786 |
| DBSCAN (eps=2.2, min_samples=4) | 2 | 48 | 0.340* | 1.160* | 70.76* | 0.378 | 0.466 |
| GMM (k=3) | 3 | 0 | 0.284 | 1.394 | 70.69 | 0.880 | 0.861 |

\*DBSCAN metrics exclude the 48 noise points (27 %), so they are not directly comparable with the other rows.

* All four internal criteria (elbow, silhouette, DBI, CH) select **k = 3**.
* **K-Means** recovers the three wine cultivars with only 6 of 178 wines misplaced (96.6 %); it is also very stable
  (mean ARI 0.97 across 20 seeds and 20 random 80 % sub-samples).
* **Scaling matters:** without it, K-Means is dominated by `proline` and ARI falls from 0.897 to 0.371 – even though the
  raw-data silhouette looks higher (0.571). Internal metrics alone can be misleading.

## Figures

| | |
|---|---|
| `01_correlation_heatmap.png` | feature correlations |
| `02_feature_distributions.png` | histograms before scaling |
| `03_optimal_k_selection.png` | elbow / silhouette / DBI / CH vs k |
| `04_silhouette_kmeans.png` | per-sample silhouette plot |
| `05_dendrogram.png` | Ward dendrogram |
| `06_dbscan_kdistance.png` | k-distance graph for eps |
| `07_gmm_bic_aic.png` | GMM model selection |
| `08_all_models_pca.png` | all algorithms in 2-D PCA space |
| `09_kmeans_profile_heatmap.png` | cluster profiles |
| `10_metric_comparison.png` | metric bar charts |
| `11_top_feature_pairplot.png` | most discriminating features |

## Ethical note

The Wine data are chemical measurements with no personal information. When the same pipeline is applied to people
(e.g. customer segmentation) privacy, consent and the risk of discriminatory groupings must be assessed, and cluster
labels should never be treated as ground truth about an individual.

## Dataset citation

Forina, M. et al. (1991). *PARVUS – An Extendible Package for Data Exploration, Classification and Correlation.*
UCI Machine Learning Repository – Wine dataset (bundled with `scikit-learn`).
