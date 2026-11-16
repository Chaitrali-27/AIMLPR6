"""
Practical Assignment 6: Clustering Algorithms
TY B.Tech. - Artificial Intelligence and Machine Learning

Goal
----
Apply several clustering algorithms to a real-world dataset (UCI Wine
Recognition dataset) to discover meaningful groups, and evaluate the quality
of every result with internal metrics (Silhouette, Davies-Bouldin,
Calinski-Harabasz), external metrics (ARI, NMI) and visual inspection.

Algorithms compared
-------------------
1. K-Means
2. Agglomerative (Hierarchical) Clustering - Ward linkage
3. DBSCAN (density based)
4. Gaussian Mixture Model (GMM)

Run
---
    python src/clustering_analysis.py
All figures, tables and logs are written to the ``outputs/`` folder.
"""

import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")  # no GUI needed; figures are saved to disk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, homogeneity_score,
                             completeness_score, normalized_mutual_info_score,
                             silhouette_samples, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
RANDOM_STATE = 42
K_RANGE = range(2, 11)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
FIG_DIR = os.path.join(ROOT, "outputs", "figures")
RES_DIR = os.path.join(ROOT, "outputs", "results")

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = sns.color_palette("Set2", 10)


# --------------------------------------------------------------------------
# 1. Data loading and preprocessing
# --------------------------------------------------------------------------
def load_data():
    """Load the Wine dataset. Labels are kept ONLY for external validation;
    they are never given to any clustering algorithm."""
    wine = load_wine(as_frame=True)
    df = wine.frame.copy()
    y_true = df.pop("target").values
    df.to_csv(os.path.join(DATA_DIR, "wine_features.csv"), index=False)
    pd.Series(y_true, name="cultivar").to_csv(
        os.path.join(DATA_DIR, "wine_true_labels.csv"), index=False)
    return df, y_true, list(wine.target_names)


def data_overview(df, y_true):
    """Basic exploratory data analysis (EDA) and quality checks."""
    info = {
        "n_samples": int(df.shape[0]),
        "n_features": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "true_class_counts": {int(k): int(v) for k, v in
                              zip(*np.unique(y_true, return_counts=True))},
    }
    df.describe().T.round(3).to_csv(os.path.join(RES_DIR, "dataset_summary.csv"))

    # Correlation heat-map
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), cmap="coolwarm", center=0, annot=False,
                square=True, cbar_kws={"shrink": .8})
    plt.title("Feature Correlation Heatmap (Wine Dataset)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "01_correlation_heatmap.png"), dpi=150)
    plt.close()

    # Feature distributions
    df.hist(bins=20, figsize=(14, 10), color=PALETTE[0], edgecolor="white")
    plt.suptitle("Feature Distributions Before Scaling", y=1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "02_feature_distributions.png"), dpi=150)
    plt.close()
    return info


def preprocess(df):
    """Standardise features (mean 0, std 1). Required because K-Means, DBSCAN
    and hierarchical clustering are distance based: 'proline' (values ~1000)
    would otherwise dominate 'hue' (values ~1)."""
    scaler = StandardScaler()
    X = scaler.fit_transform(df)
    return X, scaler


def pca_projection(X, n_components=2):
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X2 = pca.fit_transform(X)
    return X2, pca


# --------------------------------------------------------------------------
# 2. Evaluation helpers
# --------------------------------------------------------------------------
def internal_metrics(X, labels):
    """Internal validity metrics (no ground truth needed). Noise points
    (label -1, DBSCAN only) are excluded from the calculation."""
    mask = labels != -1
    n_clusters = len(set(labels[mask]))
    if n_clusters < 2:
        return {"silhouette": np.nan, "davies_bouldin": np.nan,
                "calinski_harabasz": np.nan}
    return {
        "silhouette": silhouette_score(X[mask], labels[mask]),
        "davies_bouldin": davies_bouldin_score(X[mask], labels[mask]),
        "calinski_harabasz": calinski_harabasz_score(X[mask], labels[mask]),
    }


def external_metrics(y_true, labels):
    """External validity metrics (compare against known cultivar labels)."""
    return {
        "ARI": adjusted_rand_score(y_true, labels),
        "NMI": normalized_mutual_info_score(y_true, labels),
        "homogeneity": homogeneity_score(y_true, labels),
        "completeness": completeness_score(y_true, labels),
    }


# --------------------------------------------------------------------------
# 3. Choosing the number of clusters (K-Means)
# --------------------------------------------------------------------------
def find_optimal_k(X):
    """Elbow (inertia), Silhouette, Davies-Bouldin and Calinski-Harabasz for
    k = 2..10 to select k using several independent criteria."""
    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(X)
        m = internal_metrics(X, km.labels_)
        rows.append({"k": k, "inertia": km.inertia_, **m})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RES_DIR, "kmeans_k_selection.csv"), index=False)

    fig, ax = plt.subplots(1, 4, figsize=(20, 4))
    specs = [("inertia", "Elbow Method (Inertia / WCSS)", "Inertia"),
             ("silhouette", "Silhouette Score (higher is better)", "Silhouette"),
             ("davies_bouldin", "Davies-Bouldin Index (lower is better)", "DBI"),
             ("calinski_harabasz", "Calinski-Harabasz (higher is better)", "CH")]
    for a, (col, title, yl) in zip(ax, specs):
        a.plot(res["k"], res[col], "o-", color=PALETTE[1], lw=2)
        a.set_title(title, fontsize=10)
        a.set_xlabel("Number of clusters k")
        a.set_ylabel(yl)
        a.axvline(3, ls="--", color="grey", alpha=.6)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "03_optimal_k_selection.png"), dpi=150)
    plt.close()

    best_k_sil = int(res.loc[res["silhouette"].idxmax(), "k"])
    best_k_dbi = int(res.loc[res["davies_bouldin"].idxmin(), "k"])
    return res, best_k_sil, best_k_dbi


def silhouette_plot(X, labels, title, fname):
    """Per-sample silhouette diagram for visual inspection of cluster quality."""
    s = silhouette_samples(X, labels)
    avg = s.mean()
    fig, ax = plt.subplots(figsize=(7, 5))
    y_lower = 10
    for i, c in enumerate(sorted(set(labels))):
        vals = np.sort(s[labels == c])
        y_upper = y_lower + len(vals)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals,
                         color=PALETTE[i], alpha=.8)
        ax.text(-0.05, y_lower + 0.5 * len(vals), f"C{c}")
        y_lower = y_upper + 10
    ax.axvline(avg, color="red", ls="--", label=f"mean = {avg:.3f}")
    ax.set_title(title)
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Samples (grouped by cluster)")
    ax.set_yticks([])
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=150)
    plt.close()


# --------------------------------------------------------------------------
# 4. Individual algorithms
# --------------------------------------------------------------------------
def run_kmeans(X, k):
    return KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(X)


def run_hierarchical(X, k):
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X)


def plot_dendrogram(X):
    Z = linkage(X, method="ward")
    plt.figure(figsize=(12, 5))
    dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90,
               leaf_font_size=9, show_contracted=True,
               color_threshold=0.7 * max(Z[:, 2]))
    plt.title("Hierarchical Clustering Dendrogram (Ward linkage, last 30 merges)")
    plt.xlabel("Cluster size / sample index")
    plt.ylabel("Ward distance")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "05_dendrogram.png"), dpi=150)
    plt.close()


def tune_dbscan(X, min_samples_list=(3, 4, 5, 6), eps_list=None):
    """Grid search over eps / min_samples. A configuration is scored by its
    silhouette (noise excluded) but is only accepted if it finds 2-6 clusters
    and labels fewer than 30 % of points as noise."""
    if eps_list is None:
        eps_list = np.round(np.arange(1.0, 4.01, 0.1), 2)
    rows = []
    for ms in min_samples_list:
        for eps in eps_list:
            lab = DBSCAN(eps=eps, min_samples=ms).fit(X).labels_
            n_cl = len(set(lab)) - (1 if -1 in lab else 0)
            noise = float((lab == -1).mean())
            sil = (silhouette_score(X[lab != -1], lab[lab != -1])
                   if n_cl >= 2 else np.nan)
            rows.append({"eps": eps, "min_samples": ms, "n_clusters": n_cl,
                         "noise_pct": round(noise * 100, 1), "silhouette": sil})
    grid = pd.DataFrame(rows)
    grid.to_csv(os.path.join(RES_DIR, "dbscan_grid_search.csv"), index=False)
    valid = grid[(grid.n_clusters.between(2, 6)) & (grid.noise_pct < 30)]
    if valid.empty:
        valid = grid.dropna(subset=["silhouette"])
    best = valid.loc[valid["silhouette"].idxmax()]
    return grid, float(best["eps"]), int(best["min_samples"])


def plot_kdistance(X, min_samples, eps_chosen):
    nn = NearestNeighbors(n_neighbors=min_samples).fit(X)
    d, _ = nn.kneighbors(X)
    kd = np.sort(d[:, -1])
    plt.figure(figsize=(7, 4.5))
    plt.plot(kd, color=PALETTE[2], lw=2)
    plt.axhline(eps_chosen, color="red", ls="--", label=f"chosen eps = {eps_chosen}")
    plt.title(f"k-distance Graph (k = min_samples = {min_samples})")
    plt.xlabel("Points sorted by distance")
    plt.ylabel(f"Distance to {min_samples}-th nearest neighbour")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "06_dbscan_kdistance.png"), dpi=150)
    plt.close()


def tune_gmm(X):
    """Select number of GMM components with BIC / AIC."""
    rows = []
    for k in K_RANGE:
        g = GaussianMixture(n_components=k, covariance_type="full",
                            n_init=5, random_state=RANDOM_STATE).fit(X)
        rows.append({"k": k, "BIC": g.bic(X), "AIC": g.aic(X)})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RES_DIR, "gmm_bic_aic.csv"), index=False)
    plt.figure(figsize=(7, 4.5))
    plt.plot(res["k"], res["BIC"], "o-", label="BIC")
    plt.plot(res["k"], res["AIC"], "s-", label="AIC")
    plt.title("GMM Model Selection (lower is better)")
    plt.xlabel("Number of components")
    plt.ylabel("Information criterion")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "07_gmm_bic_aic.png"), dpi=150)
    plt.close()
    return res, int(res.loc[res["BIC"].idxmin(), "k"])


# --------------------------------------------------------------------------
# 5. Visualisation of clustering results
# --------------------------------------------------------------------------
def scatter_2d(X2, labels, title, ax, pca):
    for i, c in enumerate(sorted(set(labels))):
        pts = X2[labels == c]
        name = "Noise" if c == -1 else f"Cluster {c}"
        colour = "black" if c == -1 else PALETTE[i % 10]
        marker = "x" if c == -1 else "o"
        ax.scatter(pts[:, 0], pts[:, 1], s=35, c=[colour], marker=marker,
                   alpha=.85, edgecolor="white" if c != -1 else None,
                   linewidth=.4, label=name)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var.)")
    ax.legend(fontsize=8)


def plot_all_models(X2, pca, y_true, results):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    scatter_2d(X2, y_true, "Ground truth (cultivar) - reference only",
               axes[0, 0], pca)
    for ax, (name, info) in zip(axes.flatten()[1:], results.items()):
        scatter_2d(X2, info["labels"], f"{name}\nSilhouette = "
                   f"{info['internal']['silhouette']:.3f}", ax, pca)
    for ax in axes.flatten()[len(results) + 1:]:
        ax.axis("off")  # hide unused sub-plots
    plt.suptitle("Clustering Results in PCA Space", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "08_all_models_pca.png"), dpi=150)
    plt.close()


def cluster_profile(df, labels, name):
    """Mean feature values per cluster -> interpretation of each group."""
    prof = df.copy()
    prof["cluster"] = labels
    prof = prof[prof.cluster != -1].groupby("cluster").mean().round(2)
    prof["size"] = prof.index.map(pd.Series(labels[labels != -1]).value_counts())
    prof.to_csv(os.path.join(RES_DIR, f"cluster_profile_{name}.csv"))
    return prof


def profile_heatmap(df, labels):
    z = (df - df.mean()) / df.std()
    z["cluster"] = labels
    prof = z.groupby("cluster").mean()
    plt.figure(figsize=(12, 4))
    sns.heatmap(prof, cmap="RdBu_r", center=0, annot=True, fmt=".1f",
                cbar_kws={"label": "Standardised mean"})
    plt.title("K-Means Cluster Profiles (z-scored feature means)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "09_kmeans_profile_heatmap.png"), dpi=150)
    plt.close()


def confusion_table(y_true, labels, target_names):
    ct = pd.crosstab(pd.Series(y_true, name="True cultivar"),
                     pd.Series(labels, name="Cluster"))
    ct.index = [target_names[i] for i in ct.index]
    return ct


def plot_metric_comparison(summary):
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
    cols = [("silhouette", "Silhouette (↑)"), ("davies_bouldin", "Davies-Bouldin (↓)"),
            ("calinski_harabasz", "Calinski-Harabasz (↑)"), ("ARI", "Adjusted Rand Index (↑)")]
    for a, (c, t) in zip(ax, cols):
        a.bar(summary.index, summary[c], color=PALETTE[:len(summary)])
        a.set_title(t)
        a.tick_params(axis="x", rotation=25)
    plt.suptitle("Algorithm Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "10_metric_comparison.png"), dpi=150)
    plt.close()


# --------------------------------------------------------------------------
# 6. Robustness / improvement experiments
# --------------------------------------------------------------------------
def scaling_effect(df, y_true):
    """Shows why scaling is essential: K-Means with and without it."""
    raw = KMeans(3, n_init=20, random_state=RANDOM_STATE).fit(df.values).labels_
    scl = KMeans(3, n_init=20, random_state=RANDOM_STATE).fit(
        StandardScaler().fit_transform(df)).labels_
    out = {}
    for tag, lab, Xuse in [("Without scaling", raw, df.values),
                           ("With StandardScaler", scl,
                            StandardScaler().fit_transform(df))]:
        out[tag] = {**internal_metrics(Xuse, lab), **external_metrics(y_true, lab)}
    res = pd.DataFrame(out).T
    res.to_csv(os.path.join(RES_DIR, "scaling_effect.csv"))
    return res


def pca_improvement(X, y_true, n_comp_list=(2, 3, 5, 8, 13)):
    """Improvement experiment: cluster in a PCA-reduced space."""
    rows = []
    for n in n_comp_list:
        Xp = X if n == X.shape[1] else PCA(n, random_state=RANDOM_STATE).fit_transform(X)
        lab = KMeans(3, n_init=20, random_state=RANDOM_STATE).fit(Xp).labels_
        rows.append({"n_components": n, **internal_metrics(Xp, lab),
                     **external_metrics(y_true, lab)})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RES_DIR, "pca_improvement.csv"), index=False)
    return res


def stability_test(X, n_runs=20):
    """Stability of K-Means over different random seeds & bootstrap samples."""
    ref = KMeans(3, n_init=20, random_state=RANDOM_STATE).fit(X).labels_
    seed_ari = []
    for s in range(n_runs):
        lab = KMeans(3, n_init=1, random_state=s).fit(X).labels_
        seed_ari.append(adjusted_rand_score(ref, lab))
    rng = np.random.RandomState(RANDOM_STATE)
    boot_ari = []
    for _ in range(n_runs):
        idx = rng.choice(len(X), size=int(0.8 * len(X)), replace=False)
        lab = KMeans(3, n_init=10, random_state=RANDOM_STATE).fit(X[idx]).labels_
        boot_ari.append(adjusted_rand_score(ref[idx], lab))
    return {"seed_ARI_mean": float(np.mean(seed_ari)),
            "seed_ARI_min": float(np.min(seed_ari)),
            "subsample80_ARI_mean": float(np.mean(boot_ari)),
            "subsample80_ARI_min": float(np.min(boot_ari))}


def best_feature_pairs(df, labels):
    """Pair-plot of the most discriminating features (by between-cluster F)."""
    from sklearn.feature_selection import f_classif
    F, _ = f_classif(df, labels)
    top = df.columns[np.argsort(F)[::-1][:4]].tolist()
    d = df[top].copy()
    d["cluster"] = labels.astype(str)
    g = sns.pairplot(d, hue="cluster", palette="Set2", corner=True,
                     plot_kws=dict(s=25, alpha=.8))
    g.figure.suptitle("Top-4 Discriminating Features by K-Means Cluster", y=1.01)
    g.savefig(os.path.join(FIG_DIR, "11_top_feature_pairplot.png"), dpi=130)
    plt.close("all")
    return top


# --------------------------------------------------------------------------
# 7. Main pipeline
# --------------------------------------------------------------------------
def main():
    for d in (DATA_DIR, FIG_DIR, RES_DIR):
        os.makedirs(d, exist_ok=True)
    log = {}

    # ---- data --------------------------------------------------------
    df, y_true, target_names = load_data()
    log["dataset"] = data_overview(df, y_true)
    X, scaler = preprocess(df)
    X2, pca2 = pca_projection(X)
    log["scaled_mean_abs_max"] = float(np.abs(X.mean(axis=0)).max())
    log["scaled_std_range"] = [float(X.std(axis=0).min()), float(X.std(axis=0).max())]
    full_pca = PCA(random_state=RANDOM_STATE).fit(X)
    log["pca_var_first2"] = float(full_pca.explained_variance_ratio_[:2].sum())
    log["pca_var_first3"] = float(full_pca.explained_variance_ratio_[:3].sum())
    print("Dataset:", log["dataset"])

    # ---- choose k ----------------------------------------------------
    k_table, k_sil, k_dbi = find_optimal_k(X)
    print("\nK-selection table:\n", k_table.round(3).to_string(index=False))
    K = 3  # justified in report: elbow + silhouette + DBI + domain knowledge
    log["k_by_silhouette"], log["k_by_dbi"], log["k_chosen"] = k_sil, k_dbi, K

    # ---- algorithms --------------------------------------------------
    km = run_kmeans(X, K)
    hc = run_hierarchical(X, K)
    plot_dendrogram(X)
    grid, eps, ms = tune_dbscan(X)
    plot_kdistance(X, ms, eps)
    db = DBSCAN(eps=eps, min_samples=ms).fit(X)
    gmm_table, k_gmm = tune_gmm(X)
    gmm = GaussianMixture(K, covariance_type="full", n_init=10,
                          random_state=RANDOM_STATE).fit(X)
    log["dbscan_params"] = {"eps": eps, "min_samples": ms}
    log["gmm_k_by_bic"] = k_gmm

    all_labels = {
        "K-Means (k=3)": km.labels_,
        "Hierarchical Ward (k=3)": hc.labels_,
        f"DBSCAN (eps={eps}, ms={ms})": db.labels_,
        "GMM (k=3)": gmm.predict(X),
    }
    results = {}
    for name, lab in all_labels.items():
        results[name] = {"labels": lab,
                         "internal": internal_metrics(X, lab),
                         "external": external_metrics(y_true, lab)}

    # ---- summary table -----------------------------------------------
    rows = []
    for name, info in results.items():
        lab = info["labels"]
        rows.append({"algorithm": name,
                     "n_clusters": len(set(lab)) - (1 if -1 in lab else 0),
                     "noise_points": int((lab == -1).sum()),
                     **info["internal"], **info["external"]})
    summary = pd.DataFrame(rows).set_index("algorithm")
    summary.round(4).to_csv(os.path.join(RES_DIR, "algorithm_comparison.csv"))
    print("\nAlgorithm comparison:\n", summary.round(3).to_string())

    # ---- plots -------------------------------------------------------
    plot_all_models(X2, pca2, y_true, results)
    plot_metric_comparison(summary)
    silhouette_plot(X, km.labels_, "Silhouette Plot - K-Means (k=3)",
                    "04_silhouette_kmeans.png")
    profile_heatmap(df, km.labels_)
    top_feats = best_feature_pairs(df, km.labels_)
    log["top_features"] = top_feats

    # ---- profiles & confusion tables ----------------------------------
    prof = cluster_profile(df, km.labels_, "kmeans")
    print("\nK-Means cluster profile (raw means):\n", prof.to_string())
    ct = confusion_table(y_true, km.labels_, target_names)
    ct.to_csv(os.path.join(RES_DIR, "kmeans_vs_true_labels.csv"))
    print("\nK-Means vs true cultivars:\n", ct.to_string())
    ct_db = confusion_table(y_true, db.labels_, target_names)
    ct_db.to_csv(os.path.join(RES_DIR, "dbscan_vs_true_labels.csv"))

    # ---- improvement / robustness experiments --------------------------
    sc = scaling_effect(df, y_true)
    print("\nEffect of scaling:\n", sc.round(3).to_string())
    pc = pca_improvement(X, y_true)
    print("\nEffect of PCA dimensionality:\n", pc.round(3).to_string(index=False))
    log["stability"] = stability_test(X)
    print("\nStability:", log["stability"])

    # ---- save run log --------------------------------------------------
    with open(os.path.join(RES_DIR, "run_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print("\nDone. Figures ->", FIG_DIR, "| Tables ->", RES_DIR)


if __name__ == "__main__":
    main()
