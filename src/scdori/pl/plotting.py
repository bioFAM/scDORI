import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import seaborn as sns


def plot_downstream_targets(rna_anndata, gene_list, score_name="target_score", layer="log"):
    """
    Visualize the average expression of given genes on a UMAP embedding.

    Uses `scanpy.tl.score_genes` to compute a gene score, then plots using `scanpy.pl.umap`.

    Parameters
    ----------
    rna_anndata : anndata.AnnData
        The AnnData object containing RNA data with `.obsm["X_umap"]`.
    gene_list : list of str
        A list of gene names to score.
    score_name : str, optional
        Name of the resulting gene score in `rna_anndata.obs`. Default is "target_score".
    layer : str, optional
        Which layer to use if needed in `score_genes`. Default is "log".

    Returns
    -------
    None
        Plots the UMAP colored by the computed gene score.
    """
    sc.tl.score_genes(rna_anndata, gene_list, score_name=score_name)
    sc.pl.umap(rna_anndata, color=[score_name], layer=layer)


def plot_topic_activation_heatmap(rna_anndata, groupby_key="celltype", aggregation="median", plot=True):
    """
    Compute aggregated scDORI latent topic activation across groups, then plot a clustermap.

    Parameters
    ----------
    rna_anndata : anndata.AnnData
        An AnnData object containing scDORI latent factors in `obsm["X_scdori"]`.
    groupby_key : str, optional
        Column in `rna_anndata.obs` by which to group cells. Default is "celltype".
    aggregation : str, optional
        Either "median" or "mean" for aggregating factor values per group. Default is "median".
    plot : bool, optional
        Draw the default heatmap. Set False to obtain the table for a custom plot.

    Returns
    -------
    pd.DataFrame
        The transposed aggregated DataFrame (topics x groups).

    Notes
    -----
    Uses a Seaborn clustermap to visualize the aggregated data.
    """
    latent = rna_anndata.obsm["X_scdori"]  # shape (n_cells, num_topics)
    df_latent = pd.DataFrame(latent, columns=[f"Topic_{i}" for i in range(latent.shape[1])])
    df_latent[groupby_key] = rna_anndata.obs[groupby_key].values

    if aggregation == "median":
        df_grouped = df_latent.groupby(groupby_key).median()
    else:
        df_grouped = df_latent.groupby(groupby_key).mean()

    if not plot:
        return df_grouped.T

    sns.set(font_scale=0.5)
    sns.clustermap(
        df_grouped.T,
        cmap="RdBu_r",
        center=0,
        figsize=(8, 8),
        row_cluster=df_grouped.shape[1] > 1,
        col_cluster=len(df_grouped) > 1,
    )
    plt.show()
    return df_grouped.T
