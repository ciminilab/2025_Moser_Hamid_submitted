import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, jaccard_score
from sklearn.metrics.cluster import homogeneity_score, adjusted_mutual_info_score
from imblearn.metrics import specificity_score
import scanpy as sc
import squidpy as sq

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

def calculate_cluster_correlations(ground_truth_clustering_df, embedding_ad):
    association_value_dict = {}
    neptune_log_string = ""

    true_labels = np.unique(ground_truth_clustering_df['cell_type'].values)
    predicted_labels = np.unique(embedding_ad.obs['clusters'].values)
    for pred_label in predicted_labels:
        for true_label in true_labels:
            true_labels_vector = [1 if x==true_label else 0 for x in ground_truth_clustering_df['cell_type'].values]
            predicted_labels_vector = [1 if x==pred_label else 0 for x in embedding_ad.obs['clusters'].values]

            jaccard_value = jaccard_score(y_pred=predicted_labels_vector, y_true=true_labels_vector)
            specificity = specificity_score(y_pred=predicted_labels_vector, y_true=true_labels_vector)
            if jaccard_value > 0.1:
                if true_label not in association_value_dict:
                    association_value_dict[true_label] = (true_label, pred_label, jaccard_value, specificity)
                else:
                    association = association_value_dict[true_label]
                    if jaccard_value > association_value_dict[true_label][2]:
                        association_value_dict[true_label] = (true_label, pred_label, jaccard_value, specificity)

    missing_clusters = list(set(predicted_labels) - set(association_value_dict.keys()))
    number_of_identified_clusters = len(association_value_dict.keys())
    neptune_log_string += f"Could associate {number_of_identified_clusters} clusters to cell types (missed {len(missing_clusters)})."
    for (true_label, pred_label, jaccard_value, specificity) in association_value_dict.values():
        neptune_log_string += f"\nPredicted label: {pred_label}, True label: {true_label}, Jaccard score: {jaccard_value}, Specificity: {specificity}"
    
    return neptune_log_string

def validate_embeddings(embeddings_df, cell_meta_data_df, ground_truth_clustering_df, neptune_run, min_dist, resolution):
    embeddings_df.sort_values(by='cell_index', inplace=True)
    cell_meta_data_df.sort_values(by='cell_index', inplace=True)
    ground_truth_clustering_df.sort_values(by='cell_index', inplace=True)

    _merged_df = pd.merge(ground_truth_clustering_df, embeddings_df, on='cell_index', how='inner')
    cel_index_list = _merged_df['cell_index'].values

    cell_meta_data_df = cell_meta_data_df[cell_meta_data_df['cell_index'].isin(cel_index_list)]
    cell_meta_data_df.set_index('cell_index', inplace=True)
    embeddings_df = embeddings_df[embeddings_df['cell_index'].isin(cel_index_list)]
    embeddings_df.set_index('cell_index', inplace=True)
    ground_truth_clustering_df = ground_truth_clustering_df[ground_truth_clustering_df['cell_index'].isin(cel_index_list)]
    ground_truth_clustering_df.set_index('cell_index', inplace=True)

    spatial_cell_coordinates = cell_meta_data_df[['global_x', 'global_y']].values
    embedding_ad = sc.AnnData(X=embeddings_df.values, obs=cell_meta_data_df.values)
    embedding_ad.obsm['spatial'] = spatial_cell_coordinates
    
    sc.pp.neighbors(embedding_ad)
    sc.tl.umap(embedding_ad, min_dist=min_dist)
    sc.tl.leiden(embedding_ad, key_added="clusters", resolution=resolution)
    sq.gr.spatial_neighbors(embedding_ad, coord_type="generic", spatial_key="spatial")
    sq.gr.interaction_matrix(embedding_ad, cluster_key="clusters")
    sq.gr.nhood_enrichment(embedding_ad, cluster_key='clusters')
    sq.gr.co_occurrence(embedding_ad, cluster_key="clusters")

    # Create the plots
    umap_plot = sc.pl.umap(embedding_ad, color=["clusters"], wspace=0.4, return_fig=True)
    embedding_spatial = sc.pl.embedding(embedding_ad, basis="spatial", color="clusters", return_fig=True)

    sq.pl.nhood_enrichment(
        embedding_ad,
        cluster_key="clusters",
        mode='zscore',
        method="average",
        figsize=(5, 5),
        cmap='Blues',
        annotate=True,
        save="nhood_enrichment_plot.png"
    )

    sq.pl.co_occurrence(embedding_ad, cluster_key="clusters", clusters=['1'], figsize=(8, 5), save="co_occurrence_plot.png")

    # Calculate evaluation scores:
    homogeneity_score_value = homogeneity_score(
            embedding_ad.obs['clusters'],
            ground_truth_clustering_df['cell_type']
    )
    adjusted_mutual = adjusted_mutual_info_score(embedding_ad.obs['clusters'], ground_truth_clustering_df['cell_type'])

    neptune_log_string = calculate_cluster_correlations(ground_truth_clustering_df, embedding_ad)

    if neptune_run:
        neptune_run["embedding_validation/clusters_umap_coordinates"].upload(umap_plot)
        neptune_run["embedding_validation/clusters_spatial_coordinates"].upload(embedding_spatial)
        neptune_run["embedding_validation/nhood_enrichment"].upload('figures/nhood_enrichment_plot.png')
        neptune_run["embedding_validation/co_occurrence"].upload("figures/co_occurrence_plot.png")
        # Log to cluster metrics to Neptune
        neptune_run['embedding_validation/silhouette_score'] = silhouette_score(
            embedding_ad.obsm['spatial'],
            embedding_ad.obs['clusters']
        )
        neptune_run['embedding_validation/homogeneity_score'] = homogeneity_score_value
        neptune_run['embedding_validation/adjusted_mutual_info_score'] = adjusted_mutual

        neptune_run['embedding_validation/cluster_correlation'] = neptune_log_string
