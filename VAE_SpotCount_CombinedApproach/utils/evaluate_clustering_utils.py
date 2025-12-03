import os
import numpy as np
import pandas as pd
import squidpy as sq
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from sklearn.metrics import (silhouette_score, homogeneity_score, adjusted_mutual_info_score,
                             jaccard_score, adjusted_rand_score, roc_auc_score, balanced_accuracy_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from imblearn.metrics import specificity_score, sensitivity_score
import warnings
import tempfile


from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from scipy.optimize import linear_sum_assignment

import muon as mu
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from VAE_SpotCount_CombinedApproach.utils.anndata_utils import *
from VAE_SpotCount_CombinedApproach.utils.anndata_utils import clear_clustering_data


def calculate_cluster_correlations(ground_truth_clustering_df, data_object, clustering_key):
    """
    Calculate the correlation between true labels and predicted clusters using Jaccard, specificity and sensitivity scores.

    Parameters:
        ground_truth_clustering_df (pd.DataFrame): DataFrame containing the ground truth cell type labels.
        data_object (AnnData or MuData): AnnData or MuData object containing the predicted clusters in its .obs attribute.

    Returns:
        dict: Dictionary containing the summary (number of identified clusters, missing clusters) and detailed results
              (Jaccard score, specificity, and sensitivity scores for each cluster pair).
    """
    association_value_dict = {}  # Initialize dict for association values
    results_dict = {}  # Initialize dict for detailed results

    # Get unique true and predicted labels
    true_labels = np.unique(ground_truth_clustering_df['cell_type'].values)
    predicted_labels = np.unique(data_object.obs[clustering_key].values)

    # Iterate through each pair of predicted and true labels
    for pred_label in predicted_labels:
        for true_label in true_labels:
            # Create binary vectors for each label
            true_labels_vector = [1 if x == true_label else 0 for x in ground_truth_clustering_df['cell_type'].values]
            predicted_labels_vector = [1 if x == pred_label else 0 for x in data_object.obs[clustering_key].values]

            # Calculate Jaccard, specificity, and sensitivity scores
            jaccard_value = jaccard_score(y_pred=predicted_labels_vector, y_true=true_labels_vector)
            specificity = specificity_score(y_pred=predicted_labels_vector, y_true=true_labels_vector)
            sensitivity = sensitivity_score(y_pred=predicted_labels_vector, y_true=true_labels_vector)

            # Store results if Jaccard score is above threshold
            if jaccard_value > 0.1:
                results_dict[(true_label, pred_label)] = {
                    'jaccard_score': jaccard_value,
                    'specificity': specificity,
                    'sensitivity': sensitivity
                }

    # Determine missing clusters
    missing_clusters = list(set(predicted_labels) - {key[1] for key in results_dict.keys()})
    number_of_identified_clusters = len(results_dict)

    # Add data to dict
    association_value_dict['summary'] = {
        'number_of_identified_clusters': number_of_identified_clusters,
        'missing_clusters': missing_clusters
    }
    association_value_dict['detailed_results'] = results_dict

    return association_value_dict


def calculate_clustering_metrics(data_object, ground_truth_clustering_df, clustering_key):
    """
    Calculate various clustering metrics for a given AnnData object.

    Parameters:
        data_object (AnnData or MuData): AnnData or MuData object containing clustering results in its .obs attribute.
        ground_truth_clustering_df (pd.DataFrame): DataFrame containing the ground truth cell type labels.

     Returns:
        dict: Dictionary containing silhouette, homogeneity, adjusted mutual information, and ARI scores.
    """
    # Check if the clustering is MOFA-based so metrics are computed in the right space
    if "mofa" in clustering_key.lower():
        useMOFASpace = True
    else:
        useMOFASpace = False

    # Sanity check
    print(f"Calculating clustering metrics in {'MOFA' if useMOFASpace else 'data'} space")

    if isinstance(data_object, sc.AnnData) and len(data_object.obs[clustering_key].unique()) > 1:
        # Calculate silhouette score if there are at least two clusters
        if not useMOFASpace:
            silhouette = silhouette_score(data_object.X, data_object.obs[clustering_key])
        else:
            silhouette = silhouette_score(data_object.obsm["X_mofa_std"], data_object.obs[clustering_key])
    elif isinstance(data_object, mu.MuData) and len(data_object.obs[clustering_key].unique()) > 1:
        if not useMOFASpace:
            # Assumes only two modalities
            # Fetching modality keys
            mod1 = list(data_object.mod.keys())[0]
            mod2 = list(data_object.mod.keys())[1]

            # Calculate silhouette score if there are at least two clusters
            # As this is for MuData objects, the silhouette score is computed for each modality with the data used for the distance matrix, and then the average is used
            silhouette1 = silhouette_score(data_object.mod[mod1].X, data_object.obs[clustering_key])
            silhouette2 = silhouette_score(data_object.mod[mod2].X, data_object.obs[clustering_key])
            silhouette = (silhouette1 + silhouette2) / 2
        else:
            silhouette = silhouette_score(data_object.obsm["X_mofa_std"], data_object.obs[clustering_key])
    else:
        # Set silhouette score to NaN if there is only one cluster or wrong object type
        silhouette = np.nan

    # Calculate homogeneity score
    homogeneity = homogeneity_score(data_object.obs[clustering_key], ground_truth_clustering_df['cell_type'])

    # Calculate adjusted mutual information score
    adjusted_mutual = adjusted_mutual_info_score(data_object.obs[clustering_key], ground_truth_clustering_df['cell_type'])

    # Calculate adjusted Rand index
    ari = adjusted_rand_score(
        data_object.obs[clustering_key],
        ground_truth_clustering_df['cell_type']
    )

    return {
        'silhouette_score': silhouette,
        'homogeneity_score': homogeneity,
        'adjusted_mutual_info': adjusted_mutual,
        'adjusted_rand_index': ari,
    }


def generate_interaction_plots(AnnData_object, key, clustering_key, output_directory=None, export_individual=False):
    """
    Generate interaction plots (neighborhood enrichment, co-occurrence) for an AnnData object and optionally export them.

    Parameters:
        AnnData_object (AnnData): AnnData object containing spatial and clustering information.
        key (str): Identifier for the AnnData object, used in filenames if exporting plots.
        output_directory (str, optional): Directory to save the plots. If None, plots are not saved.
        export_individual (bool, optional): If True, exports individual plots to output_directory.
    """
    # Assign colors to clusters if not already present
    if 'clusters_colors' not in AnnData_object.uns:
        unique_clusters = AnnData_object.obs[clustering_key].unique()
        num_clusters = len(unique_clusters)
        cmap = plt.get_cmap('tab20')  # Use a colormap to generate colors for the clusters
        colors = [cmap(i / num_clusters) for i in range(num_clusters)]
        AnnData_object.uns['clusters_colors'] = colors

    # Calculate spatial neighbors and interaction matrices
    sq.gr.spatial_neighbors(AnnData_object, coord_type="generic", spatial_key="spatial")
    sq.gr.interaction_matrix(AnnData_object, cluster_key=clustering_key)
    sq.gr.nhood_enrichment(AnnData_object, cluster_key=clustering_key)
    sq.gr.co_occurrence(AnnData_object, cluster_key=clustering_key)

    # Export individual plots if requested
    if export_individual and output_directory:
        os.makedirs(output_directory, exist_ok=True)  # Create directory if it doesn't exist

        # Export neighborhood enrichment plot
        sq.pl.nhood_enrichment(AnnData_object, cluster_key=clustering_key, mode='zscore', method="average",
                               figsize=(10, 10), cmap='Blues', annotate=True, title=key,
                               save=f"{output_directory}/{key}_nhood_enrichment_plot.png")

        # Create the co-occurrence plot without saving it directly, for title param
        sq.pl.co_occurrence(AnnData_object, cluster_key=clustering_key, clusters=['1'], figsize=(8, 5))
        plt.title(f'Co-occurrence Plot for {key}', fontsize=16)

        # Save the plot with the title
        plt.savefig(f"{output_directory}/{key}_co_occurrence_plot.png", bbox_inches='tight', dpi=600)


def create_metrics_dataframe(metrics_dict, output_directory=None):
    """
    Create a DataFrame from the metrics dictionary and optionally export it as a CSV.

    Parameters:
        metrics_dict (dict): Dictionary containing clustering metrics for each AnnData object.
        output_directory (str, optional): Directory to save the metrics DataFrame as a CSV.

    Returns:
        pd.DataFrame: DataFrame containing clustering metrics for each AnnData object.
    """
    # Create DataFrame from metrics dictionary
    metrics_df = pd.DataFrame(metrics_dict)

    # Export to CSV if output directory is provided
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)  # Create directory if it doesn't exist
        metrics_df.to_csv(f'{output_directory}/clustering_metrics.csv', index=True)

    return metrics_df


def plot_clustering_metrics(metrics_df, dataset_custom_order_list=None, dataset_name_substitution_dict=None,
                            output_directory=None, export_grid=False, font_size=14,
                            vertical_dashed_line_positions=None, leiden_optimization_results=None):
    """
    Plot clustering metrics with custom group labels and brackets.

    Parameters:
        metrics_df (pd.DataFrame): DataFrame containing clustering metrics.
        dataset_custom_order_list (list): List containing column labels in the order to be shown in the plot.
        dataset_name_substitution_dict (dict, optional): Nested dictionary containing an alternative naming system for
            each dataset which will be used in the generated plot.
                E.g. {'GT-Based Subset + LS2': {'group': 'GT-Based Subset', 'new_name': 'TC+LS2'}}
        output_directory (str, optional): Directory to save the grid plot.
        export_grid (bool, optional): If True, exports the grid plot to output_directory.
        font_size (int, optional): Overall plot font size.
        vertical_dashed_line_positions (tuple, optional): Positions for vertical dashed lines.
    """
    if dataset_custom_order_list:
        metrics_df = metrics_df[dataset_custom_order_list]

    plt.rcParams.update({'font.size': font_size})
    metrics = metrics_df.index
    fig, axes = plt.subplots(nrows=len(metrics), ncols=1, figsize=(15, 16), sharex=True)

    # Prepare new x-axis labels
    if dataset_name_substitution_dict:
        new_labels = [dataset_name_substitution_dict.get(col, {}).get('new_name', col) for col in metrics_df.columns]
        groups = [dataset_name_substitution_dict.get(col, {}).get('group', None) for col in metrics_df.columns]
    else:
        new_labels = metrics_df.columns
        groups = [None] * len(metrics_df.columns)

    # Group position tracking
    group_positions = {}
    for idx, group in enumerate(groups):
        if group:
            group_positions.setdefault(group, []).append(idx)

    for i, metric in enumerate(metrics):
        bars = axes[i].bar(range(len(metrics_df.columns)), metrics_df.loc[metric],
                           color=plt.cm.Paired.colors, width=0.6)
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
        axes[i].set_ylabel('Score')
        axes[i].grid(True, linestyle='--', alpha=0.6)

        if i == 0:
            axes[i].set_ylim(-1, 1)
        else:
            axes[i].set_ylim(0, 1)

        for j, bar in enumerate(bars):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}',
                         ha='center', va='bottom', fontsize=font_size - 4, fontweight='bold')

        if vertical_dashed_line_positions:
            for pos in vertical_dashed_line_positions:
                if pos < len(metrics_df.columns):
                    axes[i].axvline(x=pos + 0.5, color='black', linestyle='dashed', alpha=0.7)

    # Set x-axis labels
    if dataset_name_substitution_dict is None:
        if leiden_optimization_results is None:
            axes[-1].set_xlabel('Feature Space')  # Brackets and weights disabled
        else:
            axes[-1].set_xlabel('Feature Space', labelpad=40)  # Brackets disabled and weights enabled

    else:
        axes[-1].set_xlabel('Feature Space', labelpad=80)  # Brackets and weights enabled

    axes[-1].set_xticks(range(len(metrics_df.columns)))
    axes[-1].set_xticklabels(new_labels, rotation=45, ha='right')

    y_offset = -0.6
    # TC weights row insertion
    if leiden_optimization_results is not None:
        # Extracting weights
        weights = []
        for dataset_name in dataset_custom_order_list:
            if dataset_name not in leiden_optimization_results['MuData']['results'].keys():
                TC_weight = 1.00
            else:
                TC_weight = next(
                    iter(leiden_optimization_results['MuData']['results'][dataset_name]['weights'].values()))

            weights.append(TC_weight)

        # Adding title for weights row
        axes[-1].text(-0.5, y_offset + 0.2, 'TC weight:', ha='right', va='top', rotation=0, fontsize=font_size - 2,
                      transform=axes[-1].get_xaxis_transform())

        # # Adding weights row
        for i, label in enumerate(new_labels):
            axes[-1].text(i, y_offset + 0.2, weights[i], ha='right', va='top', rotation=0,
                          fontsize=font_size - 2,
                          transform=axes[-1].get_xaxis_transform())

    # Draw group brackets
    for group, positions in group_positions.items():
        if len(positions) > 1:
            start = positions[0]
            end = positions[-1]
            mid = (start + end) / 2

            # Draw horizontal line
            axes[-1].plot([start - 0.3, end + 0.3], [y_offset, y_offset], color='black', lw=1.5, clip_on=False)

            # Draw vertical ticks
            axes[-1].plot([start - 0.3, start - 0.3], [y_offset, y_offset + 0.02], color='black', lw=1.5, clip_on=False)
            axes[-1].plot([end + 0.3, end + 0.3], [y_offset, y_offset + 0.02], color='black', lw=1.5, clip_on=False)

            # Add group label
            axes[-1].text(mid, y_offset - 0.05, group, ha='center', va='top', fontsize=font_size)

    plt.tight_layout()

    if export_grid and output_directory:
        plt.savefig(f'{output_directory}/clustering_metrics_grid.png', dpi=600)

    plt.show()


def export_cluster_correlations(cluster_correl_dict_of_dicts, ground_truth_clustering_df, output_directory=None):
    """
    Combines and sorts cluster correlation data into a DataFrame and optionally exports it as a CSV.
    Also returns a dictionary mapping each AnnData object to its matched cluster number and true label.

    Parameters:
        cluster_correl_dict_of_dicts (dict): Dictionary containing clustering correlation data for each AnnData object.
        ground_truth_clustering_df (pd.DataFrame): DataFrame containing ground truth cell type labels.
        output_directory (str, optional): Directory to save the output CSV file. If None, the DataFrame is not exported.

    Returns:
        pd.DataFrame: Combined DataFrame with cluster correlation data.
        dict: Dictionary where keys are AnnData object names and values are tuples of (cluster number, true label).
    """
    true_labels_list = np.unique(ground_truth_clustering_df['cell_type'].values)

    # Initialize list to store rows for the combined DataFrame
    combined_rows = []

    # Initialize dictionary to store matched clusters and true labels
    cluster_to_true_label_dict = {}

    # Iterate through true labels
    for label in true_labels_list:
        for var_name, var_value in cluster_correl_dict_of_dicts.items():
            # Iterate through detailed_results with the tuple keys
            match_found = False
            for (true_label, pred_label), metrics in var_value['detailed_results'].items():
                # Check if the current label matches
                if true_label == label:
                    match_found = True
                    combined_rows.append({
                        'true_label': label,
                        'merfish_ad': var_name,
                        'predicted_label': pred_label,
                        'jaccard_score': metrics['jaccard_score'],
                        'specificity': metrics['specificity'],
                        'sensitivity': metrics['sensitivity']
                    })

                    # Add to the dictionary
                    if var_name not in cluster_to_true_label_dict:
                        cluster_to_true_label_dict[var_name] = []
                    cluster_to_true_label_dict[var_name].append((pred_label, true_label))

            if not match_found:  # Append empty row if no cluster matched
                combined_rows.append({
                    'true_label': label,
                    'merfish_ad': var_name,
                    'predicted_label': None,
                    'jaccard_score': None,
                    'specificity': None,
                    'sensitivity': None
                })

    # Convert the list of dicts to a DataFrame
    cluster_correl_combined_df = pd.DataFrame(combined_rows)

    # Save the DataFrame as a CSV if an output directory is provided
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)  # Create directory if it doesn't exist
        output_filename = os.path.join(output_directory, 'cluster_correl_combined.csv')
        cluster_correl_combined_df.to_csv(output_filename, index=False)
        print(f"Exported cluster correlation DataFrame to {output_filename}")

    return cluster_correl_combined_df, cluster_to_true_label_dict


def aggregate_cluster_correlations(cluster_correl_dict_of_dicts, ground_truth_clustering_df):
    """
    Aggregate cluster correlations into a DataFrame, filling in missing values where no cluster matched.

    Parameters:
        cluster_correl_dict_of_dicts (dict): Dictionary containing clustering correlation data for each AnnData object.
        ground_truth_clustering_df (pd.DataFrame): DataFrame containing ground truth cell type labels.

    Returns:
        pd.DataFrame: DataFrame containing aggregated cluster correlation data.
    """
    true_labels_list = np.unique(ground_truth_clustering_df['cell_type'].values)
    combined_rows = []

    # Iterate through true labels and cluster correlation dictionaries
    for label in true_labels_list:
        for var_name, var_value in cluster_correl_dict_of_dicts.items():
            match_found = False
            for (true_label, pred_label), metrics in var_value['detailed_results'].items():
                if true_label == label:
                    match_found = True
                    combined_rows.append({
                        'true_label': label,
                        'merfish_ad': var_name,
                        'predicted_label': pred_label,
                        'jaccard_score': metrics['jaccard_score'],
                        'specificity': metrics['specificity'],
                        'sensitivity': metrics['sensitivity']
                    })
            if not match_found:  # Append row if no cluster matched
                combined_rows.append({
                    'true_label': label,
                    'merfish_ad': var_name,
                    'predicted_label': None,
                    'jaccard_score': None,
                    'specificity': None,
                    'sensitivity': None
                })

    return pd.DataFrame(combined_rows)


def generate_heatmaps(aggregated_df, metric, ground_truth_clustering_df, output_directory=None,
                      dataset_custom_order_list=None, sorted_cell_counts=None, color_map='Purples',
                      font_size=14, dataset_name_substitution_dict=None):
    """
    Generate heatmaps for clustering metrics with custom group labels and brackets.

    Parameters:
        aggregated_df (pd.DataFrame): DataFrame containing aggregated cluster correlation data.
        metric (str): The metric to be used for the heatmap ('jaccard_score' or 'specificity' or 'sensitivity').
        output_directory (str, optional): Directory to save the heatmap PNG file.
        dataset_custom_order_list (list, optional): Determines the heatmap column order.
        sorted_cell_counts (dict, optional): Sorted cell counts for reordering true labels.
        color_map (str, optional): Color map for the heatmap.
        font_size (int, optional): Font size for heatmaps.
        dataset_name_substitution_dict (dict, optional): Nested dictionary containing an alternative naming system for
            each dataset which will be used in the generated plot.
                E.g. {'GT-Based Subset + LS2': {'group': 'GT-Based Subset', 'new_name': 'TC+LS2'}}
    """
    plt.rcParams.update({'font.size': font_size})

    aggregated_df = aggregated_df.copy()
    unique_adObj_list = aggregated_df['merfish_ad'].unique()
    for adObj_name in unique_adObj_list:
        filteredDF = aggregated_df[aggregated_df['merfish_ad'] == adObj_name].reset_index(drop=True)
        filteredDF.drop(columns=['specificity', 'sensitivity', 'merfish_ad'], inplace=True)
        label_matrix = filteredDF.pivot(index='true_label', columns='predicted_label', values='jaccard_score').fillna(0)
        cost_matrix = -label_matrix.values
        rowIndices, colIndices = linear_sum_assignment(cost_matrix)
        matches = [(label_matrix.index[row], label_matrix.columns[col], -cost_matrix[row, col])
                   for row, col in zip(rowIndices, colIndices)]

        aggregated_df.loc[~aggregated_df.apply(
            lambda row: (row['true_label'], row['predicted_label'], row['jaccard_score']) in matches
            if row['merfish_ad'] == adObj_name else True, axis=1
        )] = None
        aggregated_df.dropna(inplace=True)

    true_label_set = set(ground_truth_clustering_df['cell_type'].unique())
    missing_true_labels = true_label_set - set(aggregated_df['true_label'].unique())
    if missing_true_labels:
        missing_rows = pd.DataFrame([
            {'true_label': true_label, 'predicted_label': None, 'jaccard_score': None,
             'merfish_ad': adObj_name, 'specificity': None, 'sensitivity': None}
            for true_label in missing_true_labels for adObj_name in unique_adObj_list
        ])
        aggregated_df = pd.concat([aggregated_df, missing_rows], ignore_index=True)

    heatmap_data = aggregated_df.pivot(index='true_label', columns='merfish_ad', values=metric)

    if sorted_cell_counts:
        heatmap_data = heatmap_data.reindex(index=list(sorted_cell_counts.keys()))

    if dataset_custom_order_list:
        heatmap_data = heatmap_data.reindex(columns=dataset_custom_order_list)

    if dataset_name_substitution_dict:
        new_labels = [dataset_name_substitution_dict.get(col, {}).get('new_name', col) for col in heatmap_data.columns]
        groups = [dataset_name_substitution_dict.get(col, {}).get('group', None) for col in heatmap_data.columns]
    else:
        new_labels = heatmap_data.columns
        groups = [None] * len(heatmap_data.columns)

    group_positions = {}
    for idx, group in enumerate(groups):
        if group:
            group_positions.setdefault(group, []).append(idx)

    plt.figure(figsize=(15, 15), facecolor='white')
    ax = sns.heatmap(heatmap_data, annot=True, cmap=color_map,
                     cbar_kws={'label': metric.replace("_", " ").title()},
                     linewidths=0.5, annot_kws={'size': font_size - 2})

    ax.set_facecolor('lightgrey')
    plt.title(f'{metric.replace("_", " ").title()} Heatmap', fontsize=font_size + 10)
    # Adjusting x-axis label depending on whether brackets are enabled
    if dataset_name_substitution_dict is not None:
        plt.xlabel('Feature Space', fontsize=font_size + 2, labelpad=50)
    else:
        plt.xlabel('Feature Space', fontsize=font_size + 2)

    plt.ylabel('True Labels', fontsize=font_size + 2)

    # Center xticks with heatmap grid
    ax.set_xticks([i + 0.5 for i in range(len(heatmap_data.columns))])
    ax.set_xticklabels(new_labels, rotation=45, ha='right')

    # Offsets
    y_offset = 20.7  # For horizontal lines
    bracket_padding = 0.3  # Padding for horizontal line to prevent overlap
    label_offset = 0.3  # Vertical offset for line labels to avoid overlap

    for group, positions in group_positions.items():
        if len(positions) > 1:
            start = positions[0]
            end = positions[-1]
            mid = (start + end) / 2  # Midpoint for group label

            start_aligned = start + 0.5
            end_aligned = end + 0.5

            # Draw horizontal bracket line
            ax.plot([start_aligned - bracket_padding, end_aligned + bracket_padding],
                    [y_offset, y_offset], color='black', lw=1.5, clip_on=False)

            # Draw vertical ticks at both ends
            ax.plot([start_aligned - bracket_padding, start_aligned - bracket_padding],
                    [y_offset, y_offset - 0.1], color='black', lw=1.5, clip_on=False)
            ax.plot([end_aligned + bracket_padding, end_aligned + bracket_padding],
                    [y_offset, y_offset - 0.1], color='black', lw=1.5, clip_on=False)

            # Add group label below the bracket
            ax.text(mid + 0.5, y_offset + label_offset, group,
                    ha='center', va='top', fontsize=font_size)

    plt.tight_layout()

    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
        plt.savefig(f'{output_directory}/{metric}_heatmap.png', bbox_inches='tight', dpi=600)

    plt.show()


def evaluate_clustering(data_object_dictionary, ground_truth_dataframe, output_directory, clustering_key, sort_labels=False,
                        generate_interaction_plots=True, generate_jaccard_heatmap=True,
                        generate_specificity_heatmap=True, generate_sensitivity_heatmap=True,
                        dataset_custom_order_list=None, dataset_name_substitution_dict=None,
                        vertical_dashed_line_positions=None, leiden_optimization_results=None, color_map='Purples',
                        font_size=14):
    """
    Runs clustering evaluation generating neighborhood enrichment matrices, co-occurrence plots, Jaccard/specificity/sensitivity heatmaps.
    Hinges on many other functions in this script.

    Parameters:
        data_object_dictionary (dict): Dictionary containing AnnData objects.
        ground_truth_dataframe (pd.DataFrame): DataFrame containing ground truth cell type labels.
        output_directory (str): Directory to save the plots.
        clustering_key (str): key for stored clustering in data object
        sort_labels (bool): Whether to sort true labels by cell type count (default: False).
        generate_interaction_plots (bool): Whether to generate interaction plots (default: True).
        generate_jaccard_heatmap (bool): Whether to generate jaccard heatmaps (default: True).
        generate_specificity_heatmap (bool): Whether to generate specificity heatmaps (default: True).
        generate_sensitivity_heatmap (bool): Whether to generate sensitivity heatmaps (default: True).
        dataset_custom_order_list (list): List containing column labels in the order to be shown in the plot.
        vertical_dashed_line_positions (tuple, optional): Tuple of positions to use for vertical lines. Positions
            represent indices from the dataset_custom_order_list after which a dashed vertical line is drawn on the grid.
        color_map (str, optional): Color map to be used for the heatmap, see matplotlib cmaps for arguments.
        font_size (int, optional): Font size for the metrics grid and heatmap (default: 14).

    Returns:
        None (exports data and displays in the Jupyter notebook)
    """
    # Calculate clustering metrics for each AnnData object
    clusterMetricsDict = {}
    cluster_correl_dict_of_dicts = {}
    for key, adObj in data_object_dictionary.items():
        clusterMetricsDict[key] = calculate_clustering_metrics(adObj, ground_truth_dataframe, clustering_key=clustering_key)
        cluster_correl_dict_of_dicts[key] = calculate_cluster_correlations(ground_truth_dataframe, adObj,
                                                                           clustering_key=clustering_key)
        if generate_interaction_plots:
            generate_interaction_plots(adObj, key, output_directory=output_directory, export_individual=True,
                                       clustering_key=clustering_key)

    # Export cluster correlation metrics
    cluster_matching_dict = \
        export_cluster_correlations(cluster_correl_dict_of_dicts, ground_truth_dataframe,
                                    output_directory=output_directory)[1]

    # Create and export metrics DataFrame (Silhouette, homogeneity, and mutual adjusted info scores)
    metrics_df = create_metrics_dataframe(clusterMetricsDict, output_directory=output_directory)

    # Plot clustering metrics and export
    plot_clustering_metrics(metrics_df, dataset_custom_order_list=dataset_custom_order_list,
                            output_directory=output_directory, export_grid=True, font_size=font_size,
                            vertical_dashed_line_positions=vertical_dashed_line_positions,
                            dataset_name_substitution_dict=dataset_name_substitution_dict,
                            leiden_optimization_results=leiden_optimization_results)

    # Aggregate cluster correlations (to enable heatmap generation)
    aggregated_df = aggregate_cluster_correlations(cluster_correl_dict_of_dicts, ground_truth_dataframe)

    # Sort cell types by count if sort_labels is True
    sorted_cell_counts = None
    if sort_labels:
        # Create sorted dictionary based on cell counts from ground_truth_dataframe
        cell_counts = ground_truth_dataframe['cell_type'].value_counts().to_dict()
        sorted_cell_counts = dict(sorted(cell_counts.items(), key=lambda item: item[1], reverse=True))

    # Generate heatmaps for Jaccard, specificity, and sensitivity scores passing sorted labels if required
    if generate_jaccard_heatmap:
        generate_heatmaps(aggregated_df, 'jaccard_score', ground_truth_clustering_df=ground_truth_dataframe,
                          output_directory=output_directory, dataset_custom_order_list=dataset_custom_order_list,
                          sorted_cell_counts=sorted_cell_counts, color_map=color_map, font_size=font_size,
                          dataset_name_substitution_dict=dataset_name_substitution_dict)
    if generate_specificity_heatmap:
        generate_heatmaps(aggregated_df, 'specificity', ground_truth_clustering_df=ground_truth_dataframe,
                          output_directory=output_directory, dataset_custom_order_list=dataset_custom_order_list,
                          sorted_cell_counts=sorted_cell_counts, color_map=color_map, font_size=font_size,
                          dataset_name_substitution_dict=dataset_name_substitution_dict)
    if generate_sensitivity_heatmap:
        generate_heatmaps(aggregated_df, 'sensitivity', ground_truth_clustering_df=ground_truth_dataframe,
                          output_directory=output_directory, dataset_custom_order_list=dataset_custom_order_list,
                          sorted_cell_counts=sorted_cell_counts, color_map=color_map, font_size=font_size,
                          dataset_name_substitution_dict=dataset_name_substitution_dict)


def optimize_leiden_hyperparams(object, min_clusters=None, k_range=None, resolution_range=None,
                                resolution_threshold=None, increment_multiple_after_threshold=None,
                                optimal_MuDataObj_resolution_dict=None,
                                modality_weight_increment=None, global_seed=137):
    """
    Runs a grid search to optimize the hyperparameters for mono- or multiplex Leiden clustering.
    Handles errors gracefully by skipping invalid parameter combinations.

    Parameters:
        object: An AnnData or MuData object.
        min_clusters (int): Minimum number of clusters to accept during grid search. Defaults to no minimum.
        k_range (tuple): Range and increment for the n_neighbors argument in sc.pp.neighbors(), (start, end, increment).
        resolution_range (tuple): Range and increment for the resolution argument in sc.pp.neighbors(), (start, end, increment).
        resolution_threshold (number): Threshold after which resolution increment is the argument provided in resolution_range[2] multiplied by the increment_multiple_after_threshold argument.
        increment_multiple_after_threshold (number): The resolution increment is multiplied by this argument after the resolution_threshold.
        optimal_MuDataObj_resolution_dict (dict): A dictionary containing the previously optimized resolution parameters for each modality in a MuData object.
        modality_weight_increment (float): The increment used for the grid search of the weight parameter in mu.tl.leiden().

    Returns:
        best_params (dict): A dictionary containing the best hyperparameters as per Silhouette score.
    """
    # Initializing variables
    best_score = -np.inf
    best_params = None

    try:
        # Determine object nature and conduct grid search
        if isinstance(object, sc.AnnData):
            # Grid search for optimal k and resolution
            list1 = np.arange(resolution_range[0], resolution_threshold + resolution_range[2], resolution_range[2])
            if (resolution_range[1] + resolution_range[2]) > resolution_threshold:
                list2 = np.arange(resolution_threshold + (resolution_range[2] * increment_multiple_after_threshold),
                                  resolution_range[1] + (resolution_range[2] * increment_multiple_after_threshold),
                                  resolution_range[2] * increment_multiple_after_threshold)
                resolution_list = np.concatenate((list1, list2))
            else:
                resolution_list = list1

            for k in range(k_range[0], k_range[1] + k_range[2], k_range[2]):
                for resolution in resolution_list:
                    try:
                        # Clearing previous clustering data
                        tempDict = {"object": object}
                        clear_clustering_data(tempDict, key_added='leiden_clusters')

                        # Generating kNN and clustering
                        sc.pp.neighbors(object, n_neighbors=k, n_pcs=0, random_state=global_seed)
                        sc.tl.leiden(object, key_added='leiden_clusters', resolution=resolution)

                        # Skipping if below minimum number of clusters
                        if min_clusters is not None:
                            nClusters = object.obs['leiden_clusters'].nunique()
                            if nClusters <= min_clusters:
                                continue

                        # Computing silhouette score
                        silhouette = silhouette_score(object.X, object.obs['leiden_clusters'])

                        # Comparing to previous best score
                        if silhouette > best_score:
                            best_score = silhouette
                            best_params = {
                                'n_neighbors': k,
                                'resolution': resolution,
                                'silhouette_score': silhouette
                            }
                    except Exception as e:
                        print(f"Error during AnnData optimization with k={k}, resolution={resolution}: {e}")

        elif isinstance(object, mu.MuData):
            # Generating list of weight tuples
            weights_list = [(round(weight, 2), round(1 - round(weight, 2), 2)) for weight in
                            np.arange(0 + modality_weight_increment, 1 + modality_weight_increment,
                                      modality_weight_increment) if 0 < round(weight, 2) < 1]

            # Fetching modality names
            mod1 = list(object.mod.keys())[0]
            mod2 = list(object.mod.keys())[1]

            for weights in weights_list:
                try:
                    weights_dict = {
                        mod1: weights[0],
                        mod2: weights[1]
                    }

                    # Running multiplex Leiden clustering
                    mu.tl.leiden(object, resolution=optimal_MuDataObj_resolution_dict, key_added='leiden_clusters',
                                 mod_weights=weights_dict, random_state=global_seed)

                    # Skipping if below minimum number of clusters
                    if min_clusters is not None:
                        nClusters = object.obs['leiden_clusters'].nunique()
                        if nClusters <= min_clusters:
                            continue

                    # Computing Silhouette score
                    silhouette1 = silhouette_score(object.mod[mod1].X, object.obs['leiden_clusters'])
                    silhouette2 = silhouette_score(object.mod[mod2].X, object.obs['leiden_clusters'])
                    silhouette = (silhouette1 + silhouette2) / 2

                    # Comparing to previous best score
                    if silhouette > best_score:
                        best_score = silhouette
                        best_params = {
                            'weights': weights_dict,
                            'silhouette_score': silhouette
                        }
                except Exception as e:
                    print(f"Error during MuData optimization with weights={weights}: {e}")
        else:
            raise TypeError("Object must be of either the AnnData or MuData classes.")
    except Exception as e:
        print(f"Critical error in optimize_leiden_hyperparams: {e}")

    return best_params

#
# def optimize_mofa_kmeans_hyperparams(
#     obj,
#     *,
#     min_clusters=None,
#     n_factors_range=(5, 20, 5),    # (start, end, step)
#     n_clusters_range=(5, 20, 1),   # (start, end, step)
#     likelihood="gaussian",
#     gpu_mode=True,
#     global_seed=137,
# ):
#     """
#     Grid search for MOFA (1 view) + k-means on a single AnnData.
#
#     Parameters
#     ----------
#     obj : AnnData or MuData
#         Single-modality AnnData to use as one MOFA view or multi-modality MuData.
#     min_clusters : int or None
#         Minimum number of clusters required to accept a solution.
#     n_factors_range : (int, int, int)
#         (start, end, step) for MOFA n_factors.
#     n_clusters_range : (int, int, int)
#         (start, end, step) for k-means n_clusters.
#     likelihood : str
#         MOFA likelihood for this modality ("gaussian" for your data).
#     gpu_mode : bool
#         Whether to ask MOFA to use GPU (falls back to CPU if unavailable).
#     global_seed : int
#         Random seed for both MOFA and k-means.
#
#     Returns
#     -------
#     best_params : dict or None
#         {
#           "n_factors": int,
#           "n_clusters": int,
#           "silhouette_score": float,
#         }
#         or None if nothing passed the min_clusters filter.
#     """
#     if not isinstance(obj, (sc.AnnData, mu.MuData)):
#         raise TypeError("optimize_mofa_kmeans_hyperparams expects AnnData or MuData.")
#
#     # Make sure there are no NaNs in X
#     if isinstance(obj, sc.AnnData):
#         ad_copy = obj.copy()
#         ad_copy.X = np.nan_to_num(ad_copy.X, nan=np.nanmean(ad_copy.X, axis=0))
#         views = {"view": ad_copy}
#
#         # Compute max factors data supports (min of cells and features)
#         max_factors = int(min(obj.n_vars, obj.n_obs))
#
#     elif isinstance(obj, mu.MuData):
#         # Use all modalities as MOFA views
#         views = {}
#         for mod_key, mod_obj in obj.mod.items():
#             mod_copy = mod_obj.copy()
#             mod_copy.X = np.nan_to_num(mod_copy.X, nan=np.nanmean(mod_copy.X, axis=0))
#             views[mod_key] = mod_copy
#
#         # Computing max factors the data supports (min of cells and features of all modalities)
#         min_vars = min(mod_obj.n_vars for mod_obj in obj.mod.values())
#         min_obs = min(mod_obj.n_obs for mod_obj in obj.mod.values())
#         max_factors = int(min(min_vars, min_obs))
#
#     # Prebuild ranges
#     n_factors_raw = list(range(n_factors_range[0],
#                                n_factors_range[1] + 1,
#                                n_factors_range[2]))
#
#     # keep only n_factors <= max_factors
#     n_factors_list = [k for k in n_factors_raw if k <= max_factors]
#
#     if not n_factors_list:
#         warnings.warn(
#             f"All requested n_factors ({n_factors_raw}) exceed max_factors={max_factors}; "
#             "no MOFA fits attempted.",
#             UserWarning,
#         )
#         return None
#
#     n_clusters_list = list(range(n_clusters_range[0], n_clusters_range[1] + 1, n_clusters_range[2]))
#
#     best_score = -np.inf
#     best_params = None
#
#     # for n_factors in n_factors_list:
#     #     try:
#     #         mtmp = mu.MuData(views)
#     #
#     #         # Fit MOFA
#     #         mu.tl.mofa(
#     #             mtmp,
#     #             n_factors=n_factors,
#     #             likelihoods=likelihood,
#     #             gpu_mode=gpu_mode,
#     #             seed=global_seed,
#     #         )
#     #
#     #         # Extract latent factors
#     #         F = mtmp.obsm["X_mofa"]
#     #
#     #         # Standardize factors before k-means
#     #         F_std = StandardScaler().fit_transform(F)
#
#     for n_factors in n_factors_list:
#         try:
#             mtmp = mu.MuData(views)
#
#             # Unique outfile per process to avoid HDF5 lock clashes
#             tmp_outfile = os.path.join(
#                 tempfile.gettempdir(),
#                 f"mofa_{os.getpid()}.hdf5",
#             )
#
#             mu.tl.mofa(
#                 mtmp,
#                 n_factors=n_factors,
#                 likelihoods=likelihood,
#                 gpu_mode=gpu_mode,
#                 seed=global_seed,
#                 outfile=tmp_outfile,
#                 save_interrupted=False,
#                 save_data=False,
#                 save_metadata=False,
#                 save_parameters=False,
#             )
#
#             F = mtmp.obsm["X_mofa"]
#             F_std = StandardScaler().fit_transform(F)
#
#         except Exception as e:
#             print(f"[MOFA] Error fitting n_factors={n_factors}: {e}")
#             continue
#
#         for n_clusters in n_clusters_list:
#             try:
#                 km = KMeans(
#                     n_clusters=n_clusters,
#                     n_init=50,
#                     random_state=global_seed,
#                 )
#                 labels = km.fit_predict(F_std)
#
#                 # Effective number of clusters (just in case)
#                 n_eff = len(np.unique(labels))
#                 if min_clusters is not None and n_eff <= min_clusters:
#                     continue
#
#                 # Silhouette in MOFA space
#                 sil = silhouette_score(F_std, labels)
#
#                 if sil > best_score:
#                     best_score = sil
#                     best_params = {
#                         "n_factors": n_factors,
#                         "n_clusters": n_clusters,
#                         "silhouette_score": float(sil),
#                     }
#             except Exception as e:
#                 print(f"[MOFA+kmeans] Error n_factors={n_factors}, n_clusters={n_clusters}: {e}")
#                 continue
#
#     return best_params


def optimize_mofa_kmeans_hyperparams(
    obj,
    *,
    min_clusters=None,
    n_factors_range=(5, 20, 5),    # (start, end, step)
    n_clusters_range=(5, 20, 1),   # (start, end, step)
    total_var_explained_threshold_fraction=0.95,
    likelihood="gaussian",
    gpu_mode=True,
    global_seed=137,
):

    """
    Two-stage hyperparameter optimization for MOFA + k-means clustering.

    Stage 1 (MOFA dimensionality):
        Fit MOFA for each candidate n_factors.
        For each model, compute the total variance explained (summed over all
        modalities and factors). Select the smallest n_factors that achieves at
        least `total_var_explained_threshold_fraction` of the maximum variance
        explained observed across the sweep.

    Stage 2 (k-means in MOFA space):
        Using the MOFA model corresponding to the selected n_factors, extract
        the latent factors, standardize them, sweep over a range of k values,
        and select the k that maximizes the silhouette score.

    Parameters
    ----------
    obj : AnnData or MuData
        If AnnData, it is treated as a single-view MOFA model.
        If MuData, each modality is treated as a separate MOFA view.
    min_clusters : int or None
        Require at least this many effective clusters from k-means; solutions
        producing fewer are discarded.
    n_factors_range : (int, int, int)
        (start, end, step) for candidate MOFA factor counts.
    n_clusters_range : (int, int, int)
        (start, end, step) for candidate k-means cluster counts.
    total_var_explained_threshold_fraction : float
        Fraction of the maximum total variance explained required to accept a
        candidate n_factors (e.g., 0.95 selects the smallest model whose total
        variance explained is at least 95% of the best observed).
    likelihood : str
        MOFA likelihood for all views (e.g., "gaussian").
    gpu_mode : bool
        Whether to request GPU mode for MOFA (falls back to CPU if unavailable).
    global_seed : int
        Random seed used for MOFA and k-means.

    Returns
    -------
    best_params : dict or None
        {
            "n_factors": int,
            "n_clusters": int,
            "silhouette_score": float,
        }
        or None if no valid solution was found (e.g., all fits failed or all
        k-means runs violated the min_clusters constraint).
    """

    if not isinstance(obj, (sc.AnnData, mu.MuData)):
        raise TypeError("optimize_mofa_kmeans_hyperparams expects AnnData or MuData.")

    # Local helpers
    def _get_total_variance_explained(mtmp):
        """
        Sum of variance explained over all modalities and factors.
        Expects mtmp.uns['mofa']['variance'] to be a dict:
            {modality: 1D array (n_factors,)}
        """
        try:
            var_dict = mtmp.uns["mofa"]["variance"]
        except KeyError:
            raise KeyError("mtmp.uns['mofa']['variance'] not found; cannot compute variance explained.")
        return sum(np.sum(v) for v in var_dict.values())


    # Make sure there are no NaNs in X
    if isinstance(obj, sc.AnnData):
        ad_copy = obj.copy()
        ad_copy.X = np.nan_to_num(ad_copy.X, nan=np.nanmean(ad_copy.X, axis=0))
        views = {"view": ad_copy}

        # Compute max factors data supports (min of cells and features)
        max_factors = int(min(obj.n_vars, obj.n_obs))

    elif isinstance(obj, mu.MuData):
        # Use all modalities as MOFA views
        views = {}
        for mod_key, mod_obj in obj.mod.items():
            mod_copy = mod_obj.copy()
            mod_copy.X = np.nan_to_num(mod_copy.X, nan=np.nanmean(mod_copy.X, axis=0))
            views[mod_key] = mod_copy

        # Computing max factors the data supports (min of cells and features of all modalities)
        min_vars = min(mod_obj.n_vars for mod_obj in obj.mod.values())
        min_obs = min(mod_obj.n_obs for mod_obj in obj.mod.values())
        max_factors = int(min(min_vars, min_obs))

    # Prebuild ranges
    n_factors_raw = list(range(n_factors_range[0],
                               n_factors_range[1] + 1,
                               n_factors_range[2]))

    # keep only n_factors <= max_factors
    n_factors_list = [k for k in n_factors_raw if k <= max_factors]

    if not n_factors_list:
        warnings.warn(
            f"All requested n_factors ({n_factors_raw}) exceed max_factors={max_factors}; "
            "no MOFA fits attempted.",
            UserWarning,
        )
        return None

    n_clusters_list = list(range(n_clusters_range[0], n_clusters_range[1] + 1, n_clusters_range[2]))

    # Stage-1 bookkeeping: variance per n_factors and fitted models
    n_factors_variance = []  # list of dicts: {"n_factors": ..., "total_variance": ...}
    mofa_models_by_n_factors = {}  # n_factors -> fitted MuData

    # --------- STAGE 1: OPTIMIZE n_factors. Sweep n_factors, fit MOFA, store total variance explained
    for n_factor in n_factors_list:
        try:
            mtmp = mu.MuData(views)

            # Unique outfile per process to avoid HDF5 lock clashes (in case of multiprocessing, though not implemented)
            tmp_outfile = os.path.join(
                tempfile.gettempdir(),
                f"mofa_{os.getpid()}.hdf5",
            )

            mu.tl.mofa(
                mtmp,
                n_factors=n_factor,
                likelihoods=likelihood,
                gpu_mode=gpu_mode,
                seed=global_seed,
                outfile=tmp_outfile,
                save_interrupted=False,
                save_data=False,
                save_metadata=False,
                save_parameters=False,
            )

            F = mtmp.obsm["X_mofa"]
            F_std = StandardScaler().fit_transform(F)

            # Stage-1: total variance explained at this n_factor
            total_var = _get_total_variance_explained(mtmp)
            n_factors_variance.append(
                {"n_factors": n_factor, "total_variance": float(total_var)}
            )
            mofa_models_by_n_factors[n_factor] = mtmp

        except Exception as e:
            print(f"[MOFA] Error fitting n_factors={n_factor}: {e}")
            continue

    # --------- Choose best n_factors by total variance explained
    n_factors_variance_sorted = sorted(n_factors_variance, key=lambda d: d["n_factors"])
    tvs = np.array([d["total_variance"] for d in n_factors_variance_sorted])
    nfs = np.array([d["n_factors"] for d in n_factors_variance_sorted])

    tv_max = tvs.max()

    mask = tvs >= total_var_explained_threshold_fraction * tv_max
    if mask.any():
        best_n_factors = int(nfs[mask][0])  # smallest nf within threshold of max TVE
    else:
        best_n_factors = int(nfs[tvs.argmax()])  # fallback: nf with max TVE

    # Get the corresponding fitted MOFA model and embedding
    mtmp_best = mofa_models_by_n_factors[best_n_factors]
    F_best = mtmp_best.obsm["X_mofa"]
    F_best_std = StandardScaler().fit_transform(F_best)

    # ---------- STAGE 2: OPTIMIZE K. Sweep k with silhouette score as an objective in MOFA space
    best_score = -np.inf
    best_params = None

    for n_clusters in n_clusters_list:
        try:
            km = KMeans(
                n_clusters=n_clusters,
                n_init=50,
                random_state=global_seed,
            )
            labels = km.fit_predict(F_best_std)

            # Effective number of clusters (just in case)
            n_eff = len(np.unique(labels))
            if min_clusters is not None and n_eff <= min_clusters:
                continue

            # Need at least 2 clusters for silhouette
            if len(np.unique(labels)) < 2:
                continue

            sil = silhouette_score(F_best_std, labels)

            if sil > best_score:
                best_score = sil
                best_params = {
                    "n_factors": best_n_factors,
                    "n_clusters": n_clusters,
                    "silhouette_score": float(sil),
                }
        except Exception as e:
            print(f"[MOFA+kmeans] Error n_factors={best_n_factors}, n_clusters={n_clusters}: {e}")
            continue

    return best_params


def optimize_clustering_for_object(ad_key, ad_obj, n_neighbors_range, resolution_range, metric_weights,
                                   ground_truth_clustering_df):
    """
    This is used as a worker function, for a description see calling function: optimize_clustering()
    """
    try:
        best_score = -np.inf
        best_params = None

        # Loop through each combination of parameters
        for n_neighbors in range(n_neighbors_range[0], n_neighbors_range[1] + 1, n_neighbors_range[2]):
            for resolution in np.arange(resolution_range[0], resolution_range[1] + resolution_range[2],
                                        resolution_range[2]):
                # Run clustering
                run_clustering(
                    AnnData_object_dictionary={ad_key: ad_obj},
                    n_neighbors=n_neighbors,
                    resolution=resolution
                )

                # Calculate clustering metrics
                clustering_metrics = calculate_clustering_metrics(ad_obj, ground_truth_clustering_df)
                cluster_correlations = calculate_cluster_correlations(ground_truth_clustering_df, ad_obj)

                # Penalize Jaccard score if no match
                jaccard_scores = [v['jaccard_score'] for v in cluster_correlations['detailed_results'].values()]
                jaccard_average = np.mean(jaccard_scores) if jaccard_scores else 0

                # Calculate weighted average score
                total_score = (
                        metric_weights['silhouette'] * clustering_metrics['silhouette_score'] +
                        metric_weights['adjusted_mutual_info'] * clustering_metrics['adjusted_mutual_info'] +
                        metric_weights['homogeneity'] * clustering_metrics['homogeneity_score'] +
                        metric_weights['jaccard'] * jaccard_average +
                        metric_weights['specificity'] * np.mean(
                    [v['specificity'] for v in cluster_correlations['detailed_results'].values()])
                )

                # Check if this is the best score
                if total_score > best_score:
                    best_score = total_score
                    best_params = {
                        'n_neighbors': n_neighbors,
                        'resolution': resolution,
                        'silhouette_score': clustering_metrics['silhouette_score'],
                        'adjusted_mutual_info': clustering_metrics['adjusted_mutual_info'],
                        'homogeneity': clustering_metrics['homogeneity_score'],
                        'jaccard_average': jaccard_average,
                        'specificity_average': np.mean(
                            [v['specificity'] for v in cluster_correlations['detailed_results'].values()]),
                        'total_score': total_score
                    }

        # Add the AnnData key to the best_params dictionary
        if best_params is None:
            best_params = {'AnnData_object': ad_key, 'n_neighbors': np.nan, 'resolution': np.nan, 'total_score': np.nan}
        else:
            best_params['AnnData_object'] = ad_key

        return best_params

    except Exception as e:
        # In case of failure, return NaN for the parameters and log the error message
        print(f"Error optimizing AnnData object {ad_key}: {e}")
        return {'AnnData_object': ad_key, 'n_neighbors': np.nan, 'resolution': np.nan, 'total_score': np.nan}


def optimize_clustering(
        AnnData_object_dictionary,
        n_neighbors_range=(10, 50, 5),
        resolution_range=(0.2, 1.0, 0.1),
        metric_weights=None,
        ground_truth_clustering_df=None,
        csv_export_path=None,
        n_jobs=4,
        use_threads=False  # Enable to use ThreadPoolExecutor instead of ProcessPoolExecutor
):
    """
    Optimizes clustering parameters (n_neighbors, resolution) for each AnnData object in the dictionary
    using either ProcessPoolExecutor or ThreadPoolExecutor and returns the best parameter combination.

    Parameters:
        AnnData_object_dictionary (dict): Dictionary containing AnnData objects.
        n_neighbors_range (tuple): Range and increment for n_neighbors (start, end, increment).
        resolution_range (tuple): Range and increment for Leiden resolution (start, end, increment).
        metric_weights (dict, optional): Weights for each metric (silhouette, adjusted_mutual_info, homogeneity, jaccard, specificity).
                                         Default is equal weights.
        ground_truth_clustering_df (pd.DataFrame, optional): DataFrame containing the ground truth cell type labels.
        csv_export_path (str, optional): File path to export the results as a CSV. If None, CSV will not be exported.
        n_jobs (int, optional): Number of CPU cores or threads to use for parallel processing. Default is 4.
        use_threads (bool, optional): If True, uses ThreadPoolExecutor instead of ProcessPoolExecutor. Default is False.
            This is implemented as if a ProcessPoolExecutor is used in a cell that is further down in a Jupyter notebook,
            it will fail due to a queue block (reason uncertain, as memory remains abundant).

    Returns:
        pd.DataFrame: DataFrame containing the best parameter combination for each AnnData object.
    """
    if metric_weights is None:
        # Default to equal weights if not provided
        metric_weights = {
            'silhouette': 1.0,
            'adjusted_mutual_info': 1.0,
            'homogeneity': 1.0,
            'jaccard': 1.0,
            'specificity': 1.0
        }

    # Normalize weights to sum up to 1
    total_weight = sum(metric_weights.values())
    metric_weights = {k: v / total_weight for k, v in metric_weights.items()}

    results = []

    # Use either ProcessPoolExecutor or ThreadPoolExecutor based on the use_threads argument
    executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor

    with executor_class(max_workers=n_jobs) as executor:
        futures = [
            executor.submit(
                optimize_clustering_for_object,
                ad_key, ad_obj, n_neighbors_range, resolution_range, metric_weights, ground_truth_clustering_df
            )
            for ad_key, ad_obj in AnnData_object_dictionary.items()
        ]

        for future in futures:
            results.append(future.result())

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Export to CSV if requested
    if csv_export_path:
        results_df.to_csv(csv_export_path, index=False)

    return results_df

def get_mofa_modality_importance(mdata):
    """
    mdata: MuData after mu.tl.mofa(...)
           must have mdata.uns["mofa"]["variance"] as a dict:
           {modality -> 1D array (n_factors,)}
    Returns:
      ve_df: variance explained per factor per modality
      contrib_per_factor: row-normalized (per-factor) contributions
      overall_importance: single scalar importance per modality
    """
    var = mdata.uns["mofa"]["variance"]    # dict: mod -> np.ndarray (n_factors,)
    modalities = list(var.keys())
    n_factors = var[modalities[0]].shape[0]
    factor_idx = [f"{i+1}" for i in range(n_factors)]

    # Absolute variance explained per factor per modality
    ve_df = pd.DataFrame({mod: var[mod] for mod in modalities},
                         index=factor_idx)

    # Per-factor fractional contribution (rows sum to 1)
    contrib_per_factor = ve_df.div(ve_df.sum(axis=1), axis=0)

    # Overall importance: total variance per modality, normalized
    total_var = ve_df.sum(axis=0)
    overall_importance = (total_var / total_var.sum()).sort_values(ascending=False)

    return ve_df, contrib_per_factor, overall_importance


def plot_mofa_variance_decomposition(contrib_per_factor, title=None):
    plt.figure(figsize=(3, 6))
    sns.heatmap(
        contrib_per_factor,
        cmap="Purples",
        annot=False,
        cbar=True,
        vmin=0,
        vmax=1
    )
    plt.xlabel("Feature Space")
    plt.ylabel("Factor")
    if title is not None:
        plt.title(title)
    plt.tight_layout()



def rf_cv_hyperparam_sweep(
        feature_space_dict,
        y,
        *,
        n_estimators_list=(300, 600, 1000),
        max_depth_list=(None, 20, 40),
        max_features_list=("sqrt", "log2", 0.8),
        n_splits=5,
        class_weight="balanced",
        global_seed=137,
):
    """
    Sweep RF hyperparameters over multiple feature spaces.

    Parameters
    ----------
    feature_space_dict : dict[str, AnnData]
        Keys are feature space names, values are AnnData objects with .X as features.
    y : array-like, shape (n_samples,)
        True labels, aligned with rows of each AnnData in feature_space_dict.
    n_estimators_list, max_depth_list, max_features_list : iterable
        Hyperparameter values to sweep.
    n_splits : int
        Number of CV folds.
    class_weight : str or dict
        Passed to RandomForestClassifier.
    global_seed : int
        Seed for CV splitting and RF.

    Returns
    -------
    overall_df : pd.DataFrame
        Columns:
        ['Feature Space', 'n_estimators', 'max_depth', 'max_features',
         'Mean AUC', 'AUC Std Dev', 'Mean Balanced Accuracy',
         'Balanced Accuracy Std Dev']
    per_label_df : pd.DataFrame
        Columns:
        ['Feature Space', 'Label', 'n_estimators', 'max_depth', 'max_features',
         'Mean AUC', 'Mean Balanced Accuracy']
    """
    y = np.asarray(y)
    label_names = np.unique(y)

    overall_rows = []
    per_label_rows = []

    # Fixed CV object
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=global_seed)

    # Hyperparameter grid
    hyperparam_grid = list(product(n_estimators_list, max_depth_list, max_features_list))

    for feat_name, adata in feature_space_dict.items():
        print(f"Processing feature space: {feat_name}")

        # X: features, standardized once per feature space
        X = np.nan_to_num(adata.X, nan=np.nanmean(adata.X, axis=0))
        X_std = StandardScaler().fit_transform(X)

        # Pre-compute splits once so they are identical across hyperparams
        splits = list(cv.split(X_std, y))

        for n_estimators, max_depth, max_features in hyperparam_grid:
            # Initialize classifier with current hyperparams
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_features=max_features,
                class_weight=class_weight,
                random_state=global_seed,
                n_jobs=-1,
            )

            # Per-label storage for this (feat, hyperparam) combo
            per_label_auc = {lbl: [] for lbl in label_names}
            per_label_bal = {lbl: [] for lbl in label_names}

            # Overall macro metrics
            auc_scores = []
            bal_scores = []

            # CV loop
            for train_idx, test_idx in splits:
                model.fit(X_std[train_idx], y[train_idx])
                y_prob = model.predict_proba(X_std[test_idx])
                y_pred = model.predict(X_std[test_idx])

                # Overall multi-class metrics
                auc_scores.append(
                    roc_auc_score(y[test_idx], y_prob, multi_class="ovr")
                )
                bal_scores.append(
                    balanced_accuracy_score(y[test_idx], y_pred)
                )

                # Per-label metrics (one-vs-rest)
                for i, lbl in enumerate(label_names):
                    label_mask = (y[test_idx] == lbl)
                    if np.sum(label_mask) > 1:
                        per_label_auc[lbl].append(
                            roc_auc_score(label_mask, y_prob[:, i])
                        )
                        per_label_bal[lbl].append(
                            balanced_accuracy_score(label_mask, (y_pred == lbl))
                        )

            # Store overall results for this hyperparam combo
            overall_rows.append({
                "Feature Space": feat_name,
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "max_features": max_features,
                "Mean AUC": np.mean(auc_scores),
                "AUC Std Dev": np.std(auc_scores),
                "Mean Balanced Accuracy": np.mean(bal_scores),
                "Balanced Accuracy Std Dev": np.std(bal_scores),
            })

            # Store per-label averaged results for this combo
            for lbl in label_names:
                lbl_auc_vals = per_label_auc[lbl]
                lbl_bal_vals = per_label_bal[lbl]

                per_label_rows.append({
                    "Feature Space": feat_name,
                    "Label": lbl,
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "max_features": max_features,
                    "Mean AUC": np.mean(lbl_auc_vals) if len(lbl_auc_vals) > 0 else np.nan,
                    "Mean Balanced Accuracy": np.mean(lbl_bal_vals) if len(lbl_bal_vals) > 0 else np.nan,
                })

    overall_df = pd.DataFrame(overall_rows)
    per_label_df = pd.DataFrame(per_label_rows)

    return overall_df, per_label_df
