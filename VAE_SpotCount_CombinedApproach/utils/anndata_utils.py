# This script contains functions for creating, modifying, and merging AnnData objects

import pandas as pd
import scanpy as sc
import anndata
import numpy as np




def create_anndata_obj(dataframe_dictionary, cell_metadata_df, dataframe_key_list=None, indices_to_concatenate=None):
    """
    Creates individual AnnData objects from a dictionary of DataFrames and optionally concatenates them based on the specified indices.

    Parameters:
        dataframe_dictionary (dict): A dictionary of DataFrames, each representing different data modalities.
        cell_metadata_df (pd.DataFrame): DataFrame containing cell metadata. Must have indices that match the indices of the DataFrames in dataframe_dictionary.
        dataframe_key_list (list, optional): List of keys from dataframe_dictionary to generate individual AnnData objects. If None, uses all keys in dataframe_dictionary.
        indices_to_concatenate (list of lists, optional): List of lists of indices specifying which AnnData objects to concatenate. If None, no concatenation is performed.

    Returns:
        dict: A dictionary of AnnData objects. Contains individual AnnData objects and optionally combined ones.
    """
    # Checking proper input
    if not dataframe_dictionary or not isinstance(dataframe_dictionary, dict):
        raise ValueError("dataframe_dictionary must be a non-empty dictionary of DataFrames.")
    if cell_metadata_df is None or not isinstance(cell_metadata_df, pd.DataFrame):
        raise ValueError("cell_metadata_df must be a non-empty DataFrame.")
    if dataframe_key_list is not None and not isinstance(dataframe_key_list, list):
        raise ValueError("dataframe_key_list must be a list of DataFrame keys.")

    # Defaulting to all keys if no list is provided
    if dataframe_key_list is None:
        dataframe_key_list = list(dataframe_dictionary.keys())

    if not all(keyStr in dataframe_dictionary for keyStr in dataframe_key_list):
        raise ValueError("All strings in dataframe_key_list must correspond to keys in dataframe_dictionary.")

    # Checking if cell_metadata_df indices match the DataFrame indices
    for keyStr in dataframe_key_list:
        if not cell_metadata_df.index.equals(dataframe_dictionary[keyStr].index):
            raise ValueError(
                f"Index mismatch between cell_metadata_df and DataFrame '{keyStr}' in dataframe_dictionary.")

    if indices_to_concatenate is not None and (
            not isinstance(indices_to_concatenate, list) or
            not all(isinstance(lst, list) and all(isinstance(i, int) and i >= 0 for i in lst) for lst in
                    indices_to_concatenate)
    ):
        raise ValueError("indices_to_concatenate must be a list of lists of non-negative integers if provided.")

    # Initializing dict of AnnData objects
    ad_obj_dict = {}

    # Generating AnnData objects
    for keyStr in dataframe_key_list:
        # Generating the metadata variable for the AnnData object
        metadataDF = pd.DataFrame(index=dataframe_dictionary[keyStr].columns.tolist())

        # Generating AnnData object
        ad_obj_dict[keyStr] = sc.AnnData(X=dataframe_dictionary[keyStr].values, obs=cell_metadata_df, var=metadataDF)

    # Generating concatenated AnnData objects if indices_to_concatenate provided, and adding them to the dict
    if indices_to_concatenate is not None:
        for lst in indices_to_concatenate:
            # Get the corresponding keys for the indices in `lst`
            keys_to_combine = [dataframe_key_list[idx] for idx in lst]
            # Collecting objects to concatenate
            obj_to_concat = [ad_obj_dict[key] for key in keys_to_combine]
            # Concatenating the objects along the feature/column/1 axis
            concatenated_adObj = anndata.concat(obj_to_concat, join='inner', axis=1, label=None)
            # Constructing the new key for the concatenated DataFrame
            combined_key = "combined_" + "_".join(keys_to_combine)
            # Adding the concatenated AnnData object to `ad_obj_dict`
            ad_obj_dict[combined_key] = concatenated_adObj

    return ad_obj_dict


def run_clustering(AnnData_object_dictionary, pca_ncomps=None, NaN2mean=True, key_added="clusters", resolution=0.65, n_neighbors=15):
    """
    Conducts dimensionality reduction and clustering with UMAP and Leiden on AnnData Objects housed in a dictionary.

    Parameters:
        AnnData_object_dictionary (dict): a dictionary containing only AnnData objects.
        pca_ncomps (int, optional): number of principal components to use for PCA. If None, defaults to 50, or 1 - minimum dimension size of selected representation.
        NaN2mean (bool, optional): if True (default), imputes NaN values with column mean. Relevant for CellProfiler feature data.
        key_added (str, optional): The name of the key under which to add the clustering labels to the AnnData object. Default is "clusters".
        resolution (float, optional): The resolution parameter for the Leiden algorithm. Higher values lead to more clusters. Default is 0.65.
        n_neighbors (int, optional): The number of nearest neighbors to use in sc.pp.neighbors. Defaults to 15.

    Returns:
        None
    """
    # Checking proper input
    if not isinstance(AnnData_object_dictionary, dict):
        raise TypeError("AnnData_object_dictionary must be a dictionary.")
    if not all(isinstance(adObj, anndata.AnnData) for adObj in AnnData_object_dictionary.values()):
        raise ValueError("All values in AnnData_object_dictionary must be AnnData objects.")

    if pca_ncomps is not None:
        if not isinstance(pca_ncomps, int):
            raise TypeError("pca_ncomps must be an integer.")
        if pca_ncomps <= 0:
            raise ValueError("pca_ncomps must be a positive integer.")

    if not isinstance(NaN2mean, bool):
        raise TypeError("NaN2mean must be a boolean value.")

    if not isinstance(n_neighbors, int) or n_neighbors < 2:
        raise ValueError("n_neighbors must be an integer greater than 1.")

    # Iterating over AnnData objects
    for adObj in AnnData_object_dictionary.values():
        try:
            # Imputing NaNs with column mean
            if NaN2mean:
                adObj.X = np.nan_to_num(adObj.X, nan=np.nanmean(adObj.X, axis=0))

            # Determine the number of principal components for PCA
            ncomps = pca_ncomps if pca_ncomps is not None else min(50, adObj.shape[1] - 1)

            # # Perform PCA, currently commented out as neighbors will run PCA automatically
            # sc.pp.pca(adObj, n_comps=ncomps)

            # Compute neighbors
            sc.pp.neighbors(adObj, n_neighbors=n_neighbors)

            # Compute UMAP
            sc.tl.umap(adObj)

            # Perform Leiden clustering
            sc.tl.leiden(adObj, key_added=key_added, resolution=resolution)

        except Exception as e:
            raise RuntimeError(f"An error occurred while clustering AnnData object: {e}")


def clear_clustering_data(AnnData_object_dictionary, key_added="clusters"):
    """
    Clears clustering-related data from all AnnData objects in a dictionary to allow rerunning clustering.

    Parameters:
        AnnData_object_dictionary (dict): A dictionary containing AnnData objects.
        key_added (str, optional): The key used for storing clustering labels in .obs. Default is "clusters".

    Returns:
        None
    """
    for key, AnnData_object in AnnData_object_dictionary.items():
        # Remove clustering labels
        if key_added in AnnData_object.obs:
            del AnnData_object.obs[key_added]
        if f"{key_added}_colors" in AnnData_object.uns:
            del AnnData_object.uns[f"{key_added}_colors"]

        # Remove PCA-related data
        if 'X_pca' in AnnData_object.obsm:
            del AnnData_object.obsm['X_pca']
        if 'pca' in AnnData_object.uns:
            del AnnData_object.uns['pca']
        if 'PCs' in AnnData_object.varm:
            del AnnData_object.varm['PCs']

        # Remove neighbors and UMAP-related data
        if 'neighbors' in AnnData_object.uns:
            del AnnData_object.uns['neighbors']
        if 'umap' in AnnData_object.uns:
            del AnnData_object.uns['umap']
        if 'X_umap' in AnnData_object.obsm:
            del AnnData_object.obsm['X_umap']
        if 'connectivities' in AnnData_object.obsp:
            del AnnData_object.obsp['connectivities']
        if 'distances' in AnnData_object.obsp:
            del AnnData_object.obsp['distances']

        # Remove spatial-related data
        if 'spatial_neighbors' in AnnData_object.uns:
            del AnnData_object.uns['spatial_neighbors']
        if 'spatial_connectivities' in AnnData_object.obsp:
            del AnnData_object.obsp['spatial_connectivities']
        if 'spatial_distances' in AnnData_object.obsp:
            del AnnData_object.obsp['spatial_distances']

        # Remove clustering evaluation results
        if f"{key_added}_interactions" in AnnData_object.uns:
            del AnnData_object.uns[f"{key_added}_interactions"]
        if f"{key_added}_nhood_enrichment" in AnnData_object.uns:
            del AnnData_object.uns[f"{key_added}_nhood_enrichment"]
        if f"{key_added}_co_occurrence" in AnnData_object.uns:
            del AnnData_object.uns[f"{key_added}_co_occurrence"]
        if 'rank_genes_groups' in AnnData_object.uns:
            del AnnData_object.uns['rank_genes_groups']


def top_n_genes_per_cluster(AnnData_object, n_genes, clustering_obs_key):
    """
    Extract the top n genes for each cluster from an AnnData object using the Wilcoxon rank-sum method.

    Parameters:
    - AnnData_object (AnnData): An AnnData object that contains clustering information in `.obs['clusters']`.
    - n_genes (int): Number of top genes to extract for each cluster.
    - clustering_obs_key (str): Key used to store clustering information in the AnnData_object.obs.

    Returns:
    - dict: A dictionary where keys are cluster labels (as strings) and values are lists of top-ranked genes.
    """

    # Input checks
    if not isinstance(AnnData_object, sc.AnnData):
        raise TypeError("The input must be an AnnData object.")

    if clustering_obs_key not in AnnData_object.obs:
        raise ValueError("The AnnData object must contain a 'clusters' column in .obs.")

    if not isinstance(n_genes, int) or n_genes <= 0:
        raise ValueError("n_genes must be a positive integer.")

    # Initialize dictionary
    top_genes_dict = {}

    # Get unique cluster labels and convert to a list of str
    unique_clusters = np.unique(AnnData_object.obs[clustering_obs_key])
    cluster_list_str = [str(cluster) for cluster in unique_clusters]

    # Run gene ranking
    sc.tl.rank_genes_groups(AnnData_object, groupby=clustering_obs_key, method='wilcoxon', n_genes=n_genes)

    # Extract top gene per cluster
    for cluster in cluster_list_str:
        top_genes_dict[cluster] = list(AnnData_object.uns['rank_genes_groups']['names'][cluster])

    return top_genes_dict


def filter_features(AnnData_object, features_to_drop):
    """
    Filters an AnnData object to remove specified features (genes or otherwise).

    Parameters:
    - AnnData_object (AnnData): The AnnData object to filter.
    - features_to_drop (list): A list of feature names to remove.

    Returns:
    - AnnData: A filtered AnnData object with the specified features removed.
    """

    # Input checks
    if not isinstance(AnnData_object, sc.AnnData):
        raise TypeError("The input must be an AnnData object.")
    if not isinstance(features_to_drop, list):
        raise TypeError("features_to_drop must be a list.")

    # Identify features to keep (those not in features_to_drop)
    features_to_keep = [feature for feature in AnnData_object.var_names if feature not in features_to_drop]

    # Filter features to keep
    filtered_AnnData_obj = AnnData_object[:, features_to_keep].copy()

    return filtered_AnnData_obj


def compare_anndata(adata1: anndata.AnnData, adata2: anndata.AnnData):
    """
    Compares two AnnData objects to see if they are identical. For sanity/debugging purposes.
    """
    # Compare the main data matrices
    if not np.array_equal(adata1.X, adata2.X):
        print("The .X matrices are different.")
        return False

    # Compare observation annotations (metadata)
    if not adata1.obs.equals(adata2.obs):
        print("The .obs DataFrames are different.")
        return False

    # Compare variable annotations (metadata)
    if not adata1.var.equals(adata2.var):
        print("The .var DataFrames are different.")
        return False

    # Compare layers
    if adata1.layers.keys() != adata2.layers.keys():
        print("The layers are different.")
        return False
    for layer in adata1.layers:
        if not np.array_equal(adata1.layers[layer], adata2.layers[layer]):
            print(f"The layer '{layer}' is different.")
            return False

    # Compare .obsm (multi-dimensional annotations for observations)
    if adata1.obsm.keys() != adata2.obsm.keys():
        print("The .obsm keys are different.")
        return False
    for key in adata1.obsm:
        if not np.array_equal(adata1.obsm[key], adata2.obsm[key]):
            print(f"The .obsm entry '{key}' is different.")
            return False

    # Compare .varm (multi-dimensional annotations for variables)
    if adata1.varm.keys() != adata2.varm.keys():
        print("The .varm keys are different.")
        return False
    for key in adata1.varm:
        if not np.array_equal(adata1.varm[key], adata2.varm[key]):
            print(f"The .varm entry '{key}' is different.")
            return False

    # Compare unstructured annotations
    if adata1.uns.keys() != adata2.uns.keys():
        print("The .uns keys are different.")
        return False
    for key in adata1.uns:
        if adata1.uns[key] != adata2.uns[key]:
            print(f"The .uns entry '{key}' is different.")
            return False

    print("The two AnnData objects are identical.")
    return True