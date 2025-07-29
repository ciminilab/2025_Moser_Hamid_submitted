# This script contains functions to preprocess the dataframes from different modalities (gene count, embeddings, metadata, ground truth, cell profiler features)

# Imports
import os
import pandas as pd
import pycytominer as pcm
import scanpy as sc


def summarize_index_intersection(dataframe_dictionary, output_directory_path, exported_csv_name, export=True):
    """
    Summarizes the index intersection between a list of DataFrames and optionally exports the matrix as a CSV.
        This is useful for giving a general idea about the amount of cells being dropped in downstream steps due to non-overlap.

    Parameters:
        dataframe_dictionary (dict): dictionary of DataFrames to summarize.
        output_directory_path (str): Directory to save the output CSV file, can be relative (in cwd) or absolute.
        exported_csv_name (str, optional): Name of the exported CSV file. Defaults to "cell_index_intersection_matrix.csv".
        export (bool): Enable export of the intersection matrix as a CSV (this df will be returned anyway)

    Returns:
        pd.DataFrame: DF representing the index intersection matrix
    """
    # Check if provided df dict is a dict
    if not dataframe_dictionary or not isinstance(dataframe_dictionary, dict):
        raise ValueError("dataframe_list must be a non-empty dictionary of DataFrames.")


    # Create the intersection matrix
    df_names = list(dataframe_dictionary.keys())
    index_intersection_matrix = pd.DataFrame(index=df_names, columns=df_names)

    # Checking intersection between input DFs and populating intersection DF accordingly
    for name1 in df_names:
        for name2 in df_names:
            index_list1 = set(dataframe_dictionary[name1].index)
            index_list2 = set(dataframe_dictionary[name2].index)
            intersection = len(index_list1.intersection(index_list2))
            index_intersection_matrix.loc[name1, name2] = intersection

    # If export is requested
    if export:
        if not output_directory_path:
            raise ValueError("Error: export requested but no output directory provided.")

        if not isinstance(output_directory_path, str):
            raise ValueError("Error: output_directory_path must be a string")

        # Converting path to absolute if relative
        if not os.path.isabs(output_directory_path):
            output_directory_path = os.path.abspath(output_directory_path)

        # Creating directory if it does not exist
        os.makedirs(output_directory_path, exist_ok=True)

        # Handle the exported CSV name
        if exported_csv_name is None:
            export_name = "cell_index_intersection_matrix.csv"
        elif not isinstance(exported_csv_name, str):
            raise ValueError("Error: exported_csv_name must be a string")
        else:
            export_name = exported_csv_name if exported_csv_name.endswith(".csv") else exported_csv_name + ".csv"

        # Export
        index_intersection_matrix.to_csv(os.path.join(output_directory_path, export_name), index=True)

    return index_intersection_matrix


def merge_df_unq_cols_by_val(df1, df2, suffixes):
    """
    Merges two pd.DataFrames such that only unique columns by value. If columns have the same name, but different values, they are simply suffixed and both retained.

    Parameters:
        df1 (pd.DataFrame): Required. First DataFrame
        df2 (pd.DataFrame): Required. Second DataFrame
        suffixes (tuple): Required. List of suffixes to merge df1 with df2.

    Returns:
        pd.DataFrame: Merged DataFrame
        identical_columns: a list of tuples of the pairs of identical columns that were removed
    """

    if any(arg is None for arg in [df1, df2, suffixes]):
        raise ValueError("One or both DataFrames must have at least one column. Suffixes are also required.")

    merged_df = pd.merge(df1, df2, how='outer', left_index=True, right_index=True, suffixes=suffixes)

    # Initialize set to be populated with non-unique columns and a list to store identical column pairs
    columns_to_drop = set()
    identical_columns = []
    merged_columns = list(merged_df.columns)

    # Iterate to check for redundant columns
    for i, col1 in enumerate(merged_columns):
        if col1 in columns_to_drop:
            continue

        # Compare col1 with the remaining columns
        for col2 in merged_columns[i + 1:]:
            if col2 in columns_to_drop:
                continue

            # Check if columns have identical values
            if merged_df[col1].equals(merged_df[col2]):
                # Track columns to drop and store the identical column pair
                columns_to_drop.add(col2)
                identical_columns.append((col1, col2))

    # Drop redundant columns
    merged_df.drop(columns=list(columns_to_drop), inplace=True)

    return merged_df, identical_columns


def filter_to_common_indices(dataframe_dictionary):
    """
    Filters a dictionary of DataFrames to keep only the common indices across all DataFrames.

    Parameters:
        dataframe_dictionary (dict): A dictionary where keys are strings and values are pandas DataFrames.

    Returns:
        tuple: A tuple containing:
            - dataframe_dictionary_updated (dict): Dictionary with DataFrames filtered to common indices.
            - common_indices (set): The set of indices common across all DataFrames.
    """
    # Checking proper input
    if not dataframe_dictionary or not isinstance(dataframe_dictionary, dict):
        raise ValueError("dataframe_dictionary must be a non-empty dictionary of DataFrames.")

    # Get the indices of first iterated df
    common_indices = set(next(iter(dataframe_dictionary.values())).index)

    # Intersect with the indices of all other DFs
    for df in list(dataframe_dictionary.values())[1:]:
        common_indices.intersection_update(df.index)

    # Initializing dict with updated DFs
    dataframe_dictionary_updated = {}
    for key, df in dataframe_dictionary.items():
        dataframe_dictionary_updated[key] = df[df.index.isin(common_indices)]

    return dataframe_dictionary_updated, common_indices


def drop_irrelevant_cols(dataframe, col_name_kw_list):
    """
    Drops columns from a DataFrame based on keywords in column names and non-NaN string values.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame to filter.
        col_name_kw_list (list): List of keywords to identify columns for dropping.

    Returns:
        tuple: A tuple containing:
            - df (pd.DataFrame): The filtered DataFrame after dropping columns.
            - colsToDrop (list): List of columns that were dropped.
    """
    # Generating list of columns with names containing certain KWs
    colsWithKWs = [col for col in dataframe.columns if any(kw.lower() in col.lower() for kw in col_name_kw_list)]
    # Generating columns with non-NaN str values
    colsWithStr = [col for col in dataframe.columns if
                   dataframe[col].apply(lambda x: isinstance(x, str) and pd.notna(x)).any()]
    # Dropping columns
    colsToDrop = colsWithKWs + colsWithStr
    df = dataframe.copy()
    df.drop(columns=colsToDrop, inplace=True)

    # Using PyCytominer for feature selection from the CP data
    current_feature_list = df.columns.tolist()
    # Selecting operations based on profiling recipe from github.com/cytomining/profiling-recipe
    df = pcm.feature_select(profiles=df, features=current_feature_list, image_features=True, operation=[
        "variance_threshold", "correlation_threshold", "drop_na_columns", "blocklist"])

    return df, colsToDrop


def adjust_latent_var_names(dataframe_dictionary):
    """
    Adjusts the names of latent variables in the DataFrames of the given dictionary.
    If a DataFrame key contains 'embedding', each column name is prefixed with 'z_'.

    Parameters:
        dataframe_dictionary (dict): A dictionary where keys are strings and values are pandas DataFrames.

    Returns:
        dict: A dictionary with updated DataFrames, where latent variable column names are modified if applicable.
    """
    # Checking proper input
    if not dataframe_dictionary or not isinstance(dataframe_dictionary, dict):
        raise ValueError("dataframe_dictionary must be a non-empty dictionary of DataFrames.")

    # Initializing updated dict
    dataframe_dictionary_updated = {}
    for key, df in dataframe_dictionary.items():
        # Checking if DF is that of an embedding
        if 'embedding' in key:
            df = df.rename(columns={col: 'z_' + col for col in df.columns})  # Fixed the renaming issue
        # Adding df to updated dictionary
        dataframe_dictionary_updated[key] = df

    return dataframe_dictionary_updated


# No usages, retained in case of a change in approach
def generate_metadata_dfs(dataframe_dictionary, dataframe_key_list=None, indices_to_concatenate=None):
    """
    Generates metadata DataFrames from the given dictionary of DataFrames and optionally concatenates them
    based on provided indices. This function is no longer used, as now the metadata is extracted directly
    from the dataframe_dictionary, but the function is retained in case.

    Parameters:
        dataframe_dictionary (dict): A dictionary where keys are strings and values are pandas DataFrames.
        dataframe_key_list (list, optional): List of DataFrame keys to generate metadata from. Defaults to all keys.
        indices_to_concatenate (list of lists, optional): List of lists containing indices to concatenate metadata DataFrames.

    Returns:
        dict: A dictionary containing metadata DataFrames, with keys including the original and concatenated DataFrames.
    """
    # Checking proper input
    if not dataframe_dictionary or not isinstance(dataframe_dictionary, dict):
        raise ValueError("dataframe_dictionary must be a non-empty dictionary of DataFrames.")
    if dataframe_key_list is not None and not isinstance(dataframe_key_list, list):
        raise ValueError("dataframe_key_list must be a list of DataFrame keys.")

    # Defaulting to all keys if no list is provided
    if dataframe_key_list is None:
        dataframe_key_list = list(dataframe_dictionary.keys())

    if not all(keyStr in dataframe_dictionary for keyStr in dataframe_key_list):
        raise ValueError("All strings in dataframe_key_list must correspond to keys in dataframe_dictionary.")
    if indices_to_concatenate is not None and (
            not isinstance(indices_to_concatenate, list) or
            not all(isinstance(lst, list) and all(isinstance(i, int) and i >= 0 for i in lst) for lst in indices_to_concatenate)
    ):
        raise ValueError("indices_to_concatenate must be a list of lists of non-negative integers if provided.")

    # Initializing dict of metadata DF
    metadata_df_dict = {}

    # Generating metadata (empty DataFrame with indices corresponding to column names) from selected DataFrames
    for keyStr in dataframe_key_list:
        metadataDF = pd.DataFrame(index=dataframe_dictionary[keyStr].columns.tolist())
        metadata_df_dict[keyStr] = metadataDF

    # Generating concatenated metadata DataFrames if requested, and adding them to the dict
    if indices_to_concatenate is not None:
        for lst in indices_to_concatenate:
            # Get the corresponding keys for the indices in `lst`
            keys_to_combine = [dataframe_key_list[idx] for idx in lst]
            # Collecting DataFrames to concatenate
            dfs_to_concat = [metadata_df_dict[key] for key in keys_to_combine]
            # Concatenating the DataFrames along the row axis
            concatenated_df = pd.concat(dfs_to_concat, axis=0)
            # Constructing the new key for the concatenated DataFrame
            combined_key = "combined_" + "_".join(keys_to_combine)
            # Adding the concatenated DataFrame to `metadata_df_dict`
            metadata_df_dict[combined_key] = concatenated_df

    return metadata_df_dict

# Used to test and eliminate genes correlated with any latent variables in an attempt to reduce noise before clustering.
# No current usage.
def identify_correlated_latent_vars(dataframe_tuple, r_sq_threshold):
    """
    Identify and exclude highly correlated latent variables from the second DataFrame based on a threshold for R².

    Parameters:
        dataframe_tuple (tuple): A tuple containing two DataFrames to compare. First is the reference.
        r_sq_threshold (float): The R² threshold for identifying highly correlated variables.

    Returns:
        list: A list of columns in the second DataFrame that are highly correlated with columns in the first DataFrame.
            NOTE: No significance testing is conducted.
    """
    if not isinstance(dataframe_tuple, tuple) or len(dataframe_tuple) != 2:
        raise ValueError("Input must be a tuple of two DataFrames.")
    if not isinstance(r_sq_threshold, float):
        raise ValueError("r_sq_threshold must be a float.")

    df1, df2 = dataframe_tuple

    # Initializing list of correlated latent vars
    correlated_latent_vars = []

    for col1 in df1.columns:
        for col2 in df2.columns:
            correl_coeff = df1[col1].corr(df2[col2])
            r_squared = correl_coeff ** 2
            if r_squared >= r_sq_threshold and col2 not in correlated_latent_vars:
                correlated_latent_vars.append(col2)

    return correlated_latent_vars

# Useful to implement QC on transcript count data using ScanPy's library but with a DF instead of an AnnData object.
def filter_cells_dataframe(
    df: pd.DataFrame,
    min_counts: int = None,
    min_genes: int = None,
    max_counts: int = None,
    max_genes: int = None
) -> pd.DataFrame:
    """
    Filters cells (rows) from a pandas DataFrame using scanpy's filter_cells function.

    Parameters
    ----------
    df : pd.DataFrame
        The data matrix of shape `n_cells` × `n_genes`. Rows correspond to cells and columns to genes.
    min_counts : int, optional
        Minimum number of counts required for a cell to pass filtering.
    min_genes : int, optional
        Minimum number of genes expressed required for a cell to pass filtering.
    max_counts : int, optional
        Maximum number of counts required for a cell to pass filtering.
    max_genes : int, optional
        Maximum number of genes expressed required for a cell to pass filtering.

    Returns
    -------
    pd.DataFrame
        The filtered DataFrame with cells that meet the specified threshold criteria.
    """
    # Convert the DataFrame to a NumPy array
    data_array = df.values

    # Call scanpy's filter_cells function
    cells_subset, _ = sc.pp.filter_cells(
        data_array,
        min_counts=min_counts,
        min_genes=min_genes,
        max_counts=max_counts,
        max_genes=max_genes,
        inplace=False
    )

    # Filter the DataFrame using the boolean mask
    filtered_df = df[cells_subset]

    return filtered_df




