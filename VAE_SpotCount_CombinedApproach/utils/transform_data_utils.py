import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from pycytominer.operations import RobustMAD



def norm_df_to_total(dataframe_dictionary, dataframe_key_list, target_sum=None):
    """
    Normalizes DataFrames (in this context, only gene counts) to the total gene count per cell,
    following the logic used by scanpy.pp.normalize_total.

    Parameters:
        dataframe_dictionary (dict): Dictionary containing DataFrames to select from for normalization.
        dataframe_key_list (list): List of DataFrame keys to fetch from the dictionary and normalize.
        target_sum (float or int, optional): Target sum to multiply with after normalization.
            Defaults to the median count, excluding zero total counts.

    Returns:
        dict: Dictionary with DataFrames normalized to the specified target sum.
    """
    # Checking proper input
    if not dataframe_dictionary or not isinstance(dataframe_dictionary, dict):
        raise ValueError("dataframe_dictionary must be a non-empty dictionary of DataFrames.")
    if not dataframe_key_list or not isinstance(dataframe_key_list, list):
        raise ValueError("dataframe_key_list must be a non-empty list of DataFrame keys.")
    if not all(keyStr in dataframe_dictionary for keyStr in dataframe_key_list):
        raise ValueError("All strings in dataframe_key_list must correspond to keys in dataframe_dictionary.")
    if not isinstance(target_sum, (float, int)) and target_sum is not None:
        raise ValueError("target_sum must be a float or int.")

    # Duplicating df dict
    dataframe_dictionary_updated = {key: df.copy() for key, df in dataframe_dictionary.items()}

    # Iterating over DataFrames selected by provided keys, from the provided dictionary
    for idx, keyStr in enumerate(dataframe_key_list):
        df = dataframe_dictionary[keyStr]

        # Ensure the DataFrame is of float type to match Scanpy's behavior
        if not np.issubdtype(df.values.dtype, np.floating):
            df = df.astype(np.float32)

        # Computing total transcript count per cell
        cell_total_transcripts = df.sum(axis=1)

        # Set target_sum to the median of non-zero total counts if not provided
        if target_sum is None:
            cell_total_transcripts_nonzero = cell_total_transcripts[cell_total_transcripts > 0]
            target_sum = np.median(cell_total_transcripts_nonzero)

        # Normalizing
        # Avoid division by zero by setting the factor to 1 for cells with zero counts
        normalization_factor = np.where(cell_total_transcripts > 0, cell_total_transcripts / target_sum, 1)
        df = df.div(normalization_factor, axis=0)

        # Store the normalized DataFrame in the updated dictionary
        dataframe_dictionary_updated[keyStr + "_norm"] = df

    return dataframe_dictionary_updated


def log1p_transform_df(dataframe_dictionary, dataframe_key_list, base=None):
    """
    Applies log1p transformation to the selected DataFrames in the dictionary.

    Parameters:
        dataframe_dictionary (dict): Dictionary containing DataFrames to select from for log1p transformation.
        dataframe_key_list (list): List of DataFrame keys to fetch from the dictionary and apply transformation.
        base (float, optional): The logarithmic base. If None, uses the natural logarithm.

    Returns:
        dict: Dictionary with the log1p-transformed DataFrames.
    """
    # Checking proper input
    if not dataframe_dictionary or not isinstance(dataframe_dictionary, dict):
        raise ValueError("dataframe_dictionary must be a non-empty dictionary of DataFrames.")
    if not dataframe_key_list or not isinstance(dataframe_key_list, list):
        raise ValueError("dataframe_key_list must be a non-empty list of DataFrame keys.")
    if not all(keyStr in dataframe_dictionary for keyStr in dataframe_key_list):
        raise ValueError("All strings in dataframe_key_list must correspond to keys in dataframe_dictionary.")
    if base is not None and (base <= 0 or base == 1):
        raise ValueError("Base must be a positive non-zero, non-one number.")

    # Initializing log-transformed DF dict
    dataframe_dictionary_updated = {key: df.copy() for key, df in dataframe_dictionary.items()}

    # Iterating over DataFrames selected by provided keys, from the provided dictionary
    for idx, keyStr in enumerate(dataframe_key_list):
        df = dataframe_dictionary[keyStr]

        # Log1p transforming, defaulting to natural log if base=None
        if base is None:
            df = np.log1p(df)
        else:
            df = np.log1p(df) / np.log(base)

        # Adding to updated df
        dataframe_dictionary_updated[keyStr + "_log"] = df

    return dataframe_dictionary_updated


def scale_df(dataframe_dictionary,
             dataframe_key_list,
             method,
             mad_robustize_epsilon=1e-18,
             robustize_unit_variance=False,
             minmax_feature_range=(0, 1)):
    """
    Standardizes the selected DataFrames in the dictionary using different scaling methods.

    Parameters:
        dataframe_dictionary (dict): Dictionary containing DataFrames to select from for standardization.
        dataframe_key_list (list): List of DataFrame keys to fetch from the dictionary and apply standardization.

        Duplicated from the pycytominer.normalize() function:
        method (str): Standardization method. Available methods are standardize, robustize, mad_robustize, minmax.
        mad_robustize_epsilon: float, optional
            The mad_robustize fudge factor parameter. The function only uses
            this variable if method = "mad_robustize". Set this to 0 if
            mad_robustize generates features with large values.
        robustize_unit_variance: bool, optional
            If True, and method is 'robustize', scales data so that normally distributed features have variance of 1.
        minmax_feature_range: tuple (min, max), optional
            Defines the feature range (min, max) for MinMaxScaler. Default is (0, 1).

    Returns:
        dict: Dictionary with the standardized DataFrames.
    """
    # Checking proper input
    if not dataframe_dictionary or not isinstance(dataframe_dictionary, dict):
        raise ValueError("dataframe_dictionary must be a non-empty dictionary of DataFrames.")
    if not dataframe_key_list or not isinstance(dataframe_key_list, list):
        raise ValueError("dataframe_key_list must be a non-empty list of DataFrame keys.")
    if not all(keyStr in dataframe_dictionary for keyStr in dataframe_key_list):
        raise ValueError("All strings in dataframe_key_list must correspond to keys in dataframe_dictionary.")

    avail_methods = ["standardize", "robustize", "mad_robustize", "minmax"]
    method = method.lower()
    if method not in avail_methods:
        raise ValueError(f"operation must be one of {avail_methods}")

    # Setting scaler
    if method == "standardize":
        scaler = StandardScaler()
    elif method == "robustize":
        scaler = RobustScaler(unit_variance=robustize_unit_variance)
    elif method == "mad_robustize":
        scaler = RobustMAD(epsilon=mad_robustize_epsilon)
    elif method == "minmax":
        scaler = MinMaxScaler(feature_range=minmax_feature_range)

    # Initializing standardized DF dict
    dataframe_dictionary_updated = {key: df.copy() for key, df in dataframe_dictionary.items()}

    # Iterating over DataFrames selected by provided keys, from the provided dictionary
    for idx, keyStr in enumerate(dataframe_key_list):
        df = dataframe_dictionary[keyStr]

        try:
            standardizedArray = scaler.fit_transform(df)
        except Exception as e:
            raise ValueError(f"Error during scaling DataFrame '{keyStr}': {e}")

        # Convert the np array back to a df
        standardizedDF = pd.DataFrame(standardizedArray, index=df.index, columns=df.columns)

        # Adding to updated df
        dataframe_dictionary_updated[keyStr + "_scaled"] = standardizedDF

    return dataframe_dictionary_updated