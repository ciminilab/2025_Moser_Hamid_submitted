# This script contains functions for importing csv and db files as dfs

# Imports
import os
import pandas as pd
import sqlite3

def load_csv_to_df(parent_path, file_name, index_col=None):
    """
    Loads csv file(s) as pandas dataframes.

    Parameters:
        parent_path (str): Required. Single string referring to relative or absolute path to parent directory containing the csv file.
        file_name (str): Required. Single file name string. Can but does not need '.csv' extension.
        index_col (None or int): Optional (defaults to None). Single value (None or non-negative integer)  referring to whether and which column to denote as index.

    Returns:
        pd.DataFrame
    """

    # Confirming proper args
    if (not isinstance(parent_path, str) or
            not isinstance(file_name, str) or
            (index_col is not None and (not isinstance(index_col, int) or index_col < 0))):
        raise ValueError(f"Error: unexpected argument. Ensure parent_path and file_name are strings and index_col is None or a non-negative integer.")

    # Converting path to absolute if relative
    if not os.path.isabs(parent_path):
        parent_path = os.path.abspath(parent_path)

    # Confirming path is a directory
    if not os.path.isdir(parent_path):
        raise ValueError(f"Error: The folder '{parent_path}' does not exist or is not a directory.")

    # Appending '.csv' to file name if not included
    if not file_name.endswith('.csv'):
        file_name += '.csv'

    file_path = os.path.join(parent_path, file_name)

    # Importing
    try:
        dataframe = pd.read_csv(file_path, index_col=index_col)
        return dataframe
    except FileNotFoundError:
        raise ValueError(f"Error: The file '{file_name}' does not exist at the specified location.")
    except Exception as e:
        raise ValueError(f"An error occurred while reading '{file_name}': {e}")


def load_db_to_df_dict(parent_path, file_name):
    """
    Loads SQLite database tables to pandas dataframes in the form of a df dictionary.

    Parameters:
        parent_path (str): Required. Single string referring to relative or absolute path to parent directory containing the db file.
        file_name (str): Required. Single file name string. Can but does not need '.db' extension.

    Returns:
        dictionary of pd.DataFrames
    """
    # Confirming string inputs
    if not isinstance(parent_path, str) or not isinstance(file_name, str):
        raise ValueError(f"Error: unexpected argument. Ensure parent_path and file_name are strings.")

    # Converting path to absolute if relative
    if not os.path.isabs(parent_path):
        parent_path = os.path.abspath(parent_path)

    # Confirming path is a directory
    if not os.path.isdir(parent_path):
        raise ValueError(f"Error: The folder '{parent_path}' does not exist or is not a directory.")

    # Appending '.db' to file name if not included
    if not file_name.endswith('.db'):
        file_name += '.db'

    file_path = os.path.join(parent_path, file_name)

    # Load database tables into DFs
    try:
        with sqlite3.connect(file_path) as conn:
            # Fetch the list of all tables
            tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
            # Initialize the dictionary
            dataframes = {}
            # Load each table into a df
            for table_name in tables['name']:
                dataframes[table_name] = pd.read_sql_query(f"SELECT * FROM {table_name};", conn)

        return dataframes

    except Exception as e:
        raise ValueError(f"An error occurred while reading the database '{file_name}': {e}")


