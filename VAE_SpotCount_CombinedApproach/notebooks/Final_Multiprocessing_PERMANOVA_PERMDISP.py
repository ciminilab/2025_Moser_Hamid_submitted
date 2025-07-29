# IMPORTS
# Necessarily early imports
import numpy as np, random
# Ensure deterministic output
np.random.seed(137)
random.seed(137)
import warnings
from cryptography.utils import CryptographyDeprecationWarning
warnings.filterwarnings(
    "ignore",
    category=CryptographyDeprecationWarning,
    module=r"paramiko.*",
)

# Local
from utils.preprocess_data_utils import *
from utils.read_utils import *
from utils.evaluate_clustering_utils import *

# Standard / external
import os, concurrent.futures
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import scanpy as sc, scipy.spatial, scipy.spatial.distance
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from skbio.stats.distance import permanova, permdisp, DistanceMatrix
from scipy.sparse import SparseEfficiencyWarning

# Suppress chatter
sc.settings.verbosity = 0
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.simplefilter("ignore", category=SparseEfficiencyWarning)
warnings.filterwarnings("ignore", message=".*single numeric RGB or RGBA.*")
warnings.filterwarnings("ignore", message=".*More than 20 figures.*")


# DIRECTORIES and FILENAMES
fileNameDict = {
    "gene_count" : "gene_counts.csv",
    "cell_metadata" : "cell_meta_data.csv",
    "cell_assignment" : "cell_assignment.csv",
    "embedding1" : "baysor_model_inference_VSTE-328.csv",
    "embedding2" : "staining_crops_model_inference_VSTE-322.csv",
    "embedding3" : "baysor_model_inference_VSTE-347.csv",
    "embedding4" : "baysor_model_inference_VSTE-350.csv",
    'embedding5': 'baysor_model_inference_VSTE-371.csv',
    "cp_100" : "data_crop100.db",
    "cp_300" : "data_crop300.db",
}

inputDir  = "../data_restructured"
outputDir = "../analysis_output/final/permutational_stats_results"

# Function to compute PERMANOVA + PERMDISP  (runs in worker processes)
def compute_modality_stats(args):
    """
        Computes pairwise PERMANOVA and PERMDISP p-values for a given modality's feature matrix,
        using a provided vector of group labels (e.g. cell types). Designed for parallel execution.

        Parameters
        ----------
        args : tuple
            A tuple of (modalityName, dataMatrix, cellGroups), where:
                - modalityName (str): Label for the modality (e.g. "TC", "LS1").
                - dataMatrix (ndarray or sparse matrix): Feature matrix (cells X features).
                - cellGroups (array-like): 1D vector of labels for each cell.

        Returns
        -------
        pvalsPermanovaDF : pd.DataFrame
            DataFrame of adjusted PERMANOVA p-values (upper triangle only).

        pvalsPermdispDF : pd.DataFrame
            DataFrame of adjusted PERMDISP p-values (lower triangle only).

        Notes
        -----
        - If result CSVs already exist for the modality, they are loaded and returned directly
            (useful for customizing the plotting function separately).
        - NaNs in the matrix are imputed using column-wise means.
        - The matrix is standardized using z-score normalization.
        - Benjamini-Hochberg FDR correction is applied to all p-values.
        - Output files are saved in: {outputDir}/stats/{modalityName}_permanova.csv and _permdisp.csv
    """
    # Unpacking args
    modalityName, dataMatrix, cellGroups = args

    # Settings paths
    statsFolder = os.path.join(outputDir, "stats")
    os.makedirs(statsFolder, exist_ok=True)
    csvPermanovaPath = os.path.join(statsFolder, f"{modalityName}_permanova.csv")
    csvPermdispPath = os.path.join(statsFolder, f"{modalityName}_permdisp.csv")

    # Skip if both PERMANOVA and PERMDISP already done and return as DFs
    if os.path.isfile(csvPermanovaPath) and os.path.isfile(csvPermdispPath):
        pvalsPermanovaDF = pd.read_csv(csvPermanovaPath, index_col=0)
        pvalsPermdispDF = pd.read_csv(csvPermdispPath, index_col=0)
        return pvalsPermanovaDF, pvalsPermdispDF

    # Ensure matrix is dense if sparse
    if hasattr(dataMatrix, "toarray"):
        dataMatrix = dataMatrix.toarray()
    # Impute NaNs with column means
    dataMatrix = np.nan_to_num(dataMatrix, nan=np.nanmean(dataMatrix, axis=0))
    # Apply z-score normalization
    dataMatrix = StandardScaler().fit_transform(dataMatrix)

    # Fetching cell types from cellGroups vector
    unqCellGroups = np.sort(np.unique(cellGroups))
    numUnqCellGroups = len(unqCellGroups)
    # Initializing DFs for PERMANOVA and PERMDISP
    pvalsPermanovaDF = pd.DataFrame(np.nan, index=unqCellGroups, columns=unqCellGroups)
    pvalsPermdispDF = pd.DataFrame(np.nan, index=unqCellGroups, columns=unqCellGroups)

    # Pair-wise PERMANOVA and PERMDISP comparisons
    for i, g1 in enumerate(unqCellGroups):
        for j, g2 in enumerate(unqCellGroups):
            # Avoid redundant and diagonal comparisons
            if i >= j:
                continue
            # Finding indices of cells in either label group and subsetting the dataMatrix on them
            cellIndices = np.where((cellGroups == g1) | (cellGroups == g2))[0]
            dataMatrixSubset = dataMatrix[cellIndices]

            # Generate string IDs for the cellIndices
            ids = [f"s{i}" for i in range(len(cellIndices))]
            # Subset the cellGroups vector to the cellIndices with string ID as index, wrapped as pd series
            cellGroupsSubset = pd.Series(cellGroups[cellIndices], index=ids, name="grp")

            # Compute pair-wise distances between all cells
            euclidDist = scipy.spatial.distance.pdist(dataMatrixSubset, metric="euclidean")
            # Converting the condensed distance vector to a squareform distance matrix
            euclidDistMatrix = DistanceMatrix(scipy.spatial.distance.squareform(euclidDist), ids)

            # Run PERMANOVA and PERMDISP for this pair and fetch p-values, stored in the pval DFs
            pvalsPermanovaDF.loc[g1, g2] = permanova(euclidDistMatrix, grouping=cellGroupsSubset,
                                                     permutations=999)["p-value"]
            pvalsPermdispDF.loc[g2, g1] = permdisp(euclidDistMatrix, grouping=cellGroupsSubset,
                                                   permutations=999)["p-value"]

    # Generate a tuple of two arrays (rows & cols) for the upper and lower triangles of a matrix
    # of numUnqCellGroups X numUnqCellGroups size, exclude diagonal (k)
    upperTriangleIndexTuple = np.triu_indices(numUnqCellGroups, k=1)
    lowerTriangleIndexTuple = np.tril_indices(numUnqCellGroups, k=-1)
    # Run Benjamini-Hochberg FDR for multiple testing correction
    if pvalsPermanovaDF.values[upperTriangleIndexTuple].size:
        pvalsPermanovaDF.values[upperTriangleIndexTuple] = multipletests(pvalsPermanovaDF.values[upperTriangleIndexTuple],
                                                                         method="fdr_bh")[1]
    if pvalsPermdispDF.values[lowerTriangleIndexTuple].size:
        pvalsPermdispDF.values[lowerTriangleIndexTuple] = multipletests(pvalsPermdispDF.values[lowerTriangleIndexTuple],
                                                                        method="fdr_bh")[1]
    # Saving p-value DFs as CSVs and returning them for downstream plotting
    pvalsPermanovaDF.to_csv(csvPermanovaPath)
    pvalsPermdispDF.to_csv(csvPermdispPath)
    return pvalsPermanovaDF, pvalsPermdispDF

# Function to plot heatmaps
def plot_modality_heatmap(modalityName, pvalPermanovaMatrix, pvalPermdispMatrix):
    """
        Plots a heatmap of adjusted PERMANOVA and PERMDISP p-values for a given modality.

        The upper triangle of the matrix displays PERMANOVA p-values (group centroid differences),
        while the lower triangle displays PERMDISP p-values (group dispersion differences).
        P-values above 0.05 are colored black for visual emphasis.

        Parameters
        ----------
        modalityName : str
            Descriptive name for the modality (used for figure title and file naming).

        pvalPermanovaMatrix : pd.DataFrame
            Square DataFrame of PERMANOVA p-values (upper triangle filled).

        pvalPermdispMatrix : pd.DataFrame
            Square DataFrame of PERMDISP p-values (lower triangle filled).

        Returns
        -------
        None
            The function saves a heatmap image to disk and does not return any value.

        Notes
        -----
        - The output image is saved to: {outputDir}/stats/{modalityName}_pvalues_heatmap.png
        - P-values > 0.05 are shown in black to indicate non-significance.
        - The color map is fixed to Reds with a max value of 0.05.
    """
    # Setting paths
    statsFolder = os.path.join(outputDir, "stats")
    os.makedirs(statsFolder, exist_ok=True)
    heatmapExportPath = os.path.join(statsFolder, f"{modalityName}_pvalues_heatmap.png")

    # Fetching cell groups
    unqCellGroups = pvalPermanovaMatrix.index
    numUnqCellGroups = len(unqCellGroups)
    # Copying PERMANOVA p-val matrix to insert PERMDISP p-vals
    combinedPvalMatrix = pvalPermanovaMatrix.copy()
    for i in range(numUnqCellGroups):
        # Looping only where i > j to populate only lower triangle with PERMDISP p-vals
        for j in range(i):
            combinedPvalMatrix.iloc[i, j] = pvalPermdispMatrix.iloc[i, j]

    # Load cmap and set anything over vmax (0.05) to black
    cmap = plt.cm.Reds.copy()
    cmap.set_over("black")
    plt.figure(figsize=(8, 6))
    img = plt.imshow(combinedPvalMatrix, cmap=cmap, vmin=0, vmax=0.05,
                     origin="upper", interpolation="nearest")

    # Add white grid
    ax = plt.gca()
    ax.set_xticks(np.arange(numUnqCellGroups + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(numUnqCellGroups + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Color-bar and labels
    plt.colorbar(img, extend="max").set_label("p-value")

    # Adding diagonal separator
    plt.plot([-0.5, numUnqCellGroups - 0.5], [-0.5, numUnqCellGroups - 0.5], color="black")

    plt.xticks(np.arange(numUnqCellGroups), unqCellGroups, rotation=45, ha="right")
    plt.yticks(np.arange(numUnqCellGroups), unqCellGroups)
    plt.title(f"{modalityName} Adjusted p-values\n(upper: PERMANOVA, lower: PERMDISP)")
    plt.tight_layout()
    plt.savefig(heatmapExportPath, dpi=600)
    plt.close()

# MAIN
if __name__ == "__main__":
    # Read all raw data files, CSVs and DBs
    rawDF = {}
    for lbl, fn in fileNameDict.items():
        p = os.path.join(inputDir, fn)
        if fn.endswith("csv"):
            rawDF[lbl] = load_csv_to_df(parent_path=inputDir,
                                        file_name=fn, index_col=0)
        else:
            d = load_db_to_df_dict(parent_path=inputDir,
                                   file_name=fn)
            rawDF[lbl + "_perObj"] = d["Per_Object"]
            rawDF[lbl + "_perImg"] = d["Per_Image"]

    # Pre-processing CP data
    # Checking if feature-selected CP data exists (post PyCytominer)
    os.makedirs(outputDir + "/CP_after_pcm", exist_ok=True)
    cp100Path = outputDir + "/CP_after_pcm/cp100.csv"
    cp300Path = outputDir + "/CP_after_pcm/cp300.csv"
    cpExists = os.path.isfile(cp100Path) and os.path.isfile(cp300Path)
    # Exclude cp_100 and cp_300 from DF dict as they will be handled separately
    excl = ["cp_100", "cp_300"]
    preDF = {k: v for k, v in rawDF.items()
             if not any(s in k for s in excl)}

    # Setting cell_index as index as a guard
    for key, df in preDF.items():
        if 'cell_index' in df.columns:
            preDF[key] = df.set_index('cell_index')

    # If no pre-processed CP data exists, call the utils fxn merge_df_unq_cols_by_val to merge the per img and obj data
    if not cpExists:
        preDF["cp_100"] = merge_df_unq_cols_by_val(
            rawDF["cp_100_perObj"], rawDF["cp_100_perImg"],
            suffixes=("_perObj", "_perImg"))[0]
        preDF["cp_300"] = merge_df_unq_cols_by_val(
            rawDF["cp_300_perObj"], rawDF["cp_300_perImg"],
            suffixes=("_perObj", "_perImg"))[0]

    # Processing the GT DF (cell assignment) (renaming, dropping 'removed' cell types and such),
    # before common-index filtering or generating intersection matrix
    ca = rawDF["cell_assignment"].copy()
    ca.rename(columns={"leiden_final": "cell_type"}, inplace=True)
    ca = ca[ca["cell_type"] != "Removed"]
    # Mapping unique cell types to integers
    mapping = {ct: i+1 for i, ct in enumerate(ca["cell_type"].unique())}
    ca["cell_type_number"] = ca["cell_type"].map(mapping)
    preDF["cell_assignment"] = ca
    # Filter common indices
    preDF = filter_to_common_indices(preDF)[0]

    # If pre-processed CP data doesn't exist, run feature selection from utils and save
    if not cpExists:
        kws = ["meta", "index", "imagenumber", "_y_", "_x_"]
        preDF["cp_100"] = drop_irrelevant_cols(preDF["cp_100"], kws)[0]
        preDF["cp_300"] = drop_irrelevant_cols(preDF["cp_300"], kws)[0]
        preDF["cp_100"].to_csv(cp100Path)
        preDF["cp_300"].to_csv(cp300Path)
    # If pre-processed CP data exists, no feature selection/clean-up needed
    else:
        preDF["cp_100"] = pd.read_csv(cp100Path, index_col=0)
        preDF["cp_300"] = pd.read_csv(cp300Path, index_col=0)

    # Adjusting latent variable names by prefixing with 'z_'
    preDF = adjust_latent_var_names(preDF)

    # Create AnnData object dict
    keys = ["gene_count", "embedding1", "embedding2",
            "embedding3", "embedding4", "embedding5", "cp_100", "cp_300"]
    annDataObjDict = create_anndata_obj(preDF, preDF["cell_metadata"],
                                        keys, indices_to_concatenate=None)

    # Add cell spatial coords to the AnnData objects
    xy = preDF["cell_metadata"][["x", "y"]].values
    for o in annDataObjDict.values():
        o.obsm["spatial"] = xy.copy()

    # Normalize and log transform TC
    sc.pp.normalize_total(annDataObjDict["gene_count"], inplace=True)
    sc.pp.log1p(annDataObjDict["gene_count"])

    # Build pickle-safe matrices + label vector
    dataMatrixDict = {
        "TC"  : annDataObjDict["gene_count"].X,
        "LS1" : annDataObjDict["embedding1"].X,
        "LS2" : annDataObjDict["embedding2"].X,
        "LS3" : annDataObjDict["embedding3"].X,
        "LS4" : annDataObjDict["embedding4"].X,
        "LS5" : annDataObjDict["embedding5"].X,
        "CPsm": annDataObjDict["cp_100"].X,
        "CPlg": annDataObjDict["cp_300"].X,
    }
    # Extract cell type labels as a 1D NumPy array for use in downstream comparisons
    labels = preDF["cell_assignment"]["cell_type"].values

    # Run PERMANOVA and PERMDISP in parallel
    mods  = ["TC", "LS1", "LS2", "LS3", "LS4", "LS5", "CPsm", "CPlg"]
    # Packing tasks of modality name, feature matrix, and cell groupings
    tasks = [(m, dataMatrixDict[m], labels) for m in mods]
    # Initialize results dict
    results = {}

    # Launch parallel processing for the statistical computation
    # Each task contains: (modality name, feature matrix, cell type labels)
    # Each compute_modality_stats() call returns: (PERMANOVA p-value matrix, PERMDISP p-value matrix)
    with concurrent.futures.ProcessPoolExecutor() as exe:
        for (mod, _X, _g), (p_perm, p_disp) in zip(
                tasks, exe.map(compute_modality_stats, tasks)):
            # Store the results for each modality by name
            results[mod] = (p_perm, p_disp)


    # Plot the heatmaps (serially, parallelism unnecessary)
    for mod, (p_perm, p_disp) in results.items():
        plot_modality_heatmap(mod, p_perm, p_disp)

    print("Done!")