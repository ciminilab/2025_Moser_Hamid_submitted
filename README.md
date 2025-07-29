# VSTE
Variational Spatial Transcriptomics Encoder (VSTE)

## Description
This codebase contains two branches with separate conda environments, one for VAE training under `src` with the conda env file `requirements.txt` and `Dockerfile`, 
and one for data analysis and clustering under `VAE_SpotCount_CombinedApproach` with the conda env file `vste_analysis_environment.txt`. 
The VAE training branch is composed of several unimodal and multimodal VAE models with a single encoder and single or double decoder architecture. 
The analysis branch is composed of Jupyter notebooks and a script for the analysis of the latent spaces obtained from the training branch 
along with transcript counts and classical CellProfiler features. 

## Docker
To run the training scripts build the Dockerfile provided in this repository and execute the scripts in the docker container.  

#### Build the docker image:    
`docker build -t vste .`

#### Run the docker image:  
`docker run --gpus all -itd --ipc=host --name vste -v $(pwd):/VSTE -v /path/to/_saved_data_:/VSTE/src/_saved_data_  vste`

#### Execute the training scripts:  
`docker exec -it vste python src/baysor_spatial_staining_combined_double_decoder.py`


## Baysor dataset
Please download the public [Baysor dataset](https://datadryad.org/stash/dataset/doi:10.5061/dryad.jm63xsjb2), and save the file according to the following file structure: 
```plaintext
src/
└── _data_/
    └── baysor_output/
        ├── cell_assignment.csv
        ├── cell_meta_data.csv
        ├── dapi_stack.tiff
        ├── membrane_stack.tiff
        ├── gene_counts.csv
        ├── molecules.csv
        ├── segmentation_counts.tsv
        └── segmentation.csv
```
## Training scripts
The repository contains three training scripts for training a VAE model on the MERFISH transcripts (VAE1), dapi and membrane crops (VAE2) and on the combination of these (VAE3-5).  
#### Modalities:
VAE1: `src/baysor_spatial_data.py`  
VAE2: `src/baysor_staining_crops.py`  
VAE3-5: `src/baysor_spatial_staining_combined_double_decoder.py`  

#### Parameters and experiment configuration:
The scripts contains a number of tunable parameters found in the beginning of the script such as 'Z_DIM', 'EPOCHS', or 'LEARNING_RATE'.  
Additionally, users can specify the model configuration i.e. number of layers and channels in the encoder and decoder or adapt weight decay and loss function.

#### Dry run:
Setting 'DRY_RUN' to 'True' will run the script for a small number of epochs without logging any metrics. If 'DRY_RUN' is False all parameters will be logged to [Neptune](https://app.neptune.ai/).  
These logs include training loss and validation loss along with all required parameter to reproduce results.

#### Evaluation:
By setting 'RUN_INFERENCE' to True, a scripts will evaluate the latent representations by performing a clustering analysis and calculating various metrics (`src/validate_embeddings.py`).  
These metrics along with clustering results and cluster/cell-type correlation will be logged to Neptune as well.

#### Notebooks:
`src/notebooks` contains a selection of Jupyter notebooks to experiment with latent vectors and clustering (including an mnist example).  
NOTE: The notebooks might be outdated and not use the latest changes of model classes.

## Analysis scripts
The repository contains two analysis Jupyter notebooks and a script for the processing, clustering, and predictive modeling of 
the data in `VAE_SpotCount_CombinedApproach/data_restructured/` composed of latent spaces, transcript counts, and CellProfiler features (produced with `cellprofiler_classical_feature_measurement_pipeline.cppipe`):
- `Final_Full_Gene_Multimodal_Analysis.ipynb`: comparative joint analysis of the full feature spaces.
- `Final_Gene_Subset_Multimodal_Analysis.ipynb`: comparative joint analysis of a subset of genes with the morphological latent representation (VAE2).
- `Final_Multiprocessing_PERMANOVA_PERMDISP`: statistical analysis (PERMANOVA and PERMDISP) is separated into a standalone script to enable multiprocessing.

