import os
import sys
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch
import neptune

from models.vae_model.model_config import CNNVAEConfig
from models.model_class import Model
from models.loss_functions.validation_logger import SegmentationValidationLogger
from models.loss_functions.loss_functions import MSEDiceLossFunction, BCEDiceLossFunction, DiceFocalLossFunction, VAEDiceLossFunction
from dataset.transcripts_dataset import TranscriptDataset
from validate_embeddings import validate_embeddings

DATA_PATH = "src/_data_/"
BAYSOR_OUTPUT_PATH = os.path.join(DATA_PATH, 'baysor_output')
SAVED_DATA_PATH = "src/_saved_data_"

# Neptune experiment description
EXPERIMENT_DESCRIPTION = "VAE-1: Some experiment"
NEPTUNE_API_TOKEN = None

## Define training parameters:
LEARNING_RATE = 0.0001
EPOCHS = 2
BATCH_SIZE = 16

INPUT_DIM = 100
N_GENES = 241
Z_DIM = 100

APPLY_DATA_AUGMENTATION = False
TENSOR_TYPE = torch.float32
PADDING = 3

## Experiment variables
REGENERATE_DATASET = False # Set to true to overwrite existing dataset (i.e. after changes to dataset)
DRY_RUN = False # No Neptune run is initialized if set to True
RUN_INFERENCE = True

# Set epochs to 5 for dry runs
EPOCHS = 2 if DRY_RUN else EPOCHS

# Model and traning configuration
cnn_config = {
    "model_config": {
        "hidden_dims": [N_GENES, 115, 115, 115, 115, 115], # Channels of hidden layers
        "z_dim": Z_DIM,
        "input_shape": [N_GENES, INPUT_DIM, INPUT_DIM],
        "relu_slope": 0.1,
        "batch_norm": True,
        "apply_sigmoid": True,
    },
    "hyperparameters": {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "loss_function": DiceFocalLossFunction,
        "beta": 0
    }
}


## Load datasets
saved_train_dataset_path = os.path.join(SAVED_DATA_PATH, "baysor_transcript_train_dataset.pt")

# Load data frames with the decoded transcripts
segmentation_file_path = os.path.join(BAYSOR_OUTPUT_PATH, "segmentation.csv")
segmentation_df = pd.read_csv(segmentation_file_path)
segmentation_df.rename(columns={"cell": "cell_index"}, inplace=True)
segmentation_df.drop(segmentation_df[segmentation_df['is_noise']].index, inplace=True)

# Add column with the barcode_id (i.e. index of the gene) to each transcript
gene_list = np.unique(segmentation_df['gene'].values)
segmentation_df['barcode_id'] = segmentation_df['gene'].apply(lambda x: np.argwhere(gene_list == x).item())

cell_meta_data_file_path = os.path.join(BAYSOR_OUTPUT_PATH, "cell_meta_data.csv")
cell_meta_data_df = pd.read_csv(cell_meta_data_file_path)
cell_meta_data_df.rename(columns={"cell": "cell_index", "x": "global_x", "y": "global_y"}, inplace=True)

print("Load datasets...")
if REGENERATE_DATASET or not os.path.exists(saved_train_dataset_path):
    transcript_train_dataset = TranscriptDataset(
        transcripts_df=segmentation_df,
        input_dim=INPUT_DIM,
        n_genes=N_GENES,
        n_workers=30,
        padding=PADDING,
        apply_data_augmentation=APPLY_DATA_AUGMENTATION,
        tensor_type=TENSOR_TYPE
    )

    torch.save(transcript_train_dataset, saved_train_dataset_path)
else:
    print("Load existing dataset...")
    transcript_train_dataset = torch.load(saved_train_dataset_path)

transcript_train_loader = DataLoader(
    dataset=transcript_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=BATCH_SIZE,
    prefetch_factor=20,
    persistent_workers=True,
    pin_memory=True
)

if DRY_RUN:
    neptune_run = None
else:
    print("Initialize Neptune...")
    neptune_run = neptune.init_run(
        project="BroadImagingPlatform/VSTE",
        api_token=NEPTUNE_API_TOKEN,
    )
    
    neptune_run["dataset_config"] = {
       "dataset_class": transcript_train_dataset.__class__.__name__,
       "apply_data_augmentation": APPLY_DATA_AUGMENTATION,
       "input_dim": INPUT_DIM,
       "n_genes": N_GENES,
       "padding": PADDING,
       "tensor_type": str(TENSOR_TYPE)
    }
    neptune_run["Experiment"] = EXPERIMENT_DESCRIPTION
    neptune_run["sys/tags"].add("baysor_spatial_data")
    run_id = neptune_run["sys/id"].fetch()

print("Create VAE model")
config_object = CNNVAEConfig(**cnn_config)
validation_logger = SegmentationValidationLogger(neptune_run, logit_output=(not config_object.model_config.apply_sigmoid))
vae_cnn = Model(config_object, cuda_device_num=4, neptune_run=neptune_run, validation_logger=validation_logger, tensor_type=TENSOR_TYPE)

# Train the VAE model, catch potential exceptions and abort the run if any.
print("Training VAE model...")
vae_cnn.train(transcript_train_loader, transcript_train_loader)

print("Plot model reconstruction examples...")
def visualize_transcripts(model_shaped_tensor):
    binary_tensor = torch.round(model_shaped_tensor)
    return torch.argmax(binary_tensor, 1)

vae_cnn.plot_model_reconstruction(transcript_train_loader, visualize_transcripts, "train", cmap=None)

if not RUN_INFERENCE:
    print("Finished training.")
    sys.exit()

# Run model inference
print("Finished training, run inference...")

# Run inference
inference_data, cell_index_list = vae_cnn.inference(transcript_train_loader)
inference_data = np.concatenate(inference_data, axis=0)
model_inference_df = pd.DataFrame(inference_data)
model_inference_df['cell_index'] = [x.item() for x in cell_index_list]

# Log embedding validation metrics
if neptune_run:
    print("Log embedding validation metrics to neptune")
    # Prepare ground-truth clustering
    cell_classification_path = os.path.join(BAYSOR_OUTPUT_PATH, "cell_assignment.csv")
    ground_truth_clustering_df = pd.read_csv(cell_classification_path, header=0)
    ground_truth_clustering_df.rename(columns={"cell": "cell_index", "leiden_final": "cell_type"}, inplace=True)

    # Drop removed cells
    ground_truth_clustering_df = ground_truth_clustering_df[ground_truth_clustering_df['cell_type'] != 'Removed']

    # Assign cell type numbers
    unique_cell_types = ground_truth_clustering_df['cell_type'].unique()
    cell_type_mapping = {cell_type: index + 1 for index, cell_type in enumerate(unique_cell_types)}
    ground_truth_clustering_df['cell_type_number'] = ground_truth_clustering_df['cell_type'].map(cell_type_mapping)

    cell_meta_data_df.rename(columns={"x": "global_x", "y": "global_y"}, inplace=True)
    validate_embeddings(model_inference_df, cell_meta_data_df, ground_truth_clustering_df, neptune_run, min_dist=0.05, resolution=0.85)

# Save embedding DataFrame
model_inference_df.to_csv(f'baysor_model_inference_{run_id}.csv', index=True)

## Stop Neptune run
neptune_run.stop()
