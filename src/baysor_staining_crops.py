import os
import sys
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch
import neptune
from tifffile import imread

from models.vae_model.model_config import CNNVAEConfig
from models.model_class import Model
from models.loss_functions.loss_functions import L2ReconstructionLossFunction
from dataset.staining_crops_dataset import StainingCropsDataset
from validate_embeddings import validate_embeddings

DATA_PATH = "src/_data_/"
BAYSOR_OUTPUT_PATH = os.path.join(DATA_PATH, 'baysor_output')
SAVED_DATA_PATH = "src/_saved_data_"

# Neptune experiment description
EXPERIMENT_DESCRIPTION = "VAE-2: Data augmentation on crop size=250, 500 epochs"
NEPTUNE_API_TOKEN = None

## Define training parameters:
LEARNING_RATE = 0.00005
EPOCHS = 5
BATCH_SIZE = 16

CROP_SIZE = 250
Z_DIM = 50
APPLY_DATA_AUGMENTATION = True
TENSOR_TYPE = torch.float32

## Experiment variables
REGENERATE_DATASET = False # Set to true to overwrite existing dataset (i.e. after changes to dataset)
DRY_RUN = False # No Neptune run is initialized if set to True
RUN_INFERENCE = True

# Set epochs to 5 for dry runs
EPOCHS = 5 if DRY_RUN else EPOCHS

# Model and traning configuration
cnn_config = {
    "model_config": {
        "hidden_dims": [2, 4, 8, 12, 24, 48], # Channels of hidden layers
        "z_dim": Z_DIM,
        "input_shape": [2, CROP_SIZE, CROP_SIZE],
        "relu_slope": 0.1,
        "batch_norm": True,
        "apply_sigmoid": True,
    },
    "hyperparameters": {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "loss_function": L2ReconstructionLossFunction,
        "beta": 0
    }
}

## Load datasets
saved_train_dataset_path = os.path.join(SAVED_DATA_PATH, "staining_crop_train_dataset.pt")

# Load the data frame containing the data
dapi_stack_file_path = os.path.join(BAYSOR_OUTPUT_PATH, "dapi_stack.tif")
dapi_stack = imread(dapi_stack_file_path)

membrane_stack_file_path = os.path.join(BAYSOR_OUTPUT_PATH, "membrane_stack.tif")
membrane_stack = imread(membrane_stack_file_path)

segmentation_file_path = os.path.join(BAYSOR_OUTPUT_PATH, "segmentation.csv")
segmentation_df = pd.read_csv(segmentation_file_path)
segmentation_df.rename(columns={"cell": "cell_index", "mol_id": "barcode_id"}, inplace=True)
segmentation_df.drop(segmentation_df[segmentation_df['is_noise']].index, inplace=True)

cell_meta_data_file_path = os.path.join(BAYSOR_OUTPUT_PATH, "cell_meta_data.csv")
cell_meta_data_df = pd.read_csv(cell_meta_data_file_path)
cell_meta_data_df.rename(columns={"cell": "cell_index"}, inplace=True)

print("Load datasets...")
if REGENERATE_DATASET or not os.path.exists(saved_train_dataset_path):
    staining_crop_train_dataset = StainingCropsDataset(
        cell_meta_data_df.copy(),
        dapi_stack,
        membrane_stack,
        crop_size=CROP_SIZE,
        z_index=2,
        n_workers=10,
        apply_data_augmentation=APPLY_DATA_AUGMENTATION,
        tensor_type=TENSOR_TYPE
    )

    torch.save(staining_crop_train_dataset, saved_train_dataset_path)
else:
    print("Load existing dataset...")
    staining_crop_train_dataset = torch.load(saved_train_dataset_path)

staining_crop_train_loader = DataLoader(
    dataset=staining_crop_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=BATCH_SIZE,
    prefetch_factor=4*BATCH_SIZE,
    persistent_workers=True,
    pin_memory=True
)
staining_crop_test_loader = staining_crop_train_loader

if DRY_RUN:
    neptune_run = None
else:
    print("Initialize Neptune...")
    neptune_run = neptune.init_run(
        project="BroadImagingPlatform/VSTE",
        api_token=NEPTUNE_API_TOKEN,
    )
    
    neptune_run["dataset_config"] = {
       "dataset_class": staining_crop_train_dataset.__class__.__name__,
       "apply_data_augmentation": APPLY_DATA_AUGMENTATION,
       "crop_size": CROP_SIZE,
       "tensor_type": str(TENSOR_TYPE)
    }
    neptune_run["Experiment"] = EXPERIMENT_DESCRIPTION
    neptune_run["sys/tags"].add("staining_crops")
    run_id = neptune_run["sys/id"].fetch()

print("Create VAE model")
config_object = CNNVAEConfig(**cnn_config)
vae_cnn = Model(config_object, cuda_device_num=2, neptune_run=neptune_run, model_tag="crops_", tensor_type=TENSOR_TYPE)

# Train the VAE model, catch potential exceptions and abort the run if any.
print("Training VAE model...")
vae_cnn.train(staining_crop_train_loader, staining_crop_test_loader)

print("Plot model reconstruction examples...")
def show_dapi_reconstruction(model_shaped_tensor):
    return model_shaped_tensor[:, 0, :, :]

def show_membrane_reconstruction(model_shaped_tensor):
    return model_shaped_tensor[:, 1, :, :]

vae_cnn.plot_model_reconstruction(staining_crop_train_loader, show_dapi_reconstruction, "dapi-stain-train", cmap='grey')
vae_cnn.plot_model_reconstruction(staining_crop_train_loader, show_membrane_reconstruction, "membrane-stain-train", cmap='grey')

if not RUN_INFERENCE:
    print("Finished training.")
    sys.exit()

if DRY_RUN:
    run_id = "test"

# Run model inference
print("Finished training, run inference...")

# Run inference
inference_data, cell_index_list = vae_cnn.inference(staining_crop_train_loader)
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
model_inference_df.set_index('cell_index', inplace=True)
model_inference_df.to_csv(f'staining_crops_model_inference_{run_id}.csv', index=True)

## Stop Neptune run
neptune_run.stop()
