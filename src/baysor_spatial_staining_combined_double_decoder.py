import os
import sys
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch
import neptune
from tifffile import imread

from models.vae_model.model_config import DoubleDecoderCNNConfig
from models.model_class import Model
from models.loss_functions.validation_logger import DoubleDecoderCombinedSingleCropTranscriptsValidationLogger 
from models.loss_functions.loss_functions import DoubleDecoderLossFunction
from dataset.combined_crops_transcripts_dataset import CombinedCropsTranscriptDataset
from validate_embeddings import validate_embeddings

DATA_PATH = "src/_data_/"
BAYSOR_OUTPUT_PATH = os.path.join(DATA_PATH, 'baysor_output')
SAVED_DATA_PATH = "src/_saved_data_"

## Define training parameters:
LEARNING_RATE = 0.0005
EPOCHS = 15
BATCH_SIZE = 18

INPUT_DIM = 200 # Equal the crop size
Z_DIM = 50
PADDING = 4
GENE_COUNT_THRESHOLD = -1
APPLY_DATA_AUGMENTATION = True
TENSOR_TYPE = torch.float32

# Neptune experiment description
EXPERIMENT_DESCRIPTION = f"VAE-4 DoubleDecoder: Add gene count loss term, test for 15 epochs"
NEPTUNE_API_TOKEN = None

## Experiment variables
REGENERATE_DATASET = True # Set to true to overwrite existing dataset (i.e. after changes to dataset)
DRY_RUN = True
RUN_INFERENCE = True

# Set epochs to 5 for dry runs
EPOCHS = 1 if DRY_RUN else EPOCHS

## Load datasets
saved_train_dataset_path = os.path.join(SAVED_DATA_PATH, "baysor_staining_transcript_train_dataset.pt")

# Load data frames with the decoded transcripts
segmentation_file_path = os.path.join(BAYSOR_OUTPUT_PATH, "segmentation.csv")
segmentation_df = pd.read_csv(segmentation_file_path)
segmentation_df.rename(columns={"cell": "cell_index"}, inplace=True)
segmentation_df.drop(segmentation_df[segmentation_df['is_noise']].index, inplace=True)

dapi_stack_file_path = os.path.join(BAYSOR_OUTPUT_PATH, "dapi_stack.tif")
dapi_stack = imread(dapi_stack_file_path)

membrane_stack_file_path = os.path.join(BAYSOR_OUTPUT_PATH, "membrane_stack.tif")
membrane_stack = imread(membrane_stack_file_path)

# Apply gene count thresholding
gene_list = np.unique(segmentation_df['gene'].values)
if GENE_COUNT_THRESHOLD != -1:
    # Remove genes with less than GENE_COUNT_THRESHOLD transcripts
    cell_count_per_type = segmentation_df['gene'].value_counts()
    filtered_gene_list = cell_count_per_type[cell_count_per_type > GENE_COUNT_THRESHOLD].index.values
    N_GENES = len(filtered_gene_list)
    segmentation_df = segmentation_df[segmentation_df['gene'].isin(filtered_gene_list)]
    segmentation_df['barcode_id'] = segmentation_df['gene'].apply(lambda x: np.argwhere(filtered_gene_list == x).item())
else:
    N_GENES = len(gene_list)
    segmentation_df['barcode_id'] = segmentation_df['gene'].apply(lambda x: np.argwhere(gene_list == x).item())

# Model and traning configuration
cnn_config = {
    "model_config": {
        "hidden_dims": [N_GENES+2, 240, 240, 240, 240, 240], # Channels of hidden layers
        "z_dim": Z_DIM,
        "input_shape": [N_GENES+2, INPUT_DIM, INPUT_DIM],
        "decoder_one_input_shape": [N_GENES, INPUT_DIM, INPUT_DIM],
        "decoder_one_hidden_dims": [N_GENES, 240, 240, 240, 240, 240],
        "decoder_two_input_shape": [2, INPUT_DIM, INPUT_DIM],
        "decoder_two_hidden_dims": [2, 4, 8, 16, 32, 64],
        "relu_slope": 0.1,
        "batch_norm": True,
        "apply_sigmoid": True,
    },
    "hyperparameters": {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "loss_function": DoubleDecoderLossFunction,
        "beta": 0,
        "epsilon": 1e-6,
        "weight_decay": 1e-4
    }
}

cell_meta_data_file_path = os.path.join(BAYSOR_OUTPUT_PATH, "cell_meta_data.csv")
cell_meta_data_df = pd.read_csv(cell_meta_data_file_path)
cell_meta_data_df.rename(columns={"cell": "cell_index"}, inplace=True)

print("Load datasets...")
if REGENERATE_DATASET or not os.path.exists(saved_train_dataset_path):
    train_dataset = CombinedCropsTranscriptDataset(
        transcripts_df=segmentation_df,
        cell_meta_data_df=cell_meta_data_df,
        dapi_stack=dapi_stack,
        membrane_stack=membrane_stack,
        z_index=3,
        input_dim=INPUT_DIM,
        n_genes=N_GENES,
        n_workers=20,
        padding=PADDING,
        apply_data_augmentation=APPLY_DATA_AUGMENTATION,
        tensor_type=TENSOR_TYPE
    )

    torch.save(train_dataset, saved_train_dataset_path)
else:
    print("Load existing dataset...")
    train_dataset = torch.load(saved_train_dataset_path)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=BATCH_SIZE,
    prefetch_factor=20,
    persistent_workers=True,
    pin_memory=True
)

if DRY_RUN:
    neptune_run = None
    run_id = "DRY_RUN"
else:
    print("Initialize Neptune...")
    neptune_run = neptune.init_run(
        project="BroadImagingPlatform/VSTE",
        api_token=NEPTUNE_API_TOKEN,
    )
    
    neptune_run["dataset_config"] = {
       "dataset_class": train_dataset.__class__.__name__,
       "apply_data_augmentation": APPLY_DATA_AUGMENTATION,
       "input_dim": INPUT_DIM,
       "n_genes": N_GENES,
       "padding": PADDING,
       "gene_threshold": GENE_COUNT_THRESHOLD,
       "tensor_type": str(TENSOR_TYPE)
    }
    neptune_run["Experiment"] = EXPERIMENT_DESCRIPTION
    neptune_run["sys/tags"].add("VAE-4_double-decoder")
    run_id = neptune_run["sys/id"].fetch()

print("Create VAE model")
config_object = DoubleDecoderCNNConfig(**cnn_config)
validation_logger = DoubleDecoderCombinedSingleCropTranscriptsValidationLogger(neptune_run, logit_output=(not config_object.model_config.apply_sigmoid))
vae_cnn = Model(config_object, cuda_device_num=4, neptune_run=neptune_run, validation_logger=validation_logger, tensor_type=TENSOR_TYPE)

# Train the VAE model, catch potential exceptions and abort the run if any.
print("Training VAE model...")
vae_cnn.train(train_loader, train_loader)

print("Plot model reconstruction examples...")
def visualize_transcripts_model_output(model_shaped_tensor):
    binary_tensor = torch.round(model_shaped_tensor[0])
    transcript_reconstruction_tensor = torch.argmax(binary_tensor, 1)
    return transcript_reconstruction_tensor

def visualize_transcripts(model_shaped_tensor):
    transcript_reconstruction_tensor = torch.argmax(model_shaped_tensor[:, 2:], 1)
    return transcript_reconstruction_tensor

def visualize_dapi_crop_reconstruction_model_output(model_shaped_tensor):
    return model_shaped_tensor[1][:, 0]

def visualize_dapi_crop(model_shaped_tensor):
    return model_shaped_tensor[:, 0]

def visualize_membrane_crop_reconstruction_model_output(model_shaped_tensor):
    return model_shaped_tensor[1][:, 1]

def visualize_membrane_crop(model_shaped_tensor):
    return model_shaped_tensor[:, 1]

vae_cnn.plot_model_reconstruction(train_loader, visualize_transcripts, "transcripts-train", cmap=None, post_process_function_model=visualize_transcripts_model_output)
vae_cnn.plot_model_reconstruction(train_loader, visualize_dapi_crop, "dapi-stain-train", cmap='grey', post_process_function_model=visualize_dapi_crop_reconstruction_model_output)
vae_cnn.plot_model_reconstruction(train_loader, visualize_membrane_crop, "membrane-stain-train", cmap='grey', post_process_function_model=visualize_membrane_crop_reconstruction_model_output)

if not RUN_INFERENCE:
    print("Finished training.")
    sys.exit()

# Run model inference
print("Finished training, run inference...")

# Run inference
inference_data, cell_index_list = vae_cnn.inference(train_loader)
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
inference_data_file_path = os.path.join('src/__inference_data', f'baysor_model_inference_{run_id}.csv')
model_inference_df.to_csv(inference_data_file_path, index=True)

## Stop Neptune run
neptune_run.stop()
