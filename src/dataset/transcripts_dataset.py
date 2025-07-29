import random
from multiprocessing import Pool
import numpy as np
from scipy import signal
import sparse
from torch.utils.data import Dataset
from .utils import create_multichannel_representation
import torch


def get_cell_transcripts(cell_indices, transcripts_df, n_genes, input_dim, padding):
    cell_transcripts = []
    sample_weights = []
    processed_cell_indices = []
    for cell_idx in cell_indices:
        cell_transcripts_df = transcripts_df.loc[cell_idx]
        # Skip cells with fewer than 10 transcripts
        if len(cell_transcripts_df) < 10:
            continue

        try:
            x_values, y_values = cell_transcripts_df['x'], cell_transcripts_df['y']
            x_min, y_min = int(np.min(x_values)), int(np.min(y_values))
            input_array, weights = create_multichannel_representation(cell_transcripts_df, x_min, y_min, n_genes=n_genes, padding=padding, crop_size=input_dim)
            sparse_input_array = sparse.COO(input_array)
        except Exception as e:
            print("Error processing cell", cell_idx)
            continue

        cell_transcripts.append(sparse_input_array)
        sample_weights.append(weights)
        processed_cell_indices.append(cell_idx)

    return (processed_cell_indices, cell_transcripts, sample_weights)

class TranscriptDataset(Dataset):
    def __init__(self, transcripts_df, input_dim, n_genes, n_workers=2, padding=1, apply_data_augmentation=False, tensor_type=torch.float32):
        self.input_dim = input_dim
        self.n_genes = n_genes
        self.apply_data_augmentation = apply_data_augmentation
        self.tensor_type = tensor_type

        # Compute the cell of all cells in the dataset
        cell_index_list = np.unique(transcripts_df['cell_index'].values)
        # Split the list of cells into chunks for multiprocessing
        cell_index_split_list = np.array_split(cell_index_list, n_workers)
        transcripts_df.set_index('cell_index', inplace=True)

        with Pool(processes=n_workers) as pool:
            args = [[cell_index_sublist, transcripts_df.loc[cell_index_sublist], n_genes, input_dim, padding] for cell_index_sublist in cell_index_split_list]
            results = pool.starmap(get_cell_transcripts, args)

        cell_transcripts = []
        cell_indices = []
        sample_weights = []
        for process_result in results:
            cell_indices.extend(process_result[0])
            cell_transcripts.extend(process_result[1])
            sample_weights.extend(process_result[2])
        self.cell_transcripts = cell_transcripts
        self.cell_indices = cell_indices
        self.sample_weights = sample_weights

    def __getitem__(self, index):
        sparse_input_array = self.cell_transcripts[index]
        input_array = sparse_input_array.todense()

        if self.apply_data_augmentation:
            rotation_index = random.randint(0, 4)
            input_array = np.rot90(input_array, rotation_index, axes=(1, 2))

        cell_index = self.cell_indices[index]
        weights = torch.tensor(self.sample_weights[index], dtype=self.tensor_type)
        model_input_tensor = torch.tensor(input_array.copy(), dtype=self.tensor_type)
        return (model_input_tensor, weights, cell_index)

    def __len__(self):
        return len(self.cell_indices)
