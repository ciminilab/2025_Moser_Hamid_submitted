import os
from multiprocessing import Pool
import random
from dataset.utils import create_multichannel_representation, get_cell_crop
import numpy as np
from tifffile import imsave
import matplotlib.pyplot as plt
import sparse
from torch.utils.data import Dataset
import torch


def get_cell_transcripts(cell_indices, transcripts_df, n_genes, input_dim, padding, cell_meta_df, dapi_image, membrane_image):
    cell_transcripts = []
    sample_weights = []
    dapi_crop_list = []
    membrane_crop_list = []
    processed_cell_indices = []
    for cell_idx in cell_indices:
        cell_transcripts_df = transcripts_df.loc[cell_idx]
        # Skip cells with fewer than 10 transcripts
        if len(cell_transcripts_df) < 10:
            continue

        dapi_crop, membrane_crop, x_min, y_min = get_cell_crop(cell_idx, cell_meta_df, dapi_image, membrane_image, input_dim)
        try:
            input_array, weights = create_multichannel_representation(cell_transcripts_df, x_min, y_min, n_genes=n_genes, padding=padding, crop_size=input_dim)
            sparse_input_array = sparse.COO(input_array)
        except Exception as e:
            print("Error processing cell", cell_idx)
            continue

        dapi_crop_list.append(dapi_crop)
        membrane_crop_list.append(membrane_crop)
        cell_transcripts.append(sparse_input_array)
        sample_weights.append(weights)
        processed_cell_indices.append(cell_idx)

    return (processed_cell_indices, cell_transcripts, sample_weights, dapi_crop_list, membrane_crop_list)


class CombinedCropsTranscriptDataset(Dataset):
    def __init__(self, transcripts_df, cell_meta_data_df, membrane_stack, dapi_stack, input_dim, n_genes, z_index, n_workers=2, padding=1, apply_data_augmentation=False, tensor_type=torch.float32):
        self.input_dim = input_dim
        self.n_genes = n_genes
        self.apply_data_augmentation = apply_data_augmentation
        self.tensor_type = tensor_type

        dapi_image = dapi_stack[z_index].transpose()
        membrane_image = membrane_stack[z_index].transpose()

        # Compute the cell of all cells in the dataset
        cell_index_list = np.unique(cell_meta_data_df['cell_index'].values)
        # Split the list of cells into chunks for multiprocessing
        cell_index_split_list = np.array_split(cell_index_list, n_workers)
        transcripts_df.set_index('cell_index', inplace=True)
        cell_meta_data_df.set_index("cell_index", inplace=True)

        with Pool(processes=n_workers) as pool:
            args = [[cell_index_sublist, transcripts_df.loc[cell_index_sublist], n_genes, input_dim, padding, cell_meta_data_df.loc[cell_index_sublist], dapi_image, membrane_image] for cell_index_sublist in cell_index_split_list]
            results = pool.starmap(get_cell_transcripts, args)

        cell_transcripts = []
        cell_indices = []
        sample_weights = []
        dapi_stack_crop_list = []
        membrane_stack_crop_list = []
        for process_result in results:
            cell_indices.extend(process_result[0])
            cell_transcripts.extend(process_result[1])
            sample_weights.extend(process_result[2])
            dapi_stack_crop_list.extend(process_result[3])
            membrane_stack_crop_list.extend(process_result[4])
        self.cell_transcripts = cell_transcripts
        self.cell_indices = cell_indices
        self.sample_weights = sample_weights
        self.dapi_crop_list = dapi_stack_crop_list
        self.membrane_crop_list = membrane_stack_crop_list

    def __getitem__(self, index):
        sparse_input_array = self.cell_transcripts[index]
        input_array = sparse_input_array.todense()

        cell_index = self.cell_indices[index]
        weights = torch.tensor(self.sample_weights[index], dtype=self.tensor_type)
        transcripts_tensor = torch.tensor(input_array.copy(), dtype=self.tensor_type)

        dapi_crop = self.dapi_crop_list[index]
        dapi_crop_tensor = torch.tensor(dapi_crop, dtype=self.tensor_type).unsqueeze(0)
        membrane_crop = self.membrane_crop_list[index]
        membrane_crop_tensor = torch.tensor(membrane_crop, dtype=self.tensor_type).unsqueeze(0)
        model_input_tensor = torch.cat((dapi_crop_tensor, membrane_crop_tensor, transcripts_tensor), dim=0)

        if self.apply_data_augmentation:
            rotation = random.randint(0, 4)
            model_input_tensor = np.rot90(model_input_tensor, k=rotation, axes=(1,2))
        return (model_input_tensor.copy(), weights, cell_index)

    def __len__(self):
        return len(self.cell_indices)

    def output_dataset(self, output_dir):
        """
        Write the dataset to disk
        Create a folder for each cell containing images of: dapi, membrane and spots
        """
        print(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for idx, cell_index in enumerate(self.cell_indices):
            cell_dir = os.path.join(output_dir, str(cell_index))
            if not os.path.exists(cell_dir):
                os.makedirs(cell_dir)

            dapi_crop = self.dapi_crop_list[idx]
            membrane_crop = self.membrane_crop_list[idx]

            # Save overlay image
            sparse_input_array = self.cell_transcripts[idx]
            cell_transcripts = sparse_input_array.todense()
            projected_cell_transcripts = np.argmax(cell_transcripts, axis=0)
            cell_image_array = (dapi_crop + membrane_crop) * 255
            plt.figure()
            plt.imshow(cell_image_array, cmap='gray')
            plt.imshow(projected_cell_transcripts, alpha=0.2)
            plt.savefig(os.path.join(cell_dir, 'cell_overlay.png'))
            plt.close()

            imsave(os.path.join(cell_dir, 'dapi.tif'), dapi_crop)
            imsave(os.path.join(cell_dir, 'membrane.tif'), membrane_crop)
