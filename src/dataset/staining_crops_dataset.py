import random
from multiprocessing import Pool
from .utils import get_cell_crop
import numpy as np
from torch.utils.data import Dataset
import torch

def aggregate_get_cell_crops(cell_indices, cell_meta_df, dapi_image, membrane_image, crop_size):
    dapi_crop_list = []
    membrane_crop_list = []
    for cell_index in cell_indices:
        dapi_crop, membrane_crop, _, _ = get_cell_crop(cell_index, cell_meta_df, dapi_image, membrane_image, crop_size)

        dapi_crop_list.append(dapi_crop)
        membrane_crop_list.append(membrane_crop)

    return (cell_indices, dapi_crop_list, membrane_crop_list)


class StainingCropsDataset(Dataset):
    def __init__(self, cell_meta_data_df, dapi_stack, membrane_stack, crop_size, z_index, n_workers=2, apply_data_augmentation=False, tensor_type=torch.float32):
        self.apply_data_augmentation = apply_data_augmentation
        self.tensor_type = tensor_type

        dapi_image = dapi_stack[z_index].transpose()
        membrane_image = membrane_stack[z_index].transpose()

        # Compute the cell of all cells in the dataset
        cell_index_list = np.unique(cell_meta_data_df['cell_index'].values)
        # Split the list of cells into chunks for multiprocessing
        cell_index_split_list = np.array_split(cell_index_list, n_workers)
        cell_meta_data_df.set_index("cell_index", inplace=True)

        with Pool(processes=n_workers) as pool:
            args = [[cell_index_sublist, cell_meta_data_df.loc[cell_index_sublist], dapi_image, membrane_image, crop_size] for cell_index_sublist in cell_index_split_list]
            results = pool.starmap(aggregate_get_cell_crops, args)

        cell_indices = []
        dapi_stack_crop_list = []
        membrane_stack_crop_list = []
        for process_result in results:
            cell_indices.extend(process_result[0])
            dapi_stack_crop_list.extend(process_result[1])
            membrane_stack_crop_list.extend(process_result[2])

        self.cell_indicies = cell_indices
        self.dapi_crop_list = dapi_stack_crop_list
        self.membrane_crop_list = membrane_stack_crop_list

    def __getitem__(self, index):
        rotation = random.randint(0, 4)
        dapi_crop = self.dapi_crop_list[index]
        membrane_crop = self.membrane_crop_list[index]
        model_input_np = np.stack((dapi_crop, membrane_crop))
        if self.apply_data_augmentation:
            model_input_np = np.rot90(model_input_np, k=rotation, axes=(1,2))

        cell_index = self.cell_indicies[index]
        weights = torch.ones((1, 1), dtype=self.tensor_type)

        model_input_tensor = torch.tensor(model_input_np.copy(), dtype=self.tensor_type)

        return (model_input_tensor, weights, cell_index)

    def __len__(self):
        return len(self.membrane_crop_list)
