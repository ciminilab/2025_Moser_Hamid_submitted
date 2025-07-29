import numpy as np
from scipy import signal

def get_cell_crop(cell_index, cell_meta_data_df, dapi_image, membrane_image, crop_size):
    current_cell = cell_meta_data_df.loc[cell_index]

    x_center, y_center = int(current_cell['x']), int(current_cell['y'])
    padding = int(crop_size / 2)
    x_min, y_min = max(0, x_center - padding), max(0, y_center - padding)
    x_min, y_min = min(x_min, dapi_image.shape[0] - crop_size), min(y_min, dapi_image.shape[1] - crop_size)
    x_max, y_max = x_min + crop_size, y_min + crop_size

    dapi_crop = dapi_image[x_min:x_max, y_min:y_max]
    membrane_crop = membrane_image[x_min:x_max, y_min:y_max]
    normalized_dapi_crop = (dapi_crop - np.min(dapi_crop)) / (np.max(dapi_crop) - np.min(dapi_crop))
    normalized_membrane_crop = (membrane_crop - np.min(membrane_crop)) / (np.max(membrane_crop) - np.min(membrane_crop))

    return normalized_dapi_crop, normalized_membrane_crop, x_min, y_min


def create_multichannel_representation(cell_transcripts_df, x_min, y_min, n_genes, padding, crop_size):
    # Remove transcripts outside the crop
    cell_transcripts_df = cell_transcripts_df[(cell_transcripts_df['x'] > x_min) & (cell_transcripts_df['y'] > y_min)]
    cell_transcripts_df = cell_transcripts_df[(cell_transcripts_df['x'] < (x_min + crop_size)) & (cell_transcripts_df['y'] < (y_min + crop_size))]

    x_coordinates = cell_transcripts_df['x'].astype(int) - x_min
    y_coordinates = cell_transcripts_df['y'].astype(int) - y_min
    barcode_ids = cell_transcripts_df['barcode_id']
   
    # Construct a binary 3D representation of the transcripts
    binary_transcripts = np.zeros((n_genes, crop_size, crop_size), dtype=np.uint8)
    binary_transcripts[barcode_ids, x_coordinates, y_coordinates] = 1

    # The weight of each class (i.e. gene) corresponds to the number of expressed transcripts
    gene_counts = np.sum(binary_transcripts, axis=(1, 2))
    max_gene_count = np.max(gene_counts)
    weights = np.array([0 if gene_count==0 else (1 / gene_count) for gene_count in gene_counts])
    weights = weights * max_gene_count

    # Define the kernel
    kernel = np.ones((padding+2, padding+2))
    # Run 2D convolution on each gene plane
    output_array = np.array([signal.convolve2d(gene_transcripts, kernel, boundary='symm', mode='same') for gene_transcripts in binary_transcripts])

    # Make the array binary
    return (output_array != 0).astype(np.uint8), weights
