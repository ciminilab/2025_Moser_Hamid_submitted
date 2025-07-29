from torch.utils.data import Dataset
import torch

class GeneCountDataset(Dataset):
    def __init__(self, cell_meta_df, gene_count_df):
        self.cell_meta_df = cell_meta_df
        self.gene_counts = gene_count_df

    def __getitem__(self, index):
        gene_count_sample = self.gene_counts.iloc[index].values
        cell_index = self.cell_meta_df.iloc[index]['cell_index']

        return (torch.tensor(gene_count_sample, dtype=torch.float32), cell_index)

    def __len__(self):
        return len(self.cell_meta_df)