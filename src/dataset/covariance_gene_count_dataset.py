import numpy as np
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset
import torch

class GeneCountDataset(Dataset):
    def __init__(self, cell_meta_df, gene_counts_df):
        self.cell_meta_df = cell_meta_df

        # Build the nearesr neighbors graph
        nn_graph_constructor = NearestNeighbors(n_neighbors=4, radius=0.4)
        nn_graph = nn_graph_constructor.fit(cell_meta_df[['global_x', 'global_y']].values)
        _, nn_indices = nn_graph.kneighbors(cell_meta_df[['global_x', 'global_y']].values)

        # Aggregate the covariance matrix of the nearest neighbor gene counts for each cell
        gene_counts_conv_list = []
        for idx, row in cell_meta_df.iterrows():
            # Get the nearest neighbors for each cell
            nearest_neighbors = cell_meta_df.loc[nn_indices[idx]]['cell_index']
            gene_counts = gene_counts_df.iloc[idx]

            # Retreive the gene counts of all th enearest neighbors
            nn_gene_counts = gene_counts_df.loc[nearest_neighbors].values
            # Compute the covariance matrix of the gene counts
            nn_gene_counts_cov = np.cov(nn_gene_counts.transpose()).flatten()

            gene_counts_conv_list.append(np.concatenate((gene_counts, nn_gene_counts_cov), axis=None))
        
        self.gene_counts_conv_list = gene_counts_conv_list

    def __getitem__(self, index):
        cell_index = self.cell_meta_df.loc[index]['cell_index']
        gene_count_covariance = self.gene_counts_conv_list[index]

        return torch.tensor(gene_count_covariance, dtype=torch.float32), cell_index

    def __len__(self):
        return len(self.cell_meta_df)
