import torch
import os
import pickle
import random
import numpy as np
import pandas as pd
from torch_geometric import utils
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader

THRESH_DIST = 0.4

# Create edge between electrodes using Euclidian distances between them
def create_edge_index():
    weight = pd.read_csv('edge.csv', header = None)
    weight = torch.tensor(weight.values, dtype=torch.float64)
    indices = np.diag_indices(weight.shape[0])
    weight[indices[0], indices[1]] = torch.zeros(weight.shape[0], dtype=torch.float64)
    weight[weight < THRESH_DIST] = 0
    indice = torch.arange(0, weight.shape[0])
    edge_ind1 = torch.combinations(indice, with_replacement=True)
    edge_ind2 = torch.combinations(torch.flip(indice, [0]))
    edge_index = torch.cat((edge_ind1, edge_ind2))
    edge_tmp = edge_index[edge_index[:,1].sort()[1]]
    edge_index = edge_tmp[edge_tmp[:,0].sort(stable = True)[1]]
    return edge_index[weight.reshape(weight.numel()) > 0]

class EEG(InMemoryDataset):
    def __init__(self, root, seizure_types, balanced, weight = None, transform=None, pre_transform=None):

        self.root = root
        self.seizure_types = seizure_types
        self.balanced = balanced
        self.weight = []
        
        super(EEG, self).__init__(root, transform, pre_transform)

        self.transform, self.pre_transform = transform, pre_transform
        self.data, self.slices = torch.load(self.processed_paths[0])

        for s_type in range(len(seizure_types)):
            self.weight.append(len(self.data.y[self.data.y == s_type]))
        w = torch.tensor(self.weight, dtype = torch.double) / len(self.data.y)
        self.weight = 1/w

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @property
    def processed_file_names(self):
        return 'processed_data.pt'

    def download(self):
        raise NotImplementedError('No download')

    def process(self):
        data_list = []
        len_data = 250
        nb_data = 0
        edge_index = create_edge_index()
        # Read data into huge `Data` list.
        for set in self.raw_paths :
            for type in self.seizure_types:
                print(type)
                data_type = []
                for dir_file in os.listdir(os.path.join(set, type)) :
                    file = os.path.join(set, type, dir_file)
                    eeg = pickle.load(open(file, 'rb'))
                    x = torch.tensor(eeg.data)[0:19,:]
                    y = self.seizure_types.index(eeg.seizure_type)
                    for k in range(len_data, x.shape[1], len_data) :
                        data = Data(x=x[:,k-len_data:k], edge_index = edge_index.t().contiguous(), y=y)
                        data_type.append(data)
                if self.balanced == True:
                    if nb_data == 0:
                        nb_data = len(data_type)
                        data_list.append(data_type)
                    if len(data_type) > nb_data:
                        nb_del = len(data_type) - nb_data
                        del_elem = random.sample(data_type, k=nb_del)
                        data_type = [elem for elem in data_type if elem not in del_elem]
                        data_list.append(data_type)
                    if len(data_type) < nb_data:
                        nb_del = nb_data - len(data_type)
                        nb_data = len(data_type)
                        for current_list in data_list:
                            ind = data_list.index(current_list)
                            del_elem = random.sample(current_list, k=nb_del)
                            current_list = [elem for elem in current_list if elem not in del_elem]
                            data_list[ind] = current_list
                        data_list.append(data_type)
                else:
                    data_list.append(data_type)
        tmp = []
        nb_data = 0
        for current_list in data_list:
            tmp = tmp+current_list
            nb_data += len(current_list)
            self.weight.append(len(current_list))
        w = np.array(self.weight) / nb_data
        self.weight = list(w)
        self.data, self.slices = self.collate(tmp)
        torch.save((self.data, self.slices), self.processed_paths[0])
