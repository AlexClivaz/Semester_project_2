import torch
import argparse
import os
import pickle
import collections
import random
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from torch.nn import Linear
from torch_geometric import utils
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn import global_mean_pool, BatchNorm

THRESH_DIST = 0.4

edge_index = []
data_dir = []
seizure_types = []
seizure_type_data = collections.namedtuple('seizure_type_data', ['patient_id','seizure_type', 'data'])

class SAGE(torch.nn.Module):
    def __init__(self, hidden_channels, third):
        super(SAGE, self).__init__()
        self.third = third
        self.conv1 = SAGEConv(dataset.num_node_features, hidden_channels)
        self.conv1_bn = BatchNorm(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2_bn = BatchNorm(hidden_channels)
        if third == True:
            self.conv3 = SAGEConv(hidden_channels, hidden_channels)
            self.conv3_bn = BatchNorm(hidden_channels)
            self.conv4 = SAGEConv(hidden_channels, hidden_channels)
            self.conv4_bn = BatchNorm(hidden_channels)
            self.conv5 = SAGEConv(hidden_channels, hidden_channels)
            self.conv5_bn = BatchNorm(hidden_channels)
            self.conv6 = SAGEConv(hidden_channels, hidden_channels)
            self.conv6_bn = BatchNorm(hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0),dtype=torch.int64)
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = self.conv1_bn(x)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = self.conv2_bn(x)
        x = x.relu()
        if self.third == True:
            x = self.conv3(x, edge_index)
            x = self.conv3_bn(x)
            x = x.relu()
            x = self.conv4(x, edge_index)
            x = self.conv4_bn(x)
            x = x.relu()
            x = self.conv5(x, edge_index)
            x = self.conv5_bn(x)
            x = x.relu()
            x = self.conv6(x, edge_index)
            x = self.conv6_bn(x)
            x = x.relu()

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, third):
        super(GCN, self).__init__()
        self.third = third
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv1_bn = BatchNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv2_bn = BatchNorm(hidden_channels)
        if third == True:
            self.conv3 = GCNConv(hidden_channels, hidden_channels)
            self.conv3_bn = BatchNorm(hidden_channels)
            self.conv4 = GCNConv(hidden_channels, hidden_channels)
            self.conv4_bn = BatchNorm(hidden_channels)
            self.conv5 = GCNConv(hidden_channels, hidden_channels)
            self.conv5_bn = BatchNorm(hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0),dtype=torch.int64)
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = self.conv1_bn(x)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = self.conv2_bn(x)
        x = x.relu()
        if self.third == True:
            x = self.conv3(x, edge_index)
            x = self.conv3_bn(x)
            x = x.relu()
            x = self.conv4(x, edge_index)
            x = self.conv4_bn(x)
            x = x.relu()
            x = self.conv5(x, edge_index)
            x = self.conv5_bn(x)
            x = x.relu()

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

# Train classifier on train data
def train(model):
    model.train()
    for data in train_loader:  
        out = model(data.x, data.edge_index, batch=data.batch)
        loss = criterion(out, data.y) 
        loss.backward() 
        optimizer.step() 
        optimizer.zero_grad() 

# Test classifier on data and return accuracy
def test(model, loader):
    model.eval()
    correct = 0
    for data in loader: 
        out = model(data.x, data.edge_index, data.batch)  
        pred = out.argmax(dim=1) 
        correct += int((pred == data.y).sum()) 
    return correct / len(loader.dataset) 

# Return the predicted class from all the loader
def get_prediction(model, loader):
    model.eval()
    for step, data in enumerate(loader):
        if step == 0:
            out = model(data.x, data.edge_index, data.batch)  
            pred = out.argmax(dim=1)
            true = data.y
        else:
            out = model(data.x, data.edge_index, data.batch)  
            pred = torch.cat((pred, out.argmax(dim=1)))
            true = torch.cat((true, data.y))
    return true, pred
    
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
    def __init__(self, root, transform=None, pre_transform=None):

        self.root = root
        
        super(EEG, self).__init__(root, transform, pre_transform)

        self.transform, self.pre_transform = transform, pre_transform
        self.data, self.slices = torch.load(self.processed_paths[0])

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
        first = True
        # Read data into huge `Data` list.
        for set in self.raw_paths :
            for type in seizure_types:
                print(type)
                data_type = []
                for dir_file in os.listdir(os.path.join(set, type)) :
                    file = os.path.join(set, type, dir_file)
                    eeg = pickle.load(open(file, 'rb'))
                    x = torch.tensor(eeg.data)[0:19,:]
                    y = seizure_types.index(eeg.seizure_type)
                    for k in range(len_data, x.shape[1], len_data) :
                        data = Data(x=x[:,k-len_data:k], edge_index = edge_index.t().contiguous(), y=y)
                        data_type.append(data)
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
        tmp = []
        for current_list in data_list:
            tmp = tmp+current_list
        self.data, self.slices = self.collate(tmp)
        torch.save((self.data, self.slices), self.processed_paths[0])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create graph and GNN from EEG samples')
    parser.add_argument('--data_dir', default='./data/v1.5.2', help='path to the dataset')
    parser.add_argument('--seizure_types',default=['FNSZ','GNSZ'], help="types of seizures")

    args = parser.parse_args()
    seizure_types = args.seizure_types
    data_dir = args.data_dir

    # Create edge index matrix using distances between electrodes
    edge_index = create_edge_index()
    
    root = data_dir
    os.makedirs(root + "/processed", exist_ok=True)
    dataset = EEG(root)

    nb_rep = 1
    result = torch.zeros(4, nb_rep, 2)
    nb_param =[0] * 4
    
    for i in range(nb_rep):
        printstr = "Repetition number: " + str(i+1) + "  over " + str(nb_rep)
        print(printstr)
        dataset = dataset.shuffle()
        dataset = dataset[:6000]
        
        train_dataset = dataset[:int(len(dataset) * 0.8)]
        test_dataset = dataset[int(len(dataset) * 0.8):]
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        models = []
        model1 = GCN(hidden_channels=64, third = False).double()
        models.append(model1)
        model2 = GCN(hidden_channels=32, third = True).double()
        models.append(model2)
        model3 = SAGE(hidden_channels=64, third = False).double()
        models.append(model3)
        model4 = SAGE(hidden_channels=32, third = True).double()
        models.append(model4)
        
        criterion = torch.nn.CrossEntropyLoss()

        for model in models:
            num_mod = models.index(model)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            nb_param[num_mod] = sum(p.numel() for p in model.parameters() if p.requires_grad)
            for epoch in range(1, 50):
                train(model)
                train_acc = test(model, train_loader)
                test_acc = test(model, test_loader)
##                print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
            
##            result[num_mod, i, 0] = test(model, train_loader)
##            result[num_mod, i, 1] = test(model, test_loader)
##            print(f'Model: {num_mod+1}, Train Acc: {result[num_mod, i, 0]:.4f}, Test Acc: {result[num_mod, i, 1]:.4f}')
            print(f'Model: {num_mod+1}, number of parameters: {nb_param[num_mod]}')
            y_true, y_pred = get_prediction(model, test_loader)
            print(metrics.classification_report(y_true, y_pred, target_names=seizure_types))
##    for i in range(len(models)):
##        print(f'Model: {i+1}, Train Acc: {torch.mean(result[i,:,0], 0):.4f}, Test Acc: {torch.mean(result[i,:,1], 0):.4f}')
##        print("Nb parameters: ", nb_param[i])
##        print()
