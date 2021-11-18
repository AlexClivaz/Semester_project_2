import torch
import argparse
import os
import random
import collections
from tqdm import tqdm
from sklearn import metrics
from torch.nn import Linear, Dropout, ReLU
from torch_geometric import utils
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch_geometric.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn import global_mean_pool, BatchNorm
from dataset_EEG import EEG

BALANCED = False
BATCH_SIZE = 64
DROPOUT = 0.5
EPOCH = 50
LR = 0.001
HIDDEN_CHANNELS = 32
NB_LAYERS = 6

seizure_type_data = collections.namedtuple('seizure_type_data', ['patient_id','seizure_type', 'data'])

# Create a model using sequential module of pytorch
def define_model():
    layers = []

    # Add convolutional layers
    in_features = dataset.num_node_features
    for i in range(NB_LAYERS):
        layers.append((SAGEConv(in_features, HIDDEN_CHANNELS), 'x, edge_index -> x'))
        layers.append((BatchNorm(HIDDEN_CHANNELS), 'x -> x'))
        layers.append(ReLU(inplace=True))
        in_features = HIDDEN_CHANNELS

    # Add pooling, dropout and linear layers
    layers.append((global_mean_pool, 'x, batch -> x'))
    p = DROPOUT
    layers.append((Dropout(p), 'x -> x'))
    layers.append(Linear(in_features, dataset.num_classes))
   
    return nn.Sequential('x, edge_index, batch', [*layers])

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create graph and GNN from EEG samples')
    parser.add_argument('--data_dir', default='./data/v1.5.2', help='path to the dataset')
    parser.add_argument('--seizure_types',default=['FNSZ','GNSZ','CPSZ'], help="types of seizures")

    args = parser.parse_args()
    seizure_types = args.seizure_types
    data_dir = args.data_dir

    root = data_dir
    os.makedirs(root + "/processed", exist_ok=True)
    dataset = EEG(root, seizure_types, BALANCED)

    nb_rep = 1
    
    for i in range(nb_rep):
        printstr = "Repetition number: " + str(i+1) + "  over " + str(nb_rep)
        print(printstr)

        # Split dataset in training and testing
        dataset = dataset.shuffle()
        dataset = dataset[:6000]
        train_dataset = dataset[:int(len(dataset) * 0.8)]
        test_dataset = dataset[int(len(dataset) * 0.8):]
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = define_model().double()
        criterion = torch.nn.CrossEntropyLoss(weight = dataset.weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        nb_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Number of parameters: {nb_param}')
        for epoch in range(EPOCH):
            train(model)
            train_acc = test(model, train_loader)
            test_acc = test(model, test_loader)
            print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        
        print(f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        y_true, y_pred = get_prediction(model, test_loader)
        print(metrics.classification_report(y_true, y_pred, target_names=seizure_types))
