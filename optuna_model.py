import os

import optuna
from optuna.trial import TrialState
import torch
import argparse
import os
import random
import collections
from sklearn import metrics
from torch.nn import Linear, ReLU, Dropout
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric import utils
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import torch_geometric.nn as nn
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import global_mean_pool, BatchNorm

from dataset_EEG import EEG
seizure_type_data = collections.namedtuple('seizure_type_data', ['patient_id','seizure_type', 'data'])
root = []
seizure_types = []

DEVICE = torch.device("cpu")
BATCH_SIZE = 128
DIR = os.getcwd()
EPOCHS = 30

def define_model(trial):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 1, 6)
    layers = []

    in_features = dataset.num_node_features
    out_features = trial.suggest_int("n_units", 4, 128)
    for i in range(n_layers):
        layers.append((SAGEConv(in_features, out_features), 'x, edge_index -> x'))
        layers.append((BatchNorm(out_features), 'x -> x'))
        layers.append(ReLU(inplace=True))
        in_features = out_features
        
    layers.append((global_mean_pool, 'x, batch -> x'))
    p = trial.suggest_float("dropout", 0.2, 0.7)
    layers.append((Dropout(p), 'x -> x'))
    layers.append(Linear(in_features, dataset.num_classes))
##    layers.append(nn.LogSoftmax(dim=1))
   
    return nn.Sequential('x, edge_index, batch', [*layers])

# Train classifier on train data
def train(model, loader, criterion, optimizer):
    model.train()
    for data in loader:  
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

def objective(trial):

    # Generate the model.
    model = define_model(trial).to(DEVICE)
    model = model.double()

    criterion = torch.nn.CrossEntropyLoss()

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Get training and testing dataset
    dataset = EEG(root, seizure_types)
    dataset = dataset.shuffle()
    dataset = dataset[:6000]
    train_dataset = dataset[:int(len(dataset) * 0.8)]
    test_dataset = dataset[int(len(dataset) * 0.8):]
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Training of the model.
    for epoch in range(EPOCHS):
        train(model, train_loader, criterion, optimizer)

        # Validation of the model.
        test_acc = test(model, test_loader)
        trial.report(test_acc, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    print("Accuracy : ", test_acc)
    return test_acc

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create graph and GNN from EEG samples')
    parser.add_argument('--data_dir', default='./data/v1.5.2', help='path to the dataset')
    parser.add_argument('--seizure_types',default=['FNSZ','GNSZ','CPSZ'], help="types of seizures")

    args = parser.parse_args()
    seizure_types = args.seizure_types
    data_dir = args.data_dir

    root = data_dir
    os.makedirs(root + "/processed", exist_ok=True)
    dataset = EEG(root, seizure_types)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
