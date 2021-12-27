import os

import optuna
from optuna.trial import TrialState
import torch
import argparse
import os
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
from torch_geometric.nn import SAGEConv, MFConv, GATConv, JumpingKnowledge
from torch_geometric.nn import global_mean_pool, BatchNorm

from dataset_EEG import EEG
seizure_type_data = collections.namedtuple('seizure_type_data', ['patient_id','seizure_type', 'data'])
root = []
seizure_types = []

DEVICE = torch.device("cpu")
BATCH_SIZE = 128
DIR = os.getcwd()
EPOCHS = 20
BALANCED = False

def define_model(trial):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 2, 6)
    layers = []

    layers.append((lambda x : [x, x], 'x -> x1, x2'))

    in_features = dataset.num_node_features
    out_features = trial.suggest_int("n_units", 64, 128, step=16)
    for i in range(n_layers):
        layers.append((MFConv(in_features, out_features), 'x1, edge_index -> x1'))
        layers.append((BatchNorm(out_features), 'x1 -> x1'))
        layers.append(ReLU(inplace=True))
        in_features = out_features
        
    in_features = dataset.num_node_features
    for i in range(n_layers):
        layers.append((GATConv(in_features, out_features), 'x2, edge_index -> x2'))
        layers.append((BatchNorm(out_features), 'x2 -> x2'))
        layers.append(ReLU(inplace=True))
        in_features = out_features

    # Merge the data
    layers.append((lambda x1, x2: [x1, x2], 'x1, x2 -> xs'))
    layers.append((JumpingKnowledge("lstm", in_features, num_layers=n_layers), 'xs -> x'))

    layers.append((global_mean_pool, 'x, batch -> x'))
    p = 0.5
    layers.append((Dropout(p), 'x -> x'))
    layers.append(Linear(in_features, dataset.num_classes))
   
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

def objective(trial):

    # Generate the model.
    model = define_model(trial).to(DEVICE)
    model = model.double()

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    criterion = torch.nn.CrossEntropyLoss(weight = dataset.weight)
    
    train_dataset = dataset[dataset.train_mask]
    test_dataset = dataset[~dataset.train_mask]
    train_dataset = train_dataset.shuffle()
    test_dataset = test_dataset.shuffle()

    train_data = train_dataset[:1000]
    val_data = train_dataset[1000:1200]
    test_data = test_dataset[:300]
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # Training of the model.
    for epoch in range(EPOCHS):
        train(model, train_loader, criterion, optimizer)

        # Validation of the model.
        y_true, y_pred = get_prediction(model, val_loader)
        f1 = metrics.f1_score(y_true, y_pred,average='weighted')
        trial.report(f1, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    y_true, y_pred = get_prediction(model, test_loader)
    f1 = metrics.f1_score(y_true, y_pred,average='weighted')
    return f1

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create graph and GNN from EEG samples')
    parser.add_argument('--data_dir', default='./data/v1.5.2', help='path to the dataset')
    parser.add_argument('--seizure_types',default=['FNSZ','GNSZ','CPSZ','TNSZ','ABSZ'], help="types of seizures")

    args = parser.parse_args()
    seizure_types = args.seizure_types
    data_dir = args.data_dir

    root = data_dir
    os.makedirs(root + "/processed", exist_ok=True)
    dataset = EEG(root, seizure_types, BALANCED)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=3)

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
