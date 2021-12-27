from scipy.sparse import data
import torch
import argparse
import os
import random
import collections
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from torch.nn import Linear, Dropout, ReLU, LeakyReLU
from torch_geometric import utils
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch_geometric.nn as nn
from torch_geometric.nn import GATConv, GATv2Conv, SAGEConv, GCNConv, ChebConv, JumpingKnowledge, GNNExplainer
from torch_geometric.nn import global_mean_pool, BatchNorm, LayerNorm
from captum.attr import IntegratedGradients
from dataset_EEG import EEG

np.set_printoptions(precision=4, suppress=True)

TRAIN = True
BATCH_SIZE = 64
DROPOUT = 0.5
EPOCH = 80
LR = 0.00045
HIDDEN_CHANNELS = 48
NB_LAYERS = 4
K = 5
model_name = 'ChebNorm_'+str(NB_LAYERS)+'_'+str(HIDDEN_CHANNELS)+'_k_'+str(K)
# model_name = 'GCN_'+str(NB_LAYERS)+'_'+str(HIDDEN_CHANNELS)

seizure_type_data = collections.namedtuple('seizure_type_data', ['patient_id','seizure_type', 'data'])

# Create a model using sequential module of pytorch
def define_model():
    layers = []

    # Make two copies of the normalized data
    in_features = dataset.num_node_features
    p = DROPOUT
    #layers.append((BatchNorm(in_features), 'x -> x'))
    
    #layers.append((lambda x : [x, x], 'x -> x1, x2'))

    # Add convolutional layers
    for i in range(NB_LAYERS):
        first_desc = 'x'+str(i)+', edge_index, edge_weight -> x'+str(i)
        sec_desc = 'x'+str(i)+', batch -> x'+str(i+1)
        layers.append((ChebConv(in_features, HIDDEN_CHANNELS, K), first_desc))
        layers.append((LayerNorm(HIDDEN_CHANNELS), sec_desc))
        layers.append(ReLU(inplace=True))
        in_features = HIDDEN_CHANNELS

    # Merge the data
    layers.append((lambda x1, x2, x3, x4: [x1, x2, x3, x4], 'x1, x2, x3, x4 -> xs'))
    layers.append((JumpingKnowledge("cat", in_features, num_layers=NB_LAYERS), 'xs -> x'))

    # Add pooling, dropout and linear layers as the final classifier
    layers.append((global_mean_pool, 'x, batch -> x'))
    # layers.append((lambda xs : print(xs.shape), 'x -> x'))

    layers.append((Linear(NB_LAYERS*in_features, NB_LAYERS*in_features), 'x -> x'))
    layers.append((Dropout(p), 'x -> x'))
    layers.append(Linear(NB_LAYERS*in_features, dataset.num_classes))
   
    return nn.Sequential('x0, edge_index, edge_weight, batch', [*layers])

def try_layer():
    layers = []

    in_features = dataset.num_node_features
    layers.append((LayerNorm(in_features), 'x -> x'))
   
    return nn.Sequential('x', [*layers])

def try_layer2():
    layers = []

    in_features = dataset.num_node_features
    layers.append((BatchNorm(in_features), 'x -> x'))
   
    return nn.Sequential('x', [*layers])

# Train classifier on train data
def train(model):
    model.train()
    t_loss = 0
    for data in train_loader:  
        out = model(data.x, data.edge_index, data.edge_weight, batch=data.batch)
        loss = criterion(out, data.y) 
        loss.backward()
        optimizer.step() 
        optimizer.zero_grad()
        t_loss += loss.item()
    return t_loss/len(train_loader)

# Test classifier on data and return accuracy
def test(model, loader):
    model.eval()
    correct = 0
    for data in loader: 
        out = model(data.x, data.edge_index, data.edge_weight, data.batch)  
        pred = out.argmax(dim=1) 
        correct += int((pred == data.y).sum()) 
    return correct / len(loader.dataset)

# Return the predicted class from all the loader
def get_prediction(model, loader):
    model.eval()
    for step, data in enumerate(loader):
        if step == 0:
            out = model(data.x, data.edge_index, data.edge_weight, data.batch)  
            pred = out.argmax(dim=1)
            true = data.y
        else:
            out = model(data.x, data.edge_index, data.edge_weight, data.batch)  
            pred = torch.cat((pred, out.argmax(dim=1)))
            true = torch.cat((true, data.y))
    return true, pred

# Define forward function for model explanation
def model_forward(node_mask, data):
    batch = torch.zeros(node_mask.shape[0]*data.x.shape[0], dtype=int)
    edge_index = data.edge_index.detach().repeat(1,node_mask.shape[0])
    x_mask = torch.cat([(node_mask[i,:]*data.x.t()).t() for i in range(node_mask.shape[0])], 0)
    out = model(x_mask, edge_index, batch)
    return out

# Return node mask containing node importance
def explain():
    data = dataset[1]
    input_mask = torch.ones(data.x.shape[0]).unsqueeze(0).requires_grad_(True)
    ig = IntegratedGradients(model_forward)
    mask = ig.attribute(input_mask,target=0,additional_forward_args=(data,), n_steps=100, internal_batch_size=data.x.shape[0])
    node_mask = np.abs(mask.cpu().detach().numpy())
    if node_mask.max() > 0:  # avoid division by zero
        node_mask = node_mask / node_mask.max()
    print(node_mask)

# Get explanation from GNN Explainer
def gnn_explain():
    explainer = GNNExplainer(model, epochs=200, return_type='log_prob', feat_mask_type = 'scalar')
    data = dataset[1]
    x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
    node_feat_mask, edge_mask = explainer.explain_graph(x, edge_index, edge_weight)
    node_feat_mask = node_feat_mask.detach().numpy()
    print(node_feat_mask/node_feat_mask.max())
    #ax, G = explainer.visualize_subgraph(-1, edge_index, edge_mask, y=data.y, threshold=0.9)
    #plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create graph and GNN from EEG samples')
    parser.add_argument('--data_dir', default='./data/v1.5.2', help='path to the dataset')
    parser.add_argument('--seizure_types',default=['FNSZ','GNSZ'], help="types of seizures")
    parser.add_argument('--dataset_args',nargs="*",default=['2','False','FFT','Unique', 'Corr'], help="dataset characteristics")
    args = parser.parse_args()
    seizure_types = args.seizure_types
    data_dir = args.data_dir
    dataset_args = args.dataset_args

    root = data_dir
    os.makedirs(root + "/processed", exist_ok=True)
    dataset = EEG(root, seizure_types, dataset_args[0])
    print(dataset[0].x.shape)

    # data_t = dataset[list((np.random.rand(5)*1000).astype(int))]
    # model_t = try_layer().double()
    # model_b = try_layer2().double()
    # load_t = DataLoader(data_t, batch_size=5, shuffle=False)
    # for data in load_t:
    #     plt.plot(data.x[:,80].detach().numpy(), 'r')
    #     out = model_t(data.x)
    #     #plt.plot(out[:,20].detach().numpy(), 'g')
    #     out_b = model_b(data.x)
    #     plt.plot(out_b[:,80].detach().numpy(), 'y')
    # plt.show()
    # exit()

    # Split dataset in training and testing
    train_dataset = dataset[dataset.train_mask]
    test_dataset = dataset[~dataset.train_mask]
    train_dataset = train_dataset.shuffle()
    test_dataset = test_dataset.shuffle()

    train_dataset = train_dataset[:100*BATCH_SIZE]
    test_dataset = test_dataset[:20*BATCH_SIZE]
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    k_fold = 3
    conf_matrix = []
    loss_matrix = []
    f1_matrix = []
    f1_weight = []
    f1_macro = []

    if TRAIN == False:
        model = define_model().double()
        model.load_state_dict(torch.load('./model/'+model_name+'.pt'))
        explain()
        gnn_explain()
        exit()
    
    for i in range(k_fold):
        printstr = "Repetition number: " + str(i+1) + "  over " + str(k_fold)
        print(printstr)

        # Create different dataset, train, val
        nb_train = len(train_dataset)
        nb_train_frac = int(nb_train/k_fold)
        val = torch.zeros((nb_train), dtype=torch.bool)
        val[i*nb_train_frac:(i+1)*nb_train_frac] = True
        train_data = train_dataset[~val]
        val_data = train_dataset[val]
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
        
        # Define model, criterion and optimizer
        model = define_model().double()
        criterion = torch.nn.CrossEntropyLoss(weight = dataset.weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        l_loss = []
        l_f1 = []

        nb_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Number of parameters: {nb_param}')

        # Train model on train dataset and evaluate on the validation dataset
        # (Since the same patients are both in train and validation, results are far
        # better than on test dataset)
        for epoch in range(EPOCH):
            loss = train(model)
            y_true, y_pred = get_prediction(model, val_loader)
            f1_score = metrics.f1_score(y_true, y_pred,average='weighted')
            print(f'Epoch: {epoch:03d}, f1 score on validation data: {f1_score}')
            l_loss.append(loss)
            l_f1.append(f1_score)

        # Save model weights
        # torch.save(model.state_dict(), './model/'+model_name+'.pt')

        # Evaluate model on test dataset, metrics are f1 scores
        y_true, y_pred = get_prediction(model, test_loader)
        conf_matrix.append(metrics.confusion_matrix(y_true, y_pred, normalize='true'))
        f1_weight.append(metrics.f1_score(y_true, y_pred,average='weighted'))
        f1_macro.append(metrics.f1_score(y_true, y_pred,average='macro'))
        loss_matrix.append(l_loss)
        f1_matrix.append(l_f1)
        print(metrics.classification_report(y_true, y_pred, target_names=seizure_types))
        print("\n")

        # file = open("./data/Results/"+model_name+".txt","a")
        # file.write(str(metrics.confusion_matrix(y_true, y_pred, normalize='true')))
        # file.write("\n\n\n")
        # file.close()

    # Print final results
    # plt.figure(1)
    # plt.subplot(211)
    # plt.plot(np.mean(loss_matrix, axis=0))
    # plt.subplot(212)
    # plt.plot(np.mean(f1_matrix, axis=0))
    # plt.show()

    # Save results
    file = open("./data/ResultsFFT_NoTime/"+model_name+".txt","a")
    file.write("Final score (mean) for "+dataset_args[0]+"\n\n""Confusion Matrix\n")
    file.write(str(seizure_types)+"\n")
    file.write(str(np.around(np.mean(conf_matrix, axis=0), 4)))
    file.write("\n\nF1 score weighted\n"+str(np.mean(f1_weight)))
    file.write("\nF1 score macro\n"+str(np.mean(f1_macro))+"\n")
    file.close()
