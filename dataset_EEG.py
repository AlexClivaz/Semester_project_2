from typing import Tuple
import torch
import os
import pickle
import random
import pywt
import numpy as np
import pandas as pd
import scipy.fftpack as t
import scipy.signal as sig
import matplotlib.pyplot as plt
from torch_geometric import utils
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader

NB_NEIG = 5
FS = 250
interest_widths = np.array([5.75, 10, 14.5])
Truc = 5
General = ['GNSZ', 'ABSZ', 'TNSZ']
Focal = ['FNSZ', 'CPSZ']

func_corr = lambda i, j, x: np.abs(np.correlate(x[i,:], x[j,:]))
func_vect = np.vectorize(func_corr, excluded=['x'])
#func_cwt = lambda i, x: x[i,:] #np.abs(sig.cwt(x[i,:], sig.ricker,widths = np.array([10])))
#cwt_vect = np.vectorize(func_cwt, excluded=['x'])

# Create edge between electrodes using Euclidian distances between them
def create_edge_index_dist():
    weight = pd.read_csv('edge.csv', header = None)
    weight = np.array(weight)
    indices = np.diag_indices(weight.shape[0])
    weight[indices[0], indices[1]] = np.zeros(weight.shape[0])
    neig_val = np.sort(weight,axis=1)[:,-NB_NEIG]
    neig_val = neig_val.reshape((-1,1))
    weight[weight<neig_val] = 0
    edge_weight = torch.from_numpy(weight[weight>0])
    index = np.nonzero(weight)
    edge_index = torch.cat((torch.from_numpy(index[0]).unsqueeze(0),torch.from_numpy(index[1]).unsqueeze(0)),0)
    return edge_index, edge_weight


# Create edge between electrodes using correlation between them
def create_edge_index_corr(x):
    N = x.shape[0]
    corr = np.fromfunction(func_vect, shape=(N,N), dtype=int, x=x)
    neig_val = np.sort(corr,axis=1)[:,-NB_NEIG]
    neig_val = neig_val.reshape((-1,1))
    corr[corr<neig_val] = 0
    edge_weight = torch.from_numpy(corr[corr>0])
    edge_weight /= np.linalg.norm(edge_weight)
    index = np.nonzero(corr)
    edge_index = torch.cat((torch.from_numpy(index[0]).unsqueeze(0),torch.from_numpy(index[1]).unsqueeze(0)),0)
    return edge_index, edge_weight

# Extract interesting features from time domain
def feature_extractor_time(x):
    x = torch.from_numpy(x)
    avg = torch.mean(x, dim=1).reshape(-1,1)
    rect_avg = torch.mean(torch.abs(x), dim=1).reshape(-1,1)
    peak2peak = (torch.max(x,dim=1)[0] - torch.min(x,dim=1)[0]).reshape(-1,1)
    std = torch.std(x, dim=1).reshape(-1,1)
    diffs = x - avg
    zscores = diffs / std
    skews = torch.mean(torch.pow(zscores, 3.0), dim=1).reshape(-1,1)
    kurtoses = (torch.mean(torch.pow(zscores, 4.0), dim=1) - 3.0).reshape(-1,1)
    return torch.cat((avg, rect_avg, peak2peak, std, skews, kurtoses), dim=1)

def concat_PSD(f, PSD):
    f_4 = f<4
    f_8 = f<8
    f_16 = f<16
    f_32 = f<32
    f_56 = f<56

    delta = torch.from_numpy(np.sum(PSD[:,f_4], axis=1)).reshape(-1,1)
    theta = torch.from_numpy(np.sum(PSD[:,~f_4*f_8], axis=1)).reshape(-1,1)
    alpha = torch.from_numpy(np.sum(PSD[:,~f_8*f_16], axis=1)).reshape(-1,1)
    beta = torch.from_numpy(np.sum(PSD[:,~f_16*f_32], axis=1)).reshape(-1,1)
    gamma = torch.from_numpy(np.sum(PSD[:,~f_32*f_56], axis=1)).reshape(-1,1)
    return torch.cat((delta, theta, alpha, beta, gamma), dim=1)

def extract_cwt(x, type='None'):
    global Truc
    CWT = np.zeros((x.shape[0], 1500))
    for i in range(x.shape[0]):
        cwt_tmp = sig.cwt(x[i,:], sig.ricker, widths=interest_widths)
        # absz = np.sum(np.abs(cwt_tmp[-3:,:]), axis=1)
        cwt_tmp = cwt_tmp[:,:].reshape(1,-1).squeeze()
        # cwt_tmp = np.concatenate((cwt_tmp, absz))
        CWT[i,:] = cwt_tmp
        
    # if np.random.rand(1) < 0.1 and Truc>0:
    #     Truc -= 1
    #     for i in range(x.shape[0]):
    #         test = sig.cwt(x[i,:], sig.ricker, widths=np.linspace(4, 25, 20))
    #         #print(np.linalg.norm(test))
    #         test *= np.abs(test)#np.linalg.norm(test)
    #         plt.subplot(4,5,i+1)
    #         plt.imshow(test, cmap='PRGn', aspect='auto',vmax=abs(test).max(), vmin=-abs(test).max())
    #     # plt.show()
    #     plt.savefig('./data/CWT_norm/'+type+str(Truc)+'_'+str(np.around(np.random.rand(),2))+'.png')

    return torch.from_numpy(CWT)

def extract_dwt(x):
    w = pywt.Wavelet('db4')
    cA, cD = pywt.dwt(x, w)
    dwt = np.concatenate((cA, cD), axis=1)
    
    return torch.from_numpy(dwt)

class EEG(InMemoryDataset):
    def __init__(self, root, seizure_types, wid, balanced='True', mode='FFT', type='Unique', edges='Corr', transform=None, pre_transform=None):

        self.root = root
        self.seizure_types = seizure_types
        self.balanced = True if balanced=='True' else False
        self.mode = mode
        self.type = type
        self.edges = edges
        self.weight = []
        self.train_mask = None
        self.wid = float(wid)
        print(self.mode)
        
        super(EEG, self).__init__(root, transform, pre_transform)

        self.transform, self.pre_transform = transform, pre_transform
        self.data, self.slices, self.weight, self.train_mask = torch.load(self.processed_paths[0])

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
        global Truc
        print("Creating dataset")
        data_list = []
        len_data = 500
        nb_data = 0
        # Create notch filter to remove 60Hz noise
        freq_noise, qual_factor = 60, 20
        b_notch, a_notch = sig.iirnotch(freq_noise, qual_factor, FS)
        sos = sig.butter(4, 100, 'lowpass', fs=FS, output='sos')

        if self.edges == 'Dist':
            edge_index, edge_weight = create_edge_index_dist()
            #edge_index = utils.to_undirected(edge_index.contiguous())

        # Read data into huge `Data` list.
        for type in self.seizure_types:
            data_type = []
            iter = 0
            for set in self.raw_paths :
                Truc = 5
                for dir_file in os.listdir(os.path.join(set, type)) :
                    if np.random.rand() < 0.7:
                        continue
                    iter += 1
                    file = os.path.join(set, type, dir_file)
                    eeg = pickle.load(open(file, 'rb'))
                    x = eeg.data
                    x = sig.filtfilt(b_notch, a_notch, x)
                    x = sig.sosfilt(sos, x)
                    x = (x-x.mean())/x.std()
                    if self.type == 'Unique':
                        y = self.seizure_types.index(eeg.seizure_type)
                    else:
                        y = 2 if eeg.seizure_type in General else 1 if eeg.seizure_type in Focal else 0
                    for k in range(len_data, x.shape[1], len_data) :
                        x_val = x[:,k-len_data:k]
                        if self.edges == 'Corr':
                            edge_index, edge_weight = create_edge_index_corr(x_val)
                            #edge_index, edge_weight = utils.remove_self_loops(edge_index, edge_weight)
                            #edge_index = utils.to_undirected(edge_index.contiguous())
                        # Time serie
                        if self.mode == 'Normal':
                            X = torch.from_numpy(x_val)
                        # Discrete Cosinus Transform
                        if self.mode == 'DCT':
                            X = torch.from_numpy(t.dct(x_val, norm='ortho'))
                        # Fast Fourier Transform
                        if self.mode == 'FFT':
                            X = torch.from_numpy(t.fft(x_val))
                            tfreq = t.fftfreq(len_data, 1/FS)
                            X = X.abs()[:,(tfreq>0)*(tfreq<58)]
                            X = (X-X.mean())/X.std()
                            # features = feature_extractor_time(x_val)
                            # X = torch.cat((features, X), dim=1)
                        if self.mode == 'Feature':
                            features = feature_extractor_time(x_val)
                            f, PSD = sig.periodogram(x_val, FS)
                            feature_freq = concat_PSD(f, PSD)
                            CWT = extract_cwt(x_val, type)
                            X = torch.cat((features, feature_freq, CWT), dim=1)
                        data = Data(x=X, edge_index = edge_index, edge_weight=edge_weight, y=y, id=int(eeg.patient_id))
                        data_type.append(data)

            # if BALANCED = True, remove data to have the same number of data per class
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

        # Calculate weight of each class and train/test dataset
        nb_data = 0
        if self.type == 'Unique':
            nb_type = len(self.seizure_types)
        else:
            nb_type = 3
        w = torch.zeros(nb_type, dtype=torch.double)
        for current_list in data_list:
            tmp = tmp+current_list
            nb_data += len(current_list)
            w[current_list[0].y] += len(current_list)
        w = w / nb_data
        self.weight = (1/w).clone().detach()

        self.data, self.slices = self.collate(tmp)

        for seiz in range(nb_type):
            #print(self.seizure_types[seiz])
            s_data = self.data.y==seiz
            unique_id, counts = self.data.id[s_data].unique(return_counts=True)
            #print(unique_id)
            _, ind = torch.sort(counts, descending=True)
            counts = counts[ind]
            unique_id = unique_id[ind]
            train_id = unique_id[0].unsqueeze(0)
            ratio =  counts[0]/len(self.data.id[s_data])
            i=1
            while ratio < 0.8:
                ratio += counts[i]/len(self.data.id[s_data])
                if ratio == 1:
                    break
                train_id = torch.cat((train_id, unique_id[i].unsqueeze(0)))
                i += 1
            if len(unique_id) == 1:
                train_ind_tmp = s_data[s_data] * (torch.rand(s_data[s_data].shape)<0.8)
            else:
                train_ind_tmp = torch.sum(torch.stack([(self.data.id[s_data] == i) for i in train_id]), dim=0, dtype=torch.bool)
            #print(train_id)
            print("train ratio : ", len(train_ind_tmp[train_ind_tmp])/len(train_ind_tmp))
            if seiz==0:
                train_ind = train_ind_tmp
            else:
                train_ind = torch.cat((train_ind, train_ind_tmp))
            
        self.train_mask = train_ind
        print("Saving dataset")
        torch.save((self.data, self.slices, self.weight, self.train_mask), self.processed_paths[0])
