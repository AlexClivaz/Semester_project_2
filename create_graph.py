import torch
from torch_geometric.data import Data
import scipy.fftpack as t
import pandas as pd
import numpy as np
import re
import collections
import pickle
import mne

parameters = pd.read_csv('data_preparation2/parameters.csv', index_col=['parameter'])
seizure_type_data = collections.namedtuple('seizure_type_data', ['patient_id','seizure_type', 'data'])

# K nearest neighbors
##w_sorted, indices = torch.sort(weight, descending=True)
##w_sorted[:,6:] = 0
##weight = w_sorted.gather(1, indices.argsort(1))

##edge_index = edge_index[weight.reshape(weight.numel()) > 0]
##data = Data(x=x, edge_index = edge_index.t().contiguous())

##montage = str(parameters.loc['montage']['value'])
##montage_list = re.split(';', montage)
file = "data/v1.5.2/raw/dev/BG/file_0_pid_00000258_type_BG.pkl"
eeg = pickle.load(open(file, 'rb'))
x = eeg.data
##X = t.dct(x, norm='ortho')
##sort = np.sort(abs(X))
##thresh = sort[:,-1000].reshape((-1,1))
##X[abs(X) < thresh] = 0
##y = t.idct(X, norm='ortho')
