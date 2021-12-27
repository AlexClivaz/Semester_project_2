import os
import mne
import pyedflib
import numpy as np

bad_eeg = []
base = './data/v1.5.2/01_tcp_ar/'
for file in os.listdir(base):
    a_base = os.path.join(base, file)
    for files in os.listdir(a_base):
        b_base = os.path.join(a_base, files)
        for files_2 in os.listdir(b_base):
            c_base = os.path.join(b_base, files_2)
            for edf_file in os.listdir(c_base):
                if edf_file[-3:] == 'edf':
                    edf_path = os.path.join(c_base, edf_file)
                    #f = mne.io.read_raw(edf_path)
                    f = pyedflib.EdfReader(edf_path)
                    f.file_info_long()
                    
print(bad_eeg)