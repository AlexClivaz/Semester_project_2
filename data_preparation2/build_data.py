import os, shutil
import sys
print('\nCurrent working path is %s' % str(os.getcwd()),'\n')
sys.path.insert(0, os.getcwd())

import platform
import argparse
import pandas as pd
import numpy as np
import math
import collections
from tabulate import tabulate
import pyedflib
import re
from scipy.signal import resample
import pickle
import h5py
import progressbar
from time import sleep

# Run the following in \EEG_seizure
# python .\data_preparation\build_data.py
# To specify a different base directory, either modify the parser in __main__ or
# specifiy python ./data_preparation/build_data.py --base_dir [your_base_directory]

"""
This file first generates the data dict out of the excel file (for both dev and train sets using the multiple sheets
from the excel file) and loads all the .edf files of the dataset into pickle files (that will be located in 
multiple folders in data/v1.5.2/input)
"""

parameters = pd.read_csv('data_preparation2/parameters.csv', index_col=['parameter'])
seizure_type_data = collections.namedtuple('seizure_type_data', ['patient_id','seizure_type', 'data'])

def generate_data_dict(base_dir, xlsx_file_name, sheet_name, seizure_types):

    seizure_info = collections.namedtuple('seizure_info', ['patient_id','filename', 'start_time', 'end_time'])

    excel_file = os.path.join(xlsx_file_name)
    data = pd.read_excel(excel_file, sheet_name=sheet_name)
    data = data.iloc[1:] # remove first row

    col_l_file_name = data.columns[11]
    col_m_start = data.columns[12]
    col_n_stop = data.columns[13]
    col_o_szr_type = data.columns[14]
    train_files = data[[col_l_file_name, col_m_start,col_n_stop,col_o_szr_type]]

    # Save all the file names in a Series before removing the NAs and remove the duplicates
    general_files = train_files.iloc[:,0] 
    general_files = general_files.drop_duplicates()

    train_files = np.array(train_files.dropna())

    type_dict = collections.defaultdict(list)
    bckg_ids = [] # List containing the indices of the patients that have been experiencing a seizure of type in seizure_types

    for type in seizure_types :
        
        type_files = np.array([file for file in train_files if type in file])

        for item in type_files :
            a = item[0].split('/')
            patient_id = a[4]

            v = seizure_info(patient_id = patient_id, filename = item[0], start_time=item[1], end_time=item[2])
            type_dict[type].append(v)

            # Append all the patient numbers that experienced a seizure (no matter the type)
            if patient_id not in bckg_ids : bckg_ids.append(patient_id)

    # To retrieve the background samples, we need to access the ref_dev/train.txt files
    ref_txt = pd.read_csv(os.path.join(base_dir,"v1.5.2/_DOCS/ref_"+sheet_name+".txt"), sep=" ", header=None)
    ref_txt.columns = ["file_name", "t_start", "t_end", "seiz", "unit"]

    # For ref_train.txt, we need to re-format the first column as the filename is including some of the '.tse' ending
    if sheet_name == 'train' :
        ref_txt['file_name'] = ref_txt['file_name'].apply(lambda f : "00"+f[:-2] if f[-2:]==".t" else f)

    # To format the file names as in ref_dev/train.txt
    cut_files = general_files.apply(lambda n : n.split('/')[-1][:-4])

    # Extract all the background samples of the retrieved patient numbers (even when no seizure was recorded)
    for patient_nb in bckg_ids :
        # Retrieve all the files associated with this patient number from the .txt file
        bckg_data = ref_txt[(ref_txt['file_name'].apply(lambda n : n[:8]) == patient_nb) & (ref_txt['seiz'] == 'bckg')][['file_name','t_start','t_end','seiz']]

        # Retrieve the full path of file_name
        bckg_data['file_name'] = bckg_data['file_name'].apply(lambda y : general_files[y == cut_files].values[0])

        bckg_info_list = bckg_data.apply(lambda x : seizure_info(patient_id = patient_nb, filename = x.file_name, start_time = x.t_start, end_time = x.t_end), axis=1).to_list()
        # Append to the background dict
        type_dict['BG'].extend(bckg_info_list)

    return type_dict

def print_type_information(data_dict):

    l = []
    for szr_type, szr_info_list in data_dict.items():
        # how many different patient id for seizure K?
        patient_id_list = [szr_info.patient_id for szr_info in szr_info_list]
        unique_patient_id_list,counts = np.unique(patient_id_list,return_counts=True)

        dur_list = [szr_info.end_time-szr_info.start_time for szr_info in szr_info_list]
        total_dur = sum(dur_list)
        # l.append([szr_type, str(len(szr_info_list)), str(len(unique_patient_id_list)), str(total_dur)])
        l.append([szr_type, (len(szr_info_list)), (len(unique_patient_id_list)), (total_dur)])

        #  numpy.asarray((unique, counts)).T
        '''
        if szr_type=='TNSZ':
            print('TNSZ Patient ID list:')
            print(np.asarray((unique_patient_id_list, counts)).T)
        if szr_type=='SPSZ':
            print('SPSZ Patient ID list:')
            print(np.asarray((unique_patient_id_list, counts)).T)
        '''

    sorted_by_szr_num = sorted(l, key=lambda tup: tup[1], reverse=True)
    print(tabulate(sorted_by_szr_num, headers=['Seizure Type', 'Seizure Num','Patient Num','Duration(Sec)']))

def merge_train_test(train_data_dict, dev_test_data_dict):

    merged_dict = collections.defaultdict(list)
    for item in train_data_dict:
        merged_dict[item] = train_data_dict[item] + dev_test_data_dict[item]

    return merged_dict

def extract_signal(f, signal_labels, electrode_name, start, stop):

    tuh_label = [s for s in signal_labels if 'EEG ' + electrode_name + '-' in s]

    if len(tuh_label) > 1:
        print(tuh_label)
        exit('Multiple electrodes found with the same string! Abort')

    channel = signal_labels.index(tuh_label[0])
    signal = np.array(f.readSignal(channel))

    start, stop = float(start), float(stop)
    original_sample_frequency = f.getSampleFrequency(channel)
    original_start_index = int(np.floor(start * float(original_sample_frequency)))
    original_stop_index = int(np.floor(stop * float(original_sample_frequency)))

    seizure_signal = signal[original_start_index:original_stop_index]

    new_sample_frequency = int(parameters.loc['sampling_frequency']['value'])
    new_num_time_points = int(np.floor((stop - start) * new_sample_frequency))
    seizure_signal_resampled = resample(seizure_signal, new_num_time_points)

    return seizure_signal_resampled

def read_edfs_and_extract(edf_path, edf_start, edf_stop):

    f = pyedflib.EdfReader(edf_path)

    montage = str(parameters.loc['montage']['value'])
    montage_list = re.split(';', montage)
    signal_labels = f.getSignalLabels()
    x_data = []

    for electrode in montage_list:
        extracted_signal_from_electrode = extract_signal(f, signal_labels, electrode_name=electrode, start=edf_start, stop=edf_stop)
        x_data.append(extracted_signal_from_electrode)

    f._close()
    del f

    x_data = np.array(x_data)

    return x_data

def load_edf_extract_seizures(base_dir, save_data_dir, data_dict):

    seizure_data_dict = collections.defaultdict(list)

    bar = progressbar.ProgressBar(maxval=sum(len(v) for k, v in data_dict.items()),
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    for seizure_type, seizures in data_dict.items():
        seizure_data_dir = os.path.join(save_data_dir,seizure_type)
        count = 0
        for seizure in seizures :
            rel_file_location = seizure.filename.replace('.tse', '.edf').replace('./', 'edf/')
            patient_id = seizure.patient_id
            abs_file_location = os.path.join(base_dir, rel_file_location)
            try:
                temp = seizure_type_data(patient_id = patient_id, seizure_type = seizure_type, data = read_edfs_and_extract(abs_file_location, seizure.start_time, seizure.end_time))
                with open(os.path.join(seizure_data_dir, 'file_' + str(count) + '_pid_' + patient_id + '_type_' + seizure_type + '.pkl'), 'wb') as fseiz:
                    pickle.dump(temp, fseiz)
                count += 1
            except Exception as e:
                print(e)
                print(rel_file_location)
                exit()

            bar.update(count)
    bar.finish()

    return seizure_data_dict

# To convert raw edf data into pkl format raw data
def gen_raw_seizure_pkl(args, anno_file):
    
    base_dir = args.base_dir
    seizure_types = args.seizure_types # Add the background type

    # Create the directory where the raw seizure data will be extracted
    save_data_dir = os.path.join(base_dir, 'v1.5.2', 'raw_all')
    if not os.path.exists(save_data_dir):
        os.makedirs(save_data_dir)
    
    folders = ['dev','train']

    # Create the folders needed for storage
    for f_name in folders :
        dir = os.path.join(save_data_dir, f_name)
        if not os.path.exists(dir) :
            os.makedirs(dir)
        else :
            # If they already exist => empty the folders
            for dir_file in os.listdir(dir):
                file_path = os.path.join(dir, dir_file)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
        # Create separate folders for each seizure type
        for type in ['BG']+seizure_types :
            type_dir = os.path.join(dir, type)
            os.makedirs(type_dir)

    raw_data_base_dir = os.path.join(base_dir, 'v1.5.2')
    szr_annotation_file = os.path.join(raw_data_base_dir, '_DOCS', anno_file) # excel file with the seizure info for each recording

    # For training files
    # print('Parsing the seizures of the training set...\n')
    # train_data_dict = generate_data_dict(base_dir, szr_annotation_file, 'train', seizure_types)
    # print('Number of seizures by type in the training set...\n')
    # print_type_information(train_data_dict)
    # print('\n\n')
    # For dev files
    print('Parsing the seizures of the validation set...\n')
    dev_test_data_dict = generate_data_dict(base_dir, szr_annotation_file, 'dev', seizure_types)
    print('Number of seizures by type in the validation set...\n')
    print_type_information(dev_test_data_dict)
    print('\n\n')

    # Separately extract the seizures from the def and train edf files and save them
    print('Extracting seizures from the dev directory...\n')
    seizure_data_dict = load_edf_extract_seizures(raw_data_base_dir, os.path.join(save_data_dir, 'dev'), dev_test_data_dict)
    print('\n\n')
    # print('Extracting seizures from the train directory...\n')
    # seizure_data_dict = load_edf_extract_seizures(raw_data_base_dir, os.path.join(save_data_dir, 'train'), train_data_dict)
    # print('\n\n')

    print_type_information(seizure_data_dict)
    print('\n\n')

def main():

    parser = argparse.ArgumentParser(description='Build data for TUH EEG data')
    
    parser.add_argument('--base_dir', default='./data', help='path to seizure dataset')
    parser.add_argument('--seizure_types',default=['FNSZ','GNSZ','CPSZ','ABSZ','TNSZ'], help="types of seizures for the classification ('BG' not to be included)")

    args = parser.parse_args()
    parser.print_help()

    anno_file = 'seizures_v36r.xlsx'
    gen_raw_seizure_pkl(args, anno_file)
    

if __name__ == '__main__':
    main()
