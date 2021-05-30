import numpy as np
import pickle
import torch

def load_large_data(path, feature_file='/base_feature.npy', label_file='/base_label.txt'):
    '''Load and preprocess Large/Base class dataset .
    # Returns
        data: np.ndarray (cls, sample_num, feature_dim) i.e. (1000, 100, 4096)
        laebl: np.ndarray (cls, ) i.e. (1000, )
    '''
    feature, label = load_raw_large_data(path, feature_file, label_file)
    feature = torch.as_tensor(feature).reshape(1000, 100, -1)
    print('feature: ', feature.shape) # (1000, 100, 4096)

    label = torch.as_tensor(label).reshape(1000, -1)
    label = label[:, 0] # (1000, )
    print('label: ', label.shape)

    return feature, label

def load_raw_large_data(path, feature_file='/base_feature.npy', label_file='/base_label.txt'):
    '''Load and preprocess Large/Base class dataset .
    # Returns
        data: np.ndarray (cls * sample_num, feature_dim) i.e. (100000, 4096)
        laebl: np.ndarray (cls * sample_num, ) i.e. (100000, )
    '''
    feature_file = path + feature_file
    
    print('Reading feature file...')
    feature = np.load(feature_file) # (100000, 4096)
    # feature = np.zeros((100000, 4096), dtype=np.float32)
    print('Reading done!')
    feature = torch.as_tensor(feature)

    label_file = path + label_file
    with open(label_file, 'r+') as f:
        rawstr = f.read()
        label = rawstr.strip().split('\n')
    for i in range(len(label)):
        label[i] = int(label[i])
    label = torch.as_tensor(label)
    label = label - 1

    return feature, label