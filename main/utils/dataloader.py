import torch
from torch.utils.data import Dataset
from utils.dataloader import *
from utils.torch_utils import *
from utils.utils import *

from sklearn.datasets import load_breast_cancer, load_boston, load_diabetes, load_digits

import numpy as np
import pandas as pd 
import os

import sys

class TableDataset(Dataset):
    r"""
    Table dataset

    # Parameters
    X: input tableau data with missing values (float tensor type)
    M: mask (float tensor type)
    y: independent variable (target variable: long or float type)
    X_comp: complete matrix of X (true label of input)
    """ 
    def __init__(self, X, M, y, X_comp= None):
        super().__init__()
        self.X, self.y = X, y
        self.M = M
        self.X_comp= X_comp
        
    def __getitem__(self, index):
        if self.X_comp is None:
            return {"input":self.X[index], 
                    "mask":self.M[index],
                    "label": self.y[index],
                    "complete_input": None
                    }
        else: 
            return {"input":self.X[index], 
                    "mask":self.M[index],
                    "label": self.y[index],
                    "complete_input": self.X_comp[index]
                    }            

    def __len__(self):
        return len(self.X)

class TableDatasetVer2(Dataset):
    r"""
    Table dataset

    # Parameters
    y: independent variable (target variable: long or float type)
    X_comp: complete matrix of X (true label of input)
    """ 
    def __init__(self, x_complete, y):
        super().__init__()
        self.x_comp = x_complete
        self.y = y 
        
    def __getitem__(self, index):
        return {
            'complete_input': self.x_comp[index],
            'label': self.y[index]
        }        

    def __len__(self):
        return len(self.x_comp)

class BipartiteData(Dataset): 
    r"""
    Bipartite dataset

    # Parameters
    x: input tableau data with missing values (float tensor type)
    x_comp: complete matrix of x (float tensor type)
    y: independent variable (target variable: long or float type)
    edge_index: edge indices (coo-type)
    """ 
    def __init__(self, x, x_comp, y, edge_index= None): 
        
        super().__init__()
        # data
        self.x, self.y = x, y
        self.x_comp = x_comp
        
        # nodes in a bipartite graph
        n, p = x.shape
        x_src = torch.arange(n)
        x_dst = torch.arange(p)
        self.x_src = x_src 
        self.x_dst = x_dst

        # edges in a bipartite graph
        self.mask = ~torch.isnan(self.x) # 1 if not missing else 0
        if edge_index is None:
            self.edge_index =  torch.nonzero(self.mask).T
        else: 
            self.edge_index = edge_index  
        self.edge_value = self.x[self.mask]
        self.mask = self.mask * 1.
        
    def __len__(self): 
        return self.x.shape[0]

    def __getitem__(self, idx):
        # returns 
        # edges, nodes
        return {
            'x': self.x[idx], 
            'y': self.y[idx],
            'x_complete': self.x_comp[idx],
            'mask': self.mask[idx],
            'row_index': self.edge_index[0][self.edge_index[0] == idx],
            'col_index': self.edge_index[1][self.edge_index[0] == idx], 
            'edge_value': self.edge_value[self.edge_index[0] == idx]
        }

def collate_fn(samples): 
    r"""
    collate function of a bipartite dataset
    """     
    xs = [sample['x'] for sample in samples]
    ys = [sample['y'] for sample in samples]
    ms = [sample['mask'] for sample in samples]
    x_comps = [sample['x_complete'] for sample in samples]
    edge_values = [sample_value for sample in samples for sample_value in sample['edge_value']]
    edge_index = torch.tensor([[i, col] for i, sample in enumerate(samples) for row, col in zip(sample['row_index'], sample['col_index'])]).T
    return {
            'x': torch.stack(xs).contiguous(), 
            'y': torch.stack(ys).contiguous(),
            'mask': torch.stack(ms).contiguous(),
            'x_complete': torch.stack(x_comps).contiguous(),
            'edge_index': edge_index.contiguous(),
            'edge_value': torch.stack(edge_values).contiguous()       
    }

def make_mask(x_batch):
    r"""
    A fucntion to make a mask matrix

    # Parameter
    x_batch: input data (float torch type)

    # Returns
    mask: mask matrix which indicates the indices of not missing values (float torch type)
    """
    mask = ~torch.isnan(x_batch) * 1.0
    return mask

def train_valid_test_split(args, X, y, task_type= "cls"):
    r"""
    A fuction to train-validation-test split

    # Parameter
    args contains the followings...
    * data_path: a path to gesture-data
    * tr: the ratio of training data to the original data
    * val: the ratio of validation data to the original data
    remaining is the test data so, tr+val < 1.
    X: independent variables
    y: dependent variables
    task_type: regression if "regr", classification if "cls"

    # Return
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde (torch tensor type)

    """
    tr, val = args.tr, args.val
    # split the data into training, validation and test data
    n, p = X.shape
    si_val = int(n*tr)  # starting index of each validation and test data
    si_te = si_val + int(n*val)
    idx = np.random.permutation(np.arange(n)) # shuffle the index

    X_train, y_train = X.values[idx[:si_val], :], y.values[idx[:si_val], ]
    X_valid, y_valid = X.values[idx[si_val:si_te],:], y.values[idx[si_val:si_te], ]
    X_test, y_test = X.values[idx[si_te:], :], y.values[idx[si_te:], ]

    if len(args.cat_vars_pos) == 0: 
        args.numeric_vars_pos = [i for i in range(p)]
    else: 
        tot_features = list(range(p))
        numeric_vars_pos = list(set(tot_features)-set(args.cat_vars_pos))       
        args.numeric_vars_pos = numeric_vars_pos 

    args.column_names = list(X.columns)

    if args.standardize: 
        if len(args.cat_vars_pos) == 0:
            X_train, cache = min_max_scaler(X_train)
            X_valid, X_test = min_max_scaler_test(X_valid, cache), min_max_scaler_test(X_test, cache)
        else: 
            X_train[:, numeric_vars_pos], cache = min_max_scaler(X_train[:, numeric_vars_pos])
            X_valid[:, numeric_vars_pos], X_test[:, numeric_vars_pos]\
                 = min_max_scaler_test(X_valid[:, numeric_vars_pos], cache), min_max_scaler_test(X_test[:, numeric_vars_pos], cache)
        if task_type == 'regr': 
            y_train, cache = min_max_scaler(y_train) 
            y_test = min_max_scaler_test(y_test, cache)
            y_valid = min_max_scaler_test(y_valid, cache)

    X_train, X_valid, X_test\
        = torch.FloatTensor(X_train), torch.FloatTensor(X_valid), torch.FloatTensor(X_test)
    
    X_train_tilde, X_valid_tilde, X_test_tilde = X_train, X_valid, X_test
    
    if args.prob > 0.:
        # n = int(np.ceil(p * args.prob))
        X_train_tilde, _ = make_missing(X_train, args.prob)
        X_valid_tilde, _ = make_missing(X_valid, args.prob)

    if args.test_all_missing:
        args.test_n_missing = int(np.ceil(p * args.test_missing_prob))
        X_test_tilde, _ = make_missing_by_row(X_test, args.test_n_missing)
    else:
        X_test_tilde, _ = make_missing(X_test, args.test_missing_prob)

    if task_type == 'cls':
        y_train, y_valid, y_test\
            = torch.LongTensor(y_train), torch.LongTensor(y_valid), torch.LongTensor(y_test)
    else: 
        y_train, y_valid, y_test\
            = torch.FloatTensor(y_train), torch.FloatTensor(y_valid), torch.FloatTensor(y_test)
    
    args.task_type = task_type

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde

def load_gestures(args):
    r"""
    A function to load gestures-data.
    
    # Parameters
    args contains the followings...
    * data_path: a path to gesture-data
    * tr: the ratio of training data to the original data
    * val: the ratio of validation data to the original data
    remaining is the test data so, tr+val < 1.

    # Returns
    X_train, X_valid, X_test, y_train, y_valid, y_test (torch.FloatTensor for "X", torch.LongTensor for "y")    
    """
    path = os.listdir(args.data_path)
    gesture_files = []
    for g in path:
        if '_raw' in g:
            gesture_files.append(os.path.join(args.data_path, g))

    # to convert y to a numeric variable
    mapping = {'Rest': 0, 'Preparation': 1, 'Stroke': 2, 'Hold': 3, 'Retraction': 4}

    data = pd.DataFrame([])
    for i, gesture in enumerate(gesture_files):
        g = pd.read_csv(gesture)
        g = g.dropna(axis= 0)
        g = g.drop(['timestamp'], axis= 1)
        # Preparação --> Preparation (anomaly correction)
        anom_idx = g.iloc[:,-1] == 'Preparação'
        if sum(anom_idx) >= 1:
            g.loc[anom_idx, 'phase'] = 'Preparation'
        # convert y to a numeric variable
        g.iloc[:, -1] = g.iloc[:, -1].map(mapping).astype('int64')
        data = pd.concat([data, g], axis= 0)
    args.n_labels = 5 
    args.task_type = 'cls'
    print(data.info())
    print('-'*20)    
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde \
        = train_valid_test_split(args, X, y)

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde

def load_elec(args):
    r"""
    A function to load elec-data.
    
    # Parameters
    args contains the followings...
    * data_path: a path to gesture-data
    * tr: the ratio of training data to the original data
    * val: the ratio of validation data to the original data
    remaining is the test data so, tr+val < 1.

    # Returns
    X_train, X_valid, X_test, y_train, y_valid, y_test (torch.FloatTensor for "X", torch.LongTensor for "y")    
    """
    f = os.path.join(args.data_path, 'elec_data.csv') # file
    # data = pd.read_csv(f, encoding= 'cp949')
    data = pd.read_csv(f, encoding= 'cp949')
    data = data.dropna(axis= 0)
    print(data.info())
    print('-'*20)

    X, y = data.iloc[:, :8], data.iloc[:, -1] # voltage high-frequency average 
    args.cat_vars_pos = []
    # to convert y to a numeric variable
    mapping = {'정상':0, '주의':1, '경고':2}
    y = y.map(mapping)

    args.n_labels = 3
    args.task_type = 'cls'
    args.input_size= 8

    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde \
        = train_valid_test_split(args, X, y, 'cls')

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde 

def load_wind_turbin_power(args):
    r"""
    A function to load wind-turbin-data.
    
    # Parameters
    args contains the followings...
    * data_path: a path to gesture-data
    * tr: the ratio of training data to the original data
    * val: the ratio of validation data to the original data
    remaining is the test data so, tr+val < 1.

    # Returns
    X_train, X_valid, X_test, y_train, y_valid, y_test (torch.FloatTensor for "X", torch.FloatTensor for "y")    
    """
    f = os.path.join(args.data_path, 'features.csv') # file
    t = os.path.join(args.data_path, 'power.csv') # target file
    X = pd.read_csv(f)
    y = pd.read_csv(t)
    data = pd.merge(left = X, right= y, on= 'Timestamp' ,how= 'inner')
    data = data.dropna(axis= 0)
    X, y = data.iloc[:, 1:-1], data.iloc[:, -1]
    args.cat_vars_pos = []
    args.n_labels = 1
    args.task_type = 'regr'
    print(data.info())
    print('-'*20)
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde \
        = train_valid_test_split(args, X, y, task_type= 'regr')

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde       

def load_mobile(args):
    r"""
    A function to load mobile-price-prediction-data.
    
    # Parameters
    args contains the followings...
    * data_path: a path to gesture-data
    * tr: the ratio of training data to the original data
    * val: the ratio of validation data to the original data
    remaining is the test data so, tr+val < 1.

    # Returns
    X_train, X_valid, X_test, y_train, y_valid, y_test (torch.FloatTensor for "X", torch.FloatTensor for "y")    
    """
    data_file = os.path.join(args.data_path, 'train.csv')
    data = pd.read_csv(data_file)
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    cat_vars_pos = ['blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi']
    X_cat = X[cat_vars_pos]
    numeric_vars_pos = list(set(X.columns)-set(cat_vars_pos))
    X_num = X[numeric_vars_pos]
    X = pd.concat([X_cat,X_num], axis= 1)

    args.cat_vars_pos = list(range(X_cat.shape[1]))
    args.input_size = X.shape[1]
    args.n_labels = 4
    args.task_type = 'cls'
    print(data.info())
    print('-'*20)
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde \
        = train_valid_test_split(args, X, y, 'cls')

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde

def load_wine(args):
    r"""
    A function to load wine-data.
    
    # Parameters
    args contains the followings...
    * data_path: a path to gesture-data
    * tr: the ratio of training data to the original data
    * val: the ratio of validation data to the original data
    remaining is the test data so, tr+val < 1.

    # Returns
    X_train, X_valid, X_test, y_train, y_valid, y_test (torch.FloatTensor for "X", torch.FloatTensor for "y")    
    """
    data_file = os.path.join(args.data_path, 'WineQT.csv')
    data = pd.read_csv(data_file)
    data = data.dropna(axis= 0)
    X, y = data.iloc[:, :-2], data.iloc[:, -2]
    y = y-3
    args.input_size = X.shape[1] 
    args.cat_vars_pos = []
    args.n_labels = len(y.unique())
    args.task_type = 'cls'
    print(data.info())
    print('-'*20)
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde \
        = train_valid_test_split(args, X, y, task_type= 'cls')

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde

def load_appliances(args):
    r"""
    A function to load appliances-data.
    
    # Parameters
    args contains the followings...
    * data_path: a path to gesture-data
    * tr: the ratio of training data to the original data
    * val: the ratio of validation data to the original data
    remaining is the test data so, tr+val < 1.

    # Returns
    X_train, X_valid, X_test, y_train, y_valid, y_test (torch.FloatTensor for "X", torch.FloatTensor for "y")    
    """
    data_file = os.path.join(args.data_path, 'KAG_energydata_complete.csv')
    data = pd.read_csv(data_file)
    data = data.dropna(axis=0)
    X, y = data.iloc[:, 2:-2], data.iloc[:, 1]
    args.input_size = X.shape[1]
    args.n_labels = 1
    args.cat_vars_pos = []
    print(data.info())
    print('-'*20)
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde \
        = train_valid_test_split(args, X, y, task_type= 'regr')

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde

def load_pulsar(args):
    r"""
    A function to load pulsar-data.
    
    # Parameters
    args contains the followings...
    * data_path: a path to gesture-data
    * tr: the ratio of training data to the original data
    * val: the ratio of validation data to the original data
    remaining is the test data so, tr+val < 1.

    # Returns
    X_train, X_valid, X_test, y_train, y_valid, y_test (torch.FloatTensor for "X", torch.FloatTensor for "y")    
    """
    x_file = os.path.join(args.data_path, 'pulsar_x.csv')
    y_file = os.path.join(args.data_path, 'pulsar_y.csv')
    X, y = pd.read_csv(x_file), pd.read_csv(y_file).iloc[:, -1]
    data = pd.concat((X,y), axis= 1)
    data = data.dropna(axis= 0)

    X, y = data.iloc[:, :-1], data.iloc[:, -1]

    args.input_size = X.shape[1]
    args.n_labels = 2
    args.cat_vars_pos = []
    print(data.info())
    print('-'*20)
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde \
        = train_valid_test_split(args, X, y, task_type= 'cls')

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde

def load_faults(args):
    r"""
    A function to load faults-data.
    
    # Parameters
    args contains the followings...
    * data_path: a path to gesture-data
    * tr: the ratio of training data to the original data
    * val: the ratio of validation data to the original data
    remaining is the test data so, tr+val < 1.

    # Returns
    X_train, X_valid, X_test, y_train, y_valid, y_test (torch.FloatTensor for "X", torch.FloatTensor for "y")    
    """
    data_file = os.path.join(args.data_path, 'faults.csv')
    targets = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
    
    data = pd.read_csv(data_file)
    n, p = data.shape
    classes = np.zeros((n,))
    for i, target in enumerate(targets):
        idx = (data[target] == 1)
        classes[idx] = i
    data = data.drop(targets, axis=1)
    data['faults'] = classes

    data = data.dropna(axis= 0)

    X, y = data.iloc[:, :-1], data.iloc[:, -1]

    args.input_size = X.shape[1] 
    args.n_labels = 7
    args.cat_vars_pos = []

    print(data.info())
    print('-'*20)
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde \
        = train_valid_test_split(args, X, y, task_type= 'cls')

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde

def load_abalone(args):
    r"""
    A function to load abalone-data.
    
    # Parameters
    args contains the followings...
    * data_path: a path to gesture-data
    * tr: the ratio of training data to the original data
    * val: the ratio of validation data to the original data
    remaining is the test data so, tr+val < 1.

    # Returns
    X_train, X_valid, X_test, y_train, y_valid, y_test (torch.FloatTensor for "X", torch.FloatTensor for "y")    
    """
    data_file = os.path.join(args.data_path, 'abalone_csv.csv')
    data = pd.read_csv(data_file)
    data = data.dropna(axis=0)
    
    X, y = data.iloc[:, 1:-1], data.iloc[:, -1]
    dummies = pd.get_dummies(data.iloc[:, 0], drop_first= True)
    X = pd.concat([dummies, X], axis= 1)

    args.cat_vars_pos = list(range(dummies.shape[1]))
    args.input_size = X.shape[1]
    args.n_labels = 1

    data = pd.concat([X,y], axis= 1)
    print(data.info())
    print('-'*20)
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde \
        = train_valid_test_split(args, X, y, task_type= 'regr')

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde

def load_spam(args):
    r"""
    A function to load spam-data.
    
    # Parameters
    args contains the followings...
    * data_path: a path to gesture-data
    * tr: the ratio of training data to the original data
    * val: the ratio of validation data to the original data
    remaining is the test data so, tr+val < 1.

    # Returns
    X_train, X_valid, X_test, y_train, y_valid, y_test (torch.FloatTensor for "X", torch.FloatTensor for "y")    
    """
    data_file = os.path.join(args.data_path, 'spambase.data')
    data = pd.read_csv(data_file, header= None)
    data = data.dropna(axis=0)
    
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    
    args.input_size = X.shape[1]
    args.n_labels = 2
    args.cat_vars_pos = []
    print(data.info())
    print('-'*20)
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde \
        = train_valid_test_split(args, X, y, task_type= 'cls')

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde

def load_breast(args):
    r"""
    A function to load breast-data.
    
    # Parameters
    args contains the followings...
    * data_path: a path to breast-data
    * tr: the ratio of training data to the original data
    * val: the ratio of validation data to the original data
    remaining is the test data so, tr+val < 1.

    # Returns
    X_train, X_valid, X_test, y_train, y_valid, y_test (torch.FloatTensor for "X", torch.FloatTensor for "y")    
    """
    data_file = os.path.join(args.data_path, 'wpbc.data')
    data = pd.read_csv(data_file, header= None)
    data = data.dropna(axis=0)
    
    idx = data.iloc[:,-1] == '?'
    data = data.loc[~idx, :]
    data.iloc[:,-1] = data.iloc[:,-1].astype(float)

    cols = np.arange(data.shape[1])
    X, y = data.iloc[:, np.argwhere(cols!=1).flatten()], data.iloc[:, 1]

    y = y.map({'N':0, 'R':1})
    
    args.input_size = X.shape[1]
    args.n_labels = 2
    args.cat_vars_pos = []
    print(data.info())
    print('-'*20)
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde \
        = train_valid_test_split(args, X, y, task_type= 'cls')

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde

def load_letter(args):
    r"""
    A function to load letter-data.
    
    # Parameters
    args contains the followings...
    * data_path: a path to gesture-data
    * tr: the ratio of training data to the original data
    * val: the ratio of validation data to the original data
    remaining is the test data so, tr+val < 1.

    # Returns
    X_train, X_valid, X_test, y_train, y_valid, y_test (torch.FloatTensor for "X", torch.FloatTensor for "y")    
    """
    data_file = os.path.join(args.data_path, 'letter-recognition.data')
    data = pd.read_csv(data_file, header= None)
    data = data.dropna(axis=0)

    alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 
                'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
                'U', 'V', 'W', 'X', 'Y', 'Z']
    mapping = {alp:idx for idx, alp in enumerate(alphabets)}

    X, y = data.iloc[:, 1:], data.iloc[:, 0]
    y = y.map(mapping)
    
    args.input_size = X.shape[1]
    args.n_labels = len(alphabets)
    args.cat_vars_pos = []
    print(data.info())
    print('-'*20)
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde \
        = train_valid_test_split(args, X, y, task_type= 'cls')

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde

def load_eeg(args):
    r"""
    A function to load eeg-data.
    
    # Parameters
    args contains the followings...
    * data_path: a path to gesture-data
    * tr: the ratio of training data to the original data
    * val: the ratio of validation data to the original data
    remaining is the test data so, tr+val < 1.

    # Returns
    X_train, X_valid, X_test, y_train, y_valid, y_test (torch.FloatTensor for "X", torch.FloatTensor for "y")    
    """
    data_file = os.path.join(args.data_path, 'EEG_Eye_State_Classification.csv')
    data = pd.read_csv(data_file)
    data = data.dropna(axis=0)

    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    
    args.input_size = X.shape[1]
    args.n_labels = 2
    args.cat_vars_pos = []
    print(data.info())
    print('-'*20)
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde \
        = train_valid_test_split(args, X, y, task_type= 'cls')

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde

def load_recipes(args):
    r"""
    A function to load recipes-data.
    
    # Parameters
    args contains the followings...
    * data_path: a path to gesture-data
    * tr: the ratio of training data to the original data
    * val: the ratio of validation data to the original data
    remaining is the test data so, tr+val < 1.

    # Returns
    X_train, X_valid, X_test, y_train, y_valid, y_test (torch.FloatTensor for "X", torch.FloatTensor for "y")    
    """
    data_file = os.path.join(args.data_path, 'epi_r.csv')
    data = pd.read_csv(data_file)
    data = data.dropna(axis=0)
    
    X, y = data.iloc[:, 2:-1], data.iloc[:, 1]
    # dummies = pd.get_dummies(data.iloc[:, 0], drop_first= True)
    # X = pd.concat([dummies, X], axis= 1)

    args.cat_vars_pos = list(range(4, X.shape[1]))
    args.input_size = X.shape[1]
    args.n_labels = 1

    data = pd.concat([X,y], axis= 1)
    print(data.info())
    print('-'*20)
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde \
        = train_valid_test_split(args, X, y, task_type= 'regr')

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde

def load_stroke(args):
    r"""
    A function to load stroke-data.
    
    # Parameters
    args contains the followings...
    * data_path: a path to gesture-data
    * tr: the ratio of training data to the original data
    * val: the ratio of validation data to the original data
    remaining is the test data so, tr+val < 1.

    # Returns
    X_train, X_valid, X_test, y_train, y_valid, y_test (torch.FloatTensor for "X", torch.FloatTensor for "y")    
    """
    data_file = os.path.join(args.data_path, 'healthcare-dataset-stroke-data.csv')
    data = pd.read_csv(data_file)
    data = data.dropna(axis=0)
    
    X, y = data.iloc[:, 1:-1], data.iloc[:, -1]
    X_num = X[['hypertension','heart_disease','age','avg_glucose_level','bmi']]
    col_cat = X.columns.drop(['age','avg_glucose_level','bmi','hypertension','heart_disease'])
    X_cat = X[col_cat]
    X_cat = pd.get_dummies(X_cat, drop_first= True)
    X = pd.concat([X_cat, X_num], axis= 1)

    args.cat_vars_pos = list(range(13))
    args.input_size = X.shape[1]
    args.n_labels = 2

    data = pd.concat([X,y], axis= 1)
    print(data.info())
    print('-'*20)
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde \
        = train_valid_test_split(args, X, y, task_type= 'cls')

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde

def load_simul(args):
    r"""
    A function to load simul-data.
    
    # Parameters
    args contains the followings...
    * data_path: a path to gesture-data
    * tr: the ratio of training data to the original data
    * val: the ratio of validation data to the original data
    remaining is the test data so, tr+val < 1.

    # Returns
    X_train, X_valid, X_test, y_train, y_valid, y_test (torch.FloatTensor for "X", torch.FloatTensor for "y")    
    """
    target_file = os.path.join(args.data_path, 'target.csv')
    var_file = os.path.join(args.data_path, 'var.csv')
    y = pd.read_csv(target_file).iloc[:, -1]
    X = pd.read_csv(var_file)

    args.cat_vars_pos = []
    args.input_size = X.shape[1]
    args.n_labels = 2

    data = pd.concat([X,y], axis= 1)
    print(data.info())
    print('-'*20)
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde \
        = train_valid_test_split(args, X, y, task_type= 'cls')

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde

def load_bench(args):
    r"""
    A function to load bench-data.
    
    # Parameters
    args contains the followings...
    * data_path: a path to gesture-data
    * tr: the ratio of training data to the original data
    * val: the ratio of validation data to the original data
    remaining is the test data so, tr+val < 1.

    # Returns
    X_train, X_valid, X_test, y_train, y_valid, y_test (torch.FloatTensor for "X", torch.FloatTensor for "y")    
    """
    X_file = os.path.join(args.data_path, 'X_train.csv')
    y_file = os.path.join(args.data_path, 'y_train.csv')
    X = pd.read_csv(X_file).drop(['BestSquatKg'], axis= 1)
    y = pd.read_csv(y_file).iloc[:, 1]
    data = pd.concat([X,y], axis= 1)
    data = data.dropna(axis=0)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_num = X.iloc[:, 4:]
    X_cat = X.iloc[:, 2:4]
    X_cat = pd.get_dummies(X_cat, drop_first= True)
    X = pd.concat([X_cat, X_num], axis= 1)

    args.cat_vars_pos = list(range(4))
    args.input_size = X.shape[1]
    args.n_labels = 1

    data = pd.concat([X,y], axis= 1)
    print(data.info())
    print('-'*20)
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde \
        = train_valid_test_split(args, X, y, task_type= 'regr')

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde

def load_concrete(args):
    r"""
    A function to load concrete-data.
    
    # Parameters
    args contains the followings...
    * data_path: a path to gesture-data
    * tr: the ratio of training data to the original data
    * val: the ratio of validation data to the original data
    remaining is the test data so, tr+val < 1.

    # Returns
    X_train, X_valid, X_test, y_train, y_valid, y_test (torch.FloatTensor for "X", torch.FloatTensor for "y")    
    """
    try:
        data_file = os.path.join(args.data_path, 'Concrete_Data.csv')
        data = pd.read_csv(data_file)
    except:
        data_file = os.path.join(args.data_path, 'Concrete_Data.xsl')
        data = pd.read_excel(data_file)
    
    X, y = data.iloc[:, :-1], data.iloc[:,-1]
    args.cat_vars_pos = []
    args.input_size = X.shape[1]
    args.n_labels = 1
    print(data.info())
    print('-'*20)
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde \
        = train_valid_test_split(args, X, y, task_type= 'regr')

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde

def load_energy(args):
    r"""
    A function to load energy-data.
    
    # Parameters
    args contains the followings...
    * data_path: a path to gesture-data
    * tr: the ratio of training data to the original data
    * val: the ratio of validation data to the original data
    remaining is the test data so, tr+val < 1.

    # Returns
    X_train, X_valid, X_test, y_train, y_valid, y_test (torch.FloatTensor for "X", torch.FloatTensor for "y")    
    """
    try:
        data_file = os.path.join(args.data_path, 'ENB2012_data.csv')
        data = pd.read_csv(data_file)
    except:
        data_file = os.path.join(args.data_path, 'ENB2012_data.xsl')
        data = pd.read_excel(data_file)
    data = data.dropna()
    X, y = data.iloc[:, :-2], data.iloc[:,-2]
    args.cat_vars_pos = []
    args.input_size = X.shape[1]
    args.n_labels = 1
    print(data.info())
    print('-'*20)
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde \
        = train_valid_test_split(args, X, y, task_type= 'regr')

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde

def load_yacht(args):
    r"""
    A function to load yacht-data.
    
    # Parameters
    args contains the followings...
    * data_path: a path to gesture-data
    * tr: the ratio of training data to the original data
    * val: the ratio of validation data to the original data
    remaining is the test data so, tr+val < 1.

    # Returns
    X_train, X_valid, X_test, y_train, y_valid, y_test (torch.FloatTensor for "X", torch.FloatTensor for "y")    
    """

    data_file = os.path.join(args.data_path, 'yacht_hydrodynamics.data')
    with open(data_file) as data:
        lines = []
        while True:
            line = data.readline()
            if not line:
                break
            d = line.strip().split(' ')
            if '' in d:
                continue
            else: 
                d = list(map(lambda x: float(x), d))
                lines.append(d)

    data = pd.DataFrame(lines, columns = [f'x{i}' for i in range(7)])

    data = data.dropna()
    X, y = data.iloc[:, :-1], data.iloc[:,-1]
    args.cat_vars_pos = []
    args.input_size = X.shape[1]
    args.n_labels = 1
    print(data.info())
    print('-'*20)
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde \
        = train_valid_test_split(args, X, y, task_type= 'regr')

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde

def load_insurance(args):
    r"""
    A function to load insurance-data.
    
    # Parameters
    args contains the followings...
    * data_path: a path to gesture-data
    * tr: the ratio of training data to the original data
    * val: the ratio of validation data to the original data
    remaining is the test data so, tr+val < 1.

    # Returns
    X_train, X_valid, X_test, y_train, y_valid, y_test (torch.FloatTensor for "X", torch.FloatTensor for "y")    
    """
    data_file = os.path.join(args.data_path, 'insurance.csv')
    df = pd.read_csv(data_file)

    df = df.dropna()
    y = df['charges']
    df_num = df[['age', 'bmi' ,'children']]
    df_cat = pd.get_dummies(df[['sex','smoker','region']], drop_first= True)
    X = pd.concat([df_num, df_cat], axis= 1)

    args.cat_vars_pos = [i for i in range(3, df_cat.shape[1]+3)]
    args.input_size = X.shape[1]
    args.n_labels= 1 

    print(df.info())
    print('-'*20)
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde \
        = train_valid_test_split(args, X, y, task_type= 'regr')

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde

def load_cancer(args): 
    loaded = load_breast_cancer()
    x, y = loaded.data, loaded.target
    data = np.concatenate([x,y[:, np.newaxis]], axis= 1)
    df = pd.DataFrame(data, columns= list(loaded.feature_names)+['target'])

    x = df.iloc[:, :-1]
    y = df.iloc[:, -1] *1.

    df = df.dropna()
    args.cat_vars_pos = []
    args.input_size = x.shape[1]
    args.n_labels= len(df.iloc[:, -1].unique())

    print(df.info())
    print('-'*20)
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde \
        = train_valid_test_split(args, x, y, task_type= 'cls')

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde

def load_diabetes_2(args): 
    loaded = load_diabetes()
    x, y = loaded.data, loaded.target
    data = np.concatenate([x,y[:, np.newaxis]], axis= 1)
    df = pd.DataFrame(data, columns= list(loaded.feature_names)+['target'])
    
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    df = df.dropna()
    args.cat_vars_pos = []
    args.input_size = x.shape[1]
    args.n_labels= 1

    print(df.info())
    print('-'*20)
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde \
        = train_valid_test_split(args, x, y, task_type= 'regr')

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde

def load_digits_2(args): 
    loaded = load_digits()
    x, y = loaded.data, loaded.target
    data = np.concatenate([x,y[:, np.newaxis]], axis= 1)
    df = pd.DataFrame(data, columns= list(loaded.feature_names)+['target'])
    
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1] *1.

    df = df.dropna()
    args.cat_vars_pos = []
    args.input_size = x.shape[1]
    args.n_labels= len(df.iloc[:, -1].unique())

    print(df.info())
    print('-'*20)
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde \
        = train_valid_test_split(args, x, y, task_type= 'cls')

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde


def load_data(args): 

    print("Loading data...")
    if args.data_type == 'gesture':
        # load gestures-data
        X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde\
            = load_gestures(args)
        task_type = 'cls'
    elif args.data_type == 'elec':
        # load elec-data
        X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde\
            = load_elec(args)
        task_type = 'cls'
    elif args.data_type == 'wind':
        # load wind-data
        X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde\
            = load_wind_turbin_power(args)
        task_type = 'regr'
    elif args.data_type == 'mobile': 
        # load wind-data
        X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde\
            = load_mobile(args)
        task_type= 'cls'     
    elif args.data_type == 'wine': 
        # load wind-data
        X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde\
            = load_wine(args)
        task_type= 'cls'  
    elif args.data_type == 'appliances': 
        # load wind-data
        X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde\
            = load_appliances(args)
        task_type= 'regr'
    elif args.data_type == 'pulsar': 
        # load pulsar data 
        X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde\
            = load_pulsar(args)
        task_type = 'cls'     
    elif args.data_type == 'faults': 
        # load faults data
        X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde\
            = load_faults(args)        
        task_type = 'cls'         
    elif args.data_type == 'abalone': 
        # load abalone data
        X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde\
            = load_abalone(args)              
        task_type = 'regr'
    elif args.data_type == 'spam': 
        # load faults data
        X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde\
            = load_spam(args)        
        task_type = 'cls'    
    elif args.data_type == 'breast': 
        # load faults data
        X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde\
            = load_breast(args)        
        task_type = 'cls'       
    elif args.data_type == 'letter': 
        # load faults data
        X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde\
            = load_letter(args)        
        task_type = 'cls'  
    elif args.data_type == 'eeg':
        # load faults data
        X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde\
            = load_eeg(args)        
        task_type = 'cls'         
    elif args.data_type == 'recipes': 
        # load wind-data
        X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde\
            = load_recipes(args)
        task_type= 'regr'
    elif args.data_type == 'stroke': 
        # load wind-data
        X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde\
            = load_stroke(args)        
        task_type= 'cls'
    elif args.data_type == 'simul': 
        X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde\
            = load_simul(args)        
        task_type= 'cls'        
    elif args.data_type == 'bench': 
        X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde\
            = load_bench(args)        
        task_type= 'regr'   
    elif args.data_type == 'concrete': 
        X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde\
            = load_concrete(args)        
        task_type= 'regr'    
    elif args.data_type == 'energy': 
        X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde\
            = load_energy(args)        
        task_type= 'regr'     
    elif args.data_type == 'yacht': 
        X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde\
            = load_yacht(args)        
        task_type= 'regr'            
    elif args.data_type == 'insurance': 
        X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde\
            = load_insurance(args)        
        task_type= 'regr'         
    elif args.data_type == 'cancer': 
        X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde\
            = load_cancer(args)        
        task_type= 'cls'  
    elif args.data_type == 'diabetes': 
        X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde\
            = load_diabetes_2(args)        
        task_type= 'regr'  
    elif args.data_type == 'digits': 
        X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde\
            = load_digits_2(args)        
        task_type= 'cls'  
    else: 
        print("Unkown data type, data type should be one of the followings...")
        print("gesture, elec, wind, mobile, wine, appliances, pulsar, faults, abalone, spam, letter")
        sys.exit()
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde, task_type