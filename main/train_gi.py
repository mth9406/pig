import os
import sys
import argparse

import torch
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', 20)
import networkx as nx
from matplotlib import pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.torch_utils import * 
from utils.dataloader import * 
from utils.utils import *

from layers.graph_imputer import *
from trainer.gi_trainer import GraphImputerTrainer
from trainer.graph_sampler_trainer import GraphSamplerTrainer
from fancyimpute import SoftImpute
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from utils.gnn_utils import get_prior_adj_by_tree
from utils.utils import save_relation_adj

parser = argparse.ArgumentParser()

# Data path
parser.add_argument('--data_type', type= str, default= 'gestures', 
                    help= 'one of: gestures, elec, wind')
parser.add_argument('--data_path', type= str, default= './data/gesture')
parser.add_argument('--tr', type= float, default= 0.7, 
                help= 'the ratio of training data to the original data')
parser.add_argument('--val', type= float, default= 0.2, 
                help= 'the ratio of validation data to the original data')
parser.add_argument('--standardize', action= 'store_true', 
                help= 'standardize the inputs if it is true.')
parser.add_argument('--test_missing_prob', type= float, default= 0.1, 
                help= 'the ratio of missing data to make in the original data')
parser.add_argument('--prob', type= float, default= 0.2, 
                help= 'the ratio of missing data to make in the original validation data')
parser.add_argument('--test_all_missing', action= 'store_true', 
                help= 'force every observation in the test data to have missing values.')
parser.add_argument('--test_n_missing', type= int, default= 1, 
                help= 'the number of missing values to generate by row. (depreciated, it is auto-set)')
parser.add_argument('--exp_num', type = str, default= '')

# Training options 
parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--pretrain_epoch', type=int, default=30, help='the number of epochs to pre-train graph sampling layer')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  
parser.add_argument('--patience', type=int, default=30, help='patience of early stopping condition')
parser.add_argument('--delta', type= float, default=0., help='significant improvement to update a model')
parser.add_argument('--print_log_option', type= int, default= 10, help= 'print training loss every print_log_option')
parser.add_argument('--imp_loss_penalty', type= float, default= 1., 
                    help= 'the penalty term of imputation loss')
parser.add_argument('--reg_loss_peanlty', type= float, default= 0.01, 
                    help= 'the penalty term of regularization loss')
parser.add_argument('--training_missing_prob', type= float, default= 0.2, 
                help= 'the ratio of missing data to make in the original training data')

# model options
parser.add_argument('--model_path', type= str, default= './model',
                    help= 'a path to (save) the model')
parser.add_argument('--num_layers', type= int, default= 1, 
                    help= 'the number of gcn layers')
parser.add_argument('--auto_set_emb_size', action= 'store_true', help= 'auto set the embedding sizes')
parser.add_argument('--graph_emb_dim', type= int, required= False)
parser.add_argument('--alpha', type= float, default= 3., 
                    help= 'activation scale in the graph sampling layer')
parser.add_argument('--model_type', type= str, default= 'graph_imputer', 
                    help= 'graph_imputer')
parser.add_argument('--init_imp_strategy', type= str, default= 'soft_impute',
                    help= 'one of soft_impute, mean, mice (default= soft=impute)')

# test options
parser.add_argument('--test', action='store_true', help='test')
parser.add_argument('--model_file', type= str, default= 'gi_latest_checkpoint.pth.tar'
                    ,help= 'model file', required= False)
parser.add_argument('--model_name', type= str, default= 'gi_latest_checkpoint.pth.tar'
                    ,help= 'model name')
parser.add_argument('--test_results_path', type= str, default= './test_results', 
                    help= 'a path to save the results')
parser.add_argument('--num_folds', type= int, default= 1, 
                    help = 'the number of folds')

args = parser.parse_args() 
print(args) 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# make a path to save a model 
if not os.path.exists(args.model_path):
    print("Making a path to save the model...")
    os.makedirs(args.model_path, exist_ok= True)
else:
    print("The path already exists, skip making the path...")

# make a path to save a model 
if not os.path.exists(args.test_results_path):
    print("Making a path to save the results...")
    os.makedirs(args.test_results_path, exist_ok= True)
else:
    print("The path already exists, skip making the path...")

def main(args): 

    # load data
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde, task_type \
        = load_data(args)
    M_train = make_mask(X_train_tilde)
    M_valid, M_test = make_mask(X_valid_tilde), make_mask(X_test_tilde)
    # define training, validation, test datasets and their dataloaders respectively 
    train_data, valid_data, test_data \
        = TableDataset(X_train, M_train, y_train, X_comp= X_train),\
            TableDataset(X_valid_tilde, M_valid, y_valid, X_comp= X_valid),\
            TableDataset(X_test_tilde, M_test, y_test, X_comp= X_test)
    train_loader, valid_loader, test_loader \
        = DataLoader(train_data, batch_size = args.batch_size, shuffle = True),\
            DataLoader(valid_data, batch_size = args.batch_size, shuffle = True),\
            DataLoader(test_data, batch_size = args.batch_size, shuffle = False)    

    # auto setting
    if args.auto_set_emb_size:
        args.graph_emb_dim = args.input_size//2
    
    prior_adj = get_prior_adj_by_tree(X_train.numpy(), args.numeric_vars_pos, args.cat_vars_pos, device= device)

    # initial imputation strategy 
    # soft_impute, mean, mice
    if args.init_imp_strategy == 'soft_impute':
        init_impute = SoftImpute(verbose= False)
    elif args.init_imp_strategy == 'mean' or args.init_imp_strategy == 'median' or args.init_imp_strategy == 'constant': 
        init_impute = SimpleImputer(strategy=f'{args.init_imp_strategy}', verbose=False)
    elif args.init_imp_strategy == 'mice': 
        init_impute = IterativeImputer(random_state=0, verbose= False, max_iter= 100)
    else: 
        print(f'the initial imputation strategy: {args.init_imp_strategy} is not implemented yet')
        print('using soft-impute as the initial imputation strategy...')
        args.init_imp_strategy = 'soft_impute'
        init_impute = SoftImpute(verbose= False)
        
    if args.model_type == 'graph_imputer': 
        model = GraphImputer(args.input_size, args.graph_emb_dim,
                            args.n_labels, args.cat_vars_pos, args.numeric_vars_pos, args.num_layers, args.alpha,
                            args.imp_loss_penalty, args.reg_loss_peanlty,prior_adj, device, args.task_type, init_impute
                            ).to(device)
    else: 
        print('The model is yet to be implemented')
        sys.exit() 

    # save the prior adjacency matrix 
    save_relation_adj(prior_adj.detach().cpu().numpy().T, args, 'prior')    

    trainer = GraphImputerTrainer(args.training_missing_prob, args.print_log_option)
    optimizer = optim.Adam(model.parameters(), args.lr) 
    early_stopping = EarlyStopping(
        patience= args.patience,
        verbose= True,
        delta = args.delta,
        path= args.model_path,
        model_name= args.model_name
    )     

    if args.test: 
        print('loading the saved model')
        model_file = os.path.join(args.model_path, args.model_file)
        ckpt = torch.load(model_file)
        model.load_state_dict(ckpt['state_dict'])
        print('loading done!')
    else: 
        print('start pretraining graph...')
        trainer.pretrain_graph(args, model, optimizer, device)
        print('pretraining done')

        print('start pretraining downstream task...')
        trainer.pretrain_task(args, model, train_loader, valid_loader, optimizer, device)
        print('pretraining done')

        print('start training...')
        # trainer(args, model, train_loader, valid_loader, early_stopping, optimizer, scheduler, device)
        trainer(args, model, train_loader, valid_loader, early_stopping, optimizer, device= device)

    print("==============================================")
    print("Testing the model...") 
    print('loading the saved model')
    model_file = os.path.join(args.model_path, args.model_name)
    ckpt = torch.load(model_file)
    model.load_state_dict(ckpt['state_dict'])
    print('loading done!')  
    perfs = trainer.test(args, model, test_loader, device)
    for k, perf in perfs.items(): 
        print(f'{k}: {perf:.4f}')   
    
    save_relation_adj(model.get_adj().T, args, 'posterior')
    print('saving the graph done!')

    return perfs

if __name__ == '__main__': 
    perf = main(args)
    perfs = dict().fromkeys(perf, None)
    for k in perfs.keys():
        perfs[k] = [perf[k]]
    for i in range(1, args.num_folds): 
        perf = main(args)
        for k in perfs.keys():
            perfs[k].append(perf[k])
    perfs_df = pd.DataFrame(perfs)
    perfs_df = perfs_df.append(perfs_df.mean(skipna=True).to_dict(), ignore_index= True)
    perfs_df = perfs_df.append(perfs_df.std(skipna=True).to_dict(), ignore_index= True)
    perfs_df.index = [str(i) for i in range(len(perfs_df)-2)] + ['mean', 'std']

    perfs_path = os.path.join(args.test_results_path, f'{args.model_type}_{args.init_imp_strategy}/{args.data_type}')
    os.makedirs(perfs_path, exist_ok= True)
    pefs_df_file = os.path.join(perfs_path, f'{args.exp_num}_{args.model_type}_missing_{args.test_missing_prob}.csv')
    perfs_df.to_csv(pefs_df_file)
    print(perfs_df.tail())

