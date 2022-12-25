import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import csv

from utils.torch_utils import get_loss_imp
from utils.utils import *

class Trainer: 

    def __init__(self):
        super().__init__()

    def __call__(self, args, model, 
                train_loader, valid_loader, 
                early_stopping, 
                optimizer, scheduler=None, 
                device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        num_batches = len(train_loader)
        # # to store losses per epoch
        # tr_loss, valid_loss = 0, 0 # todo
        for epoch in range(args.epoch):
            training_loss, validation_loss = [], []
            training_weights, validation_weights = [], []
            # a training loop
            for batch_idx, batch in enumerate(train_loader):
                batch['x'], batch['y'], batch['edge_index'], batch['edge_value']\
                    = batch['x'].to(device), batch['y'].to(device), batch['edge_index'].to(device), batch['edge_value'].to(device)
                batch['x_complete'], batch['mask'] = batch['x_complete'].to(device), batch['mask'].to(device) 
                model.train()
                # feed forward
                with torch.set_grad_enabled(True):
                    tr_loss = model.train_step(batch)
                    model.zero_grad()
                    optimizer.zero_grad()
                    tr_loss['total_loss'].backward()
                    optimizer.step() 
                training_loss.append(tr_loss['total_loss'].detach().cpu().item())
                training_weights.append(len(batch['x']))

            # a validation loop 
            for batch_idx, batch in enumerate(valid_loader):
                batch['x'], batch['y'], batch['edge_index'], batch['edge_value']\
                    = batch['x'].to(device), batch['y'].to(device), batch['edge_index'].to(device), batch['edge_value'].to(device)
                batch['x_complete'], batch['mask'] = batch['x_complete'].to(device), batch['mask'].to(device) 
                model.eval()
                valid_loss = model.val_step(batch)
                validation_loss.append(valid_loss['total_loss'].detach().cpu().item())
                validation_weights.append(len(batch['x']))
                
            training_loss = np.average(training_loss, weights = training_weights)
            validation_loss = np.average(validation_loss, weights = validation_loss)
            print(f'Epoch [{epoch+1}/{args.epoch}]: training loss= {training_loss:.6f}, validation loss= {validation_loss:.6f}')
            early_stopping(validation_loss, model, epoch, optimizer)

            if early_stopping.early_stop:
                break     
    
    def test(self, args, model, test_loader, device, perf_measure= None):
        # initiate performance.
        perfs = {}        
        weights = []
        for batch_idx, batch in enumerate(test_loader): 
            batch['x'], batch['y'], batch['edge_index'], batch['edge_value']\
                = batch['x'].to(device), batch['y'].to(device), batch['edge_index'].to(device), batch['edge_value'].to(device)
            batch['x_complete'], batch['mask'] = batch['x_complete'].to(device), batch['mask'].to(device) 
            model.eval() 
            loss = model.test_step(batch)
            num_batch = len(batch['x'])
            weights.append(num_batch)
            for k, v in loss.items(): 
                if perfs.get(k) is None: 
                    perfs[k] = []
                perfs.get(k).append(v)
        for k, v in perfs.items(): 
            perfs[k] = np.average(perfs.get(k), weights= weights)
        
        return perfs
                
class PreTrainer: 

    def __init__(self):
        super().__init__()

    def __call__(self, args, model, 
                train_loader, valid_loader, 
                early_stopping, 
                optimizer, scheduler=None, 
                device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        num_batches = len(train_loader)
        # # to store losses per epoch
        model.pretraining = True
        for epoch in range(args.epoch):
            training_loss, validation_loss = [], []
            training_weights, validation_weights = [], []
            # a training loop
            for batch_idx, batch in enumerate(train_loader):
                batch['x'], batch['y'], batch['edge_index'], batch['edge_value']\
                    = batch['x'].to(device), batch['y'].to(device), batch['edge_index'].to(device), batch['edge_value'].to(device)
                batch['x_complete'], batch['mask'] = batch['x_complete'].to(device), batch['mask'].to(device) 
                model.train()
                # feed forward
                with torch.set_grad_enabled(True):
                    tr_loss = model.train_step(batch)
                model.zero_grad()
                optimizer.zero_grad()
                tr_loss['total_loss'].backward()
                optimizer.step()
                training_loss.append(tr_loss['total_loss'].detach().cpu().item())
                training_weights.append(len(batch['x']))

            # a validation loop 
            for batch_idx, batch in enumerate(valid_loader):
                batch['x'], batch['y'], batch['edge_index'], batch['edge_value']\
                    = batch['x'].to(device), batch['y'].to(device), batch['edge_index'].to(device), batch['edge_value'].to(device)
                batch['x_complete'], batch['mask'] = batch['x_complete'].to(device), batch['mask'].to(device) 
                model.eval()
                valid_loss = model.val_step(batch)        
                validation_loss.append(valid_loss['total_loss'].detach().cpu().item()) 
                validation_weights.append(len(batch['x']))

            training_loss = np.average(training_loss, weights = training_weights)
            validation_loss = np.average(validation_loss, weights = validation_loss)                 
            early_stopping(validation_loss, model, epoch, optimizer)

            if early_stopping.early_stop:
                break     
    
    def test(self, args, model, test_loader, device, perf_measure= None):
        # initiate performance.
        perfs = {'num_imp_loss': [], 'cat_imp_loss': []}        
        weights = []
        for batch_idx, batch in enumerate(test_loader): 
            batch['x'], batch['y'], batch['edge_index'], batch['edge_value']\
                = batch['x'].to(device), batch['y'].to(device), batch['edge_index'].to(device), batch['edge_value'].to(device)
            batch['x_complete'], batch['mask'] = batch['x_complete'].to(device), batch['mask'].to(device) 
            model.eval() 
            loss = model.test_step(batch)
            num_batch = len(batch['x'])
            weights.append(num_batch)
            perfs['cat_imp_loss'].append(loss['cat_imp_loss'])
            perfs['num_imp_loss'].append(loss['num_imp_loss'])
        for k, v in perfs.items(): 
            perfs[k] = np.average(perfs.get(k), weights= weights)
        
        return perfs
                

