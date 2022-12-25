import torch
from torch import nn 
import numpy as np
import os

class EarlyStopping:
    """
    Applies early stopping condition... 
    """
    def __init__(self, 
                patience: int= 10,
                verbose: bool= False,
                delta: float= 0,
                path= './',
                model_name= 'latest_checkpoint.pth.tar'):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.model_name = model_name
        self.path = os.path.join(path, model_name)
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.counter = 0

    def __call__(self, val_loss, model, epoch, optimizer):
        ckpt_dict = {
            'epoch':epoch+1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, ckpt_dict)
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            print(f'Early stopping counter {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, ckpt_dict)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, ckpt_dict):
        if self.verbose:
            print(f'Validation loss decreased: {self.val_loss_min:.4f} --> {val_loss:.4f}. Saving model...')
        torch.save(ckpt_dict, self.path)
        self.val_loss_min = val_loss

def get_loss_imp(x, x_hat, m, cat_features= None, test= False):
    """
    # Parameters
    x: original value
    m: mask
    x_hat: imputation
    cat_features: a list of indices of categorical features

    # Returns
    imputation loss:
    m * Loss(x_hat, x)
    """
    if cat_features is not None: 
        bs, tot_features = x.shape
        tot_features = list(range(tot_features))
        num_features = list(set(tot_features)-set(cat_features))
        
        # categorical features
        a_cat = torch.masked_select(x_hat[:, cat_features], m[:, cat_features]== 1.)
        b_cat = torch.masked_select(x[:, cat_features], m[:, cat_features]== 1.)
        num_cat = torch.sum(m[:, cat_features]== 1.).detach()
        # numeric features
        a_num = torch.masked_select(x_hat[:, num_features], m[:, num_features]== 1.)
        b_num = torch.masked_select(x[:, num_features], m[:, num_features]== 1.)
        num_num = torch.sum(m[:, num_features]== 1.).detach()

        bce_loss = nn.BCEWithLogitsLoss(reduction= 'sum')
        mse_loss = nn.MSELoss(reduction= 'sum')

        cat_loss = bce_loss(a_cat, b_cat) 
        num_loss = mse_loss(a_num, b_num)
        reconloss =\
            (cat_loss + num_loss)/(num_cat+num_num)

        if test: 
            return num_loss/num_num, cat_loss/num_cat
        
        return reconloss, -1

    a = torch.masked_select(x_hat, m==1.)
    b = torch.masked_select(x, m==1.)
    return torch.mean((a-b)**2), -1