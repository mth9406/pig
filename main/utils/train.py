import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import csv

from utils.torch_utils import get_loss_imp
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

from utils.utils import *
from fancyimpute import SoftImpute 

def train(args, 
          model, 
          train_loader, valid_loader, 
          optimizer, criterion, early_stopping,
          device):
    logs = {
        'tr_loss':[],
        'valid_loss':[]
    }

    kl_loss = None

    num_batches = len(train_loader)
    print('Start training...')
    for epoch in range(args.epoch):
        # to store losses per epoch
        tr_loss, valid_loss = 0, 0
        # a training loop
        for batch_idx, x in enumerate(train_loader):
            x['input'], x['mask'], x['label'] \
                = x['input'].to(device), x['mask'].to(device), x['label'].to(device)

            model.train()
            # feed forward
            if args.model_type == 'linear' or args.model_type == 'aei'  or args.model_type == 'vai': 
                loss = train_batch(args, x, model, criterion, optimizer)
            else: 
                loss = train_batch_by_cols(args, x, model, criterion, optimizer)
            # store the d_tr_loss
            tr_loss += loss

            if (batch_idx+1) % args.print_log_option == 0:
                print(f'Epoch [{epoch+1}/{args.epoch}] Batch [{batch_idx+1}/{num_batches}]: \
                    loss = {loss}')

        # a validation loop 
        for batch_idx, x in enumerate(valid_loader):
            x['input'], x['mask'], x['label'] \
                = x['input'].to(device), x['mask'].to(device), x['label'].to(device)
            
            # model.eval()
            loss = 0
            with torch.no_grad():
                out = model(x, cat_features= args.cat_vars_pos)
                if out['preds'] is not None:
                    loss = criterion(out['preds'], x['label'])
                if out['imputation'] is not None:
                    loss_imp, _ = get_loss_imp(x['input'], out['imputation'], x['mask'], cat_features= args.cat_vars_pos)
                    loss += args.imp_loss_penalty * loss_imp
                if out['regularization_loss'] is not None: 
                    loss += args.imp_loss_penalty * out['regularization_loss']
                if kl_loss is not None: 
                    kl = kl_loss(model)[0]
                    loss += args.kl_weight * kl
            valid_loss += loss.detach().cpu().item()
        
        # save current loss values
        tr_loss, valid_loss = tr_loss/len(train_loader), valid_loss/len(valid_loader)
        logs['tr_loss'].append(tr_loss)
        logs['valid_loss'].append(valid_loss)

        print(f'Epoch [{epoch+1}/{args.epoch}]: training loss= {tr_loss:.6f}, validation loss= {valid_loss:.6f}')
        early_stopping(valid_loss, model, epoch, optimizer)

        if early_stopping.early_stop:
            break     

    print("Training done! Saving logs...")
    log_path= os.path.join(args.model_path, 'training_logs')
    os.makedirs(log_path, exist_ok= True)
    log_file= os.path.join(log_path, 'training_logs.csv')
    with open(log_file, 'w', newline= '') as f:
        wr = csv.writer(f)
        n = len(logs['tr_loss'])
        rows = np.array(list(logs.values())).T
        wr.writerow(list(logs.keys()))
        for i in range(1, n):
            wr.writerow(rows[i, :])

def train_batch(args, x, model, criterion, optimizer): 
    model.train()
    loss = 0
    # feed forward
    with torch.set_grad_enabled(True):
        out = model(x, cat_features= args.cat_vars_pos)
        # prediction loss
        if out['preds'] is not None:
            loss += criterion(out['preds'], x['label'])
        if out['imputation'] is not None: 
            mask = (x['mask']==1).to(x['mask'].device) * 1. 
            loss_imp, _ = get_loss_imp(x['input'], out['imputation'], mask, cat_features= args.cat_vars_pos)
            loss += loss_imp
        if out['regularization_loss'] is not None: 
            loss += out['regularization_loss']
        # imputation loss 
    # backward 
    model.zero_grad()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.detach().cpu().item()

def train_batch_by_cols(args, x, model, criterion, optimizer): 
    """Calculates imputation loss + prediction loss by columns and returns the total loss       
    # Arguments     
    ___________     
    args : dict type     
        It has the following items    
        * cat_features: indices of categorical features     
    x : dict type      
        * x['input']: input vector     
        * x['mask']: mask vector        
        ...       
    model: subclass of nn.Module       

    # Returns      
    _________         
    total loss of each columns        
    if it uses variational auto-encoder, then the prior-fitting regularization will be added.            
    """
    model.train()
    n, p = x['input'].shape 
    col_idx = torch.arange(p)
    tot_loss = 0.

    with torch.set_grad_enabled(True):
        for j in torch.randperm(p): 
            loss = 0.
            input_tilde = torch.clone(x['input']).detach().to(x['input'].device)
            selected_idx = torch.randperm(n)[:int(n*0.10)]
            input_tilde[selected_idx, j] = float('nan')
            mask_tilde = torch.clone(x['mask']).detach().to(x['mask'].device)
            mask_tilde[selected_idx, j] = 0.
            # warm start imputation
            input_tilde = torch.FloatTensor(SoftImpute(verbose= False).fit_transform(input_tilde.detach().cpu())).to(x['input'].device)
            x_tilde = {
                'input': input_tilde,
                'mask': mask_tilde
            }
            # make j'th column nan
            # and predict the j'th nan columns by vai 
            # obtain imputation loss w.r.t. j'th column and reconstruction loss for else where.
            # the loss is calculated only for 
            # x['mask'] == 1 
            out = model(x_tilde, cat_features= args.cat_vars_pos)
            mask = (x['mask']==1).to(x['mask'].device) * 1. 
            loss_imp, _ = get_loss_imp(x['input'], out['imputation'], mask, cat_features= args.cat_vars_pos)
            loss += args.imp_loss_penalty * loss_imp
            if out['regularization_loss'] is not None: 
                loss += args.imp_loss_penalty * out['regularization_loss']
            
            # forward prediction 
            loss += criterion(out['preds'], x['label'])

            # backward 
            model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tot_loss += loss.detach().cpu().item()
        
    return tot_loss/p

def test_cls(args, 
          model, 
          test_loader, 
          criterion, 
          device
          ):
    
    # te_pred_loss = 0
    # te_imp_loss = 0
    te_imp_pred_loss_num = 0
    te_imp_pred_loss_cat = 0
    # te_tot_loss = 0
    
    labels = np.array([np.arange(args.n_labels)])    
    cm = np.zeros((args.n_labels, args.n_labels))

    for batch_idx, x in enumerate(test_loader):
        x['input'], x['mask'], x['label']\
            = x['input'].to(device), x['mask'].to(device), x['label'].to(device)
        
        x['complete_input'] = x['complete_input'].to(device) if x['complete_input'] is not None\
            else None 

        model.eval()
        loss = 0
        with torch.no_grad():
            out = model(x, numobs= args.vai_n_samples, cat_features= args.cat_vars_pos)
            loss_imp_num, loss_imp_cat = get_loss_imp(x['input'], out['imputation'], x['mask'], cat_features= args.cat_vars_pos, test= True)
            if args.model_type != 'ipv':
                preds = torch.argmax(F.softmax(out['preds'], dim=1), dim=1)
            else: 
                preds, _ = torch.mode(torch.argmax(torch.softmax(out['preds'], dim=2), dim=2), dim=0) 

            # preds = out['preds']
            # loss = criterion(out['preds'], x['label'])
            # loss_reg = 0.
            imp_pred_loss_num, imp_pred_loss_cat = 0., 0.
            # if out['regularization_loss'] is not None: 
            #     loss_reg += args.imp_loss_penalty * out['regularization_loss']
            # tot_loss = loss + args.imp_loss_penalty * loss_imp + loss_reg
            if x['complete_input'] is not None: 
                l1, l2 = get_loss_imp(x['complete_input'], out['imputation'], 1-x['mask'], cat_features= args.cat_vars_pos, test=True)
                imp_pred_loss_num += l1
                imp_pred_loss_cat += l2
        # loss  
        # te_tot_loss += tot_loss.detach().cpu().item()
        te_imp_pred_loss_num += imp_pred_loss_num.detach().cpu().item() if not isinstance(imp_pred_loss_num, float) else float('nan')
        # te_pred_loss += loss.detach().cpu().item()
        te_imp_pred_loss_cat += imp_pred_loss_cat.detach().cpu().item() if not isinstance(imp_pred_loss_cat, float) else float('nan')

        # confusion matrix
        preds = preds.detach().cpu().numpy()
        cm += confusion_matrix(x['label'].detach().cpu().numpy(), preds, labels= labels)
    
    acc, rec, prec, f1 = evaluate(cm, weighted= False) 
    # te_tot_loss = te_tot_loss/len(test_loader)
    # te_imp_loss = te_imp_loss/len(test_loader)
    # te_pred_loss = te_pred_loss/len(test_loader)
    te_imp_pred_loss_num = te_imp_pred_loss_num/len(test_loader) 
    te_imp_pred_loss_cat = te_imp_pred_loss_cat/len(test_loader)

    print("Test done!")
    # print(f"test total loss: {te_tot_loss:.2f}")
    # print(f"test imputation loss: {te_imp_loss:.2f}")
    # print(f"test prediction loss: {te_pred_loss:.2f}")
    print(f"test imputation prediction loss (numeric) {te_imp_pred_loss_num:.2f}")
    print(f"test imputation prediction loss (categorical) {te_imp_pred_loss_cat:.2f}")
    # print() 
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    cm_file = os.path.join(args.model_path, f"confusion_matrix.png")
    plt.savefig(cm_file)
    plt.show()
    print(f"정확도 (accuracy): {acc:.2f}   ")
    print(f"재현율 (recall): {rec:.2f}   ")
    print(f"정밀도 (precision): {prec:.2f}   ")
    print(f"F1 score: {f1:.2f}   ")
    print()   

    perf = {
        'acc': acc,
        'rec': rec,
        'prec': prec,
        'f1': f1, 
        'imp_num_error':te_imp_pred_loss_num,
        'imp_cat_error':te_imp_pred_loss_cat 
    }

    return perf 

def test_regr(args, 
          model, 
          test_loader, 
          criterion, 
          device
          ):
    
    te_loss_imp = 0
    te_loss_preds = 0
    te_loss_tot = 0
    te_imp_pred_loss_num = 0
    te_imp_pred_loss_cat = 0
    te_r2 = 0
    te_mae = 0
    te_mse = 0
    
    for batch_idx, x in enumerate(test_loader):
        x['input'], x['mask'], x['label'] \
            = x['input'].to(device), x['mask'].to(device), x['label'].to(device)
        
        x['complete_input'] = x['complete_input'].to(device) if x['complete_input'] is not None\
            else None 
        
        model.eval()
        loss = 0
        imp_pred_loss_num = 0
        imp_pred_loss_cat = 0
        with torch.no_grad():
            out = model(x, cat_features= args.cat_vars_pos)
            loss_imp, _ = get_loss_imp(x['input'], out['imputation'], x['mask'], cat_features= args.cat_vars_pos)
            loss = criterion(out['preds'], x['label'])
            loss_reg = 0. 
            if out['regularization_loss'] is not None: 
                loss_reg += args.imp_loss_penalty * out['regularization_loss']
            tot_loss = loss + args.imp_loss_penalty * loss_imp + loss_reg
            if x['complete_input'] is not None: 
                l1, l2 = get_loss_imp(x['complete_input'], out['imputation'], 1-x['mask'], cat_features= args.cat_vars_pos, test= True)        
                imp_pred_loss_num += l1 
                imp_pred_loss_cat += l2
        te_loss_imp += loss_imp.detach().cpu().numpy()
        te_loss_preds += loss.detach().cpu().numpy()
        te_loss_tot += tot_loss.detach().cpu().numpy()
        te_imp_pred_loss_num += imp_pred_loss_num.detach().cpu().item()
        te_imp_pred_loss_cat += imp_pred_loss_cat.detach().cpu().item()

        te_r2 += r2_score(x['label'].detach().cpu().numpy(),out['preds'].detach().cpu().numpy())
        te_mae += mean_absolute_error(out['preds'].detach().cpu().numpy(), x['label'].detach().cpu().numpy()) 
        te_mse += mean_squared_error(out['preds'].detach().cpu().numpy(), x['label'].detach().cpu().numpy()) 

    te_loss_imp = te_loss_imp/len(test_loader)
    te_loss_preds = te_loss_preds/len(test_loader)
    te_loss_tot = te_loss_tot/len(test_loader)
    te_imp_pred_loss_num = te_imp_pred_loss_num/len(test_loader)
    te_imp_pred_loss_cat = te_imp_pred_loss_cat/len(test_loader)
    te_r2 = te_r2/len(test_loader)
    te_mae = te_mae/len(test_loader)
    te_mse = te_mse/len(test_loader)
    print("Test done!")
    print(f"imputation loss: {te_loss_imp:.2f}")
    print(f"prediction loss: {te_loss_preds:.2f}")
    print(f"total loss: {te_loss_tot:.2f}")
    print(f"test imputation prediction loss (numeric) {te_imp_pred_loss_num:.2f}")
    print(f"test imputation prediction loss (categorical) {te_imp_pred_loss_cat:.2f}")
    print(f"r2: {te_r2:.2f}")
    print(f"mae: {te_mae:.2f}")
    print(f"mse: {te_mse:.2f}")
    print()    

    perf = {
        'r2': te_r2,
        'mae': te_mae,
        'mse': te_mse,
        'imp_num_error':te_imp_pred_loss_num,
        'imp_cat_error':te_imp_pred_loss_cat 
    }

    return perf 

    
def test_imp(args, 
          model, 
          test_loader, 
          criterion, 
          device
          ):
    
    te_loss_imp = 0
    te_imp_pred_loss_num = 0
    te_imp_pred_loss_cat = 0
    
    for batch_idx, x in enumerate(test_loader):
        x['input'], x['mask'], x['label'] \
            = x['input'].to(device), x['mask'].to(device), x['label'].to(device)
        
        x['complete_input'] = x['complete_input'].to(device) if x['complete_input'] is not None\
            else None 
        
        model.eval()
        loss = 0
        imp_pred_loss_num = 0
        imp_pred_loss_cat = 0
        with torch.no_grad():
            out = model(x, cat_features= args.cat_vars_pos)
            loss_imp, _ = get_loss_imp(x['input'], out['imputation'], x['mask'], cat_features= args.cat_vars_pos)
            loss = criterion(out['preds'], x['label'])
            loss_reg = 0. 
            if out['regularization_loss'] is not None: 
                loss_reg += args.imp_loss_penalty * out['regularization_loss']
            tot_loss = loss + args.imp_loss_penalty * loss_imp + loss_reg
            if x['complete_input'] is not None: 
                l1, l2 = get_loss_imp(x['complete_input'], out['imputation'], 1-x['mask'], cat_features= args.cat_vars_pos, test= True)        
                imp_pred_loss_num += l1 
                imp_pred_loss_cat += l2
        te_loss_imp += loss_imp.detach().cpu().numpy()

        te_imp_pred_loss_num += imp_pred_loss_num.detach().cpu().item()
        te_imp_pred_loss_cat += imp_pred_loss_cat.detach().cpu().item()

    te_loss_imp = te_loss_imp/len(test_loader)
    te_imp_pred_loss_num = te_imp_pred_loss_num/len(test_loader)
    te_imp_pred_loss_cat = te_imp_pred_loss_cat/len(test_loader)

    print("Test done!")
    print(f"imputation loss: {te_loss_imp:.2f}")
    print(f"test imputation prediction loss (numeric) {te_imp_pred_loss_num:.2f}")
    print(f"test imputation prediction loss (categorical) {te_imp_pred_loss_cat:.2f}")
    print()    

    perf = {
        'imp_error': te_loss_imp,
        'imp_num_error':te_imp_pred_loss_num,
        'imp_cat_error':te_imp_pred_loss_cat 
    }

    return perf 