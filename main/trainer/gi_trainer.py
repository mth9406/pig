import torch 
import numpy as np
import math

from utils.utils import make_missing

class GraphImputerTrainer: 

    def __init__(self, missing_prob:float = 0.2, print_every:int = 5): 
        super().__init__() 
        self.missing_prob = missing_prob 
        self.print_every = print_every
        self.logs = {
            'tr_loss':[],
            'valid_loss':[]
        }

    def __call__(self, args, model, 
                train_loader, valid_loader, 
                early_stopping, 
                optimizer, scheduler=None, 
                device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

        for epoch in range(args.epoch): 
            training_loss, validation_loss = [], []
            training_weights, validation_weights = [], []    
            validation_perf = []        
            # training loop
            model.train()
            for batch_idx, batch in enumerate(train_loader): 
                # randomly make missing values here!
                x_tilde, mask = make_missing(batch['complete_input'], self.missing_prob)
                batch['input'] = x_tilde
                batch['mask']  = mask 
                for k, v in batch.items():
                    batch[k] = batch[k].to(device)
                
                with torch.set_grad_enabled(True): 
                    out = model.train_step(batch) 
                    tr_loss = self.get_total_loss(out, model.imp_loss_penalty, model.reg_loss_peanlty)
                    model.zero_grad()
                    optimizer.zero_grad()
                    tr_loss.backward()
                    optimizer.step() 
                training_loss.append(tr_loss.detach().cpu().item())
                training_weights.append(len(batch['input']))
                
            # validation loop 
            model.eval()
            for batch_idx, batch in enumerate(valid_loader): 
                for k, v in batch.items():
                    batch[k] = batch[k].to(device)
                
                out = model.val_step(batch) 
                val_loss = self.get_total_loss(out, model.imp_loss_penalty, model.reg_loss_peanlty)
                validation_loss.append(val_loss.detach().cpu().item())
                validation_weights.append(len(batch['input']))
                if model.task_type == 'cls':
                    validation_perf.append(out.get('f1_score')) 
                else: 
                    validation_perf.append(out.get('r2'))

            training_loss = np.average(training_loss, weights = training_weights)
            validation_loss = np.average(validation_loss, weights = validation_weights)
            validation_perf = np.average(validation_perf, weights = validation_weights)
            
            # save the logs 
            self.logs.get('tr_loss').append(training_loss)
            self.logs.get('valid_loss').append(validation_loss)

            if epoch % self.print_every == 0:
                prefix = f'Epoch [{epoch+1}/{args.epoch}]'
                space = ' '*len(prefix)
                print(f'{prefix}: training loss= {training_loss:.6f}, validation loss= {validation_loss:.6f}')
                if model.task_type == 'regr':
                    print(f'{space}: validation perf = {validation_perf:.6f}')
                else: 
                    print(f'{space}: validation perf = {validation_perf:.6f}')
            
            # early stopping 
            early_stopping(validation_loss, model, epoch, optimizer)
            if early_stopping.early_stop:
                break  

    def pretrain_graph(self, args, model, 
                optimizer, 
                device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

        model.train()
        for epoch in range(args.pretrain_epoch):    
            # training loop
            with torch.set_grad_enabled(True): 
                tr_loss = getattr(model, 'graph_sampling').train_step()['total_loss']
                model.zero_grad()
                optimizer.zero_grad()
                tr_loss.backward()
                optimizer.step() 
    
            if epoch % self.print_every == 0:
                prefix = f'Epoch [{epoch+1}/{args.pretrain_epoch}]'
                print(f'{prefix}: training loss= {tr_loss.detach().cpu().item():.6f}')

    def pretrain_task(self, args, model, 
                train_loader, valid_loader,  
                optimizer,
                device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

        for epoch in range(args.pretrain_epoch): 
            training_loss, validation_loss = [], []
            training_weights, validation_weights = [], []             
            # training loop
            model.train()
            for batch_idx, batch in enumerate(train_loader): 
                for k, v in batch.items(): 
                    batch[k] = batch[k].to(device)
                num_batches = batch['complete_input'].shape[0]
                with torch.set_grad_enabled(True): 
                    tr_loss = getattr(model, 'prediction_head').train_step(batch)['total_loss']
                    model.zero_grad()
                    optimizer.zero_grad()
                    tr_loss.backward()
                    optimizer.step() 
                training_loss.append(tr_loss.detach().cpu().item())
                training_weights.append(num_batches)

            training_loss = np.average(training_loss, weights = training_weights)
            if epoch % self.print_every == 0:
                prefix = f'Epoch [{epoch+1}/{args.pretrain_epoch}]'
                print(f'{prefix}: training loss= {training_loss:.6f}')

    def test(self, args, model, test_loader, device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')): 
        perfs = {}        
        weights = []
        for batch_idx, batch in enumerate(test_loader): 
            for k, v in batch.items(): 
                batch[k] = batch[k].to(device)
            model.eval() 
            out = model.test_step(batch)
            num_batch = len(batch['input'])
            weights.append(num_batch)
            for k, v in out.items(): 
                if perfs.get(k) is None: 
                    perfs[k] = []
                perfs.get(k).append(v) 

        for k, v in perfs.items(): 
            perfs[k] = np.average(perfs.get(k), weights = weights)
        
        return perfs 

    def get_total_loss(self, out:dict, imp_loss_penalty:float, reg_loss_peanlty:float): 
        total_loss = 0

        # imputation loss of numeric variables
        num_imp_loss = out.get('num_imp_loss')
        if num_imp_loss is not None and not math.isnan(num_imp_loss): 
            total_loss += num_imp_loss * imp_loss_penalty
        
        # imputation loss of categorical variables
        cat_imp_loss = out.get('cat_imp_loss')
        if cat_imp_loss is not None and not math.isnan(cat_imp_loss): 
            total_loss += cat_imp_loss * imp_loss_penalty

        # prediction loss 
        prediction_loss = out.get('prediction_loss')
        if prediction_loss is not None and not math.isnan(prediction_loss): 
            total_loss += prediction_loss
        
        # regularization loss 
        regularization_loss = out.get('regularization_loss')
        if regularization_loss is not None and not math.isnan(regularization_loss): 
            total_loss += regularization_loss * reg_loss_peanlty

        return total_loss



