import torch 
from torch import nn
from torch.nn import functional as F
from utils.utils import get_perf_cat, get_perf_num
from fancyimpute import SoftImpute

class GraphSamplingLayer(nn.Module): 
    r"""Graph Sampling Layer 
    return (directed) logits
    """
    def __init__(self, num_features:int, embedding_dim:int, alpha:float, prior_adj, device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')): 
        super().__init__() 
        self.emb1 = nn.Embedding(num_features, embedding_dim)
        self.theta1 = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.emb2 = nn.Embedding(num_features, embedding_dim)
        self.theta2 = nn.Linear(embedding_dim, embedding_dim, bias= False)
        
        self.alpha = alpha
        self.prior_adj = prior_adj.to(device)
        self.device = device 
        self.criterion = nn.BCELoss()

    def forward(self): 
        emb1 = self.emb1.weight
        emb2 = self.emb2.weight
        emb1 = self.alpha*torch.tanh(self.theta1(emb1))
        emb2 = self.alpha*torch.tanh(self.theta2(emb2))
        logits = emb1@emb2.T
        return logits

    def train_step(self): 
        logits = self.forward()
        probs = torch.sigmoid(logits)
        loss = self.criterion(probs, self.prior_adj*1.) 
        return {
            'total_loss':loss
        } 

    @torch.no_grad()
    def val_step(self): 
        logits = self.forward()
        probs = torch.sigmoid(logits)
        loss = self.criterion(probs, self.prior_adj*1.) 
        return {
            'total_loss':loss
        } 

    @torch.no_grad()
    def test_step(self): 
        logits = self.forward()
        probs = torch.sigmoid(logits)
        loss = self.criterion(probs, self.prior_adj*1.) 
        return {
            'total_loss':loss
        } 

class GraphConvolutionLayer(nn.Module): 

    def __init__(self, in_features:int, act_func = None):
        super().__init__() 
        self.w = nn.Parameter(torch.randn((in_features, in_features), requires_grad= True))
        self.b = nn.Parameter(torch.zeros((in_features, ), requires_grad = True))
        self.act_func = act_func

        self.init_params()

    def init_params(self): 
        nn.init.xavier_normal_(self.w)

    def forward(self, x, adj_mat): 
        res = x 
        x = (x@self.w + self.b)@adj_mat
        if self.act_func is None:
            return res + x
        else: 
            return self.act_func(res+x)

class PredictionHead(nn.Module): 

    def __init__(self, in_features:int, out_features:int, numeric_vars_pos:list, cat_vars_pos:list, task_type:str): 
        super().__init__() 
        self.w = nn.Parameter(torch.randn((in_features, out_features), requires_grad= True))
        self.b = nn.Parameter(torch.zeros((out_features, ), requires_grad = True))

        self.init_params()

        self.numeric_vars_pos = numeric_vars_pos
        self.cat_vars_pos = cat_vars_pos
        self.vars_pos = numeric_vars_pos + cat_vars_pos
        self.task_type = task_type

    def init_params(self): 
        nn.init.xavier_normal_(self.w)
    
    def forward(self, x, adj_mat= None): 
        if adj_mat is None: 
            return x[:, self.vars_pos]@self.w + self.b
        else: 
            return x[:, self.vars_pos]@(self.w*adj_mat) + self.b
    
    def train_step(self, batch): 
        batch_x, batch_mask = batch.get('complete_input'), batch.get('mask')
        y_hat = self.forward(batch_x)
        prediction_loss = F.mse_loss(y_hat.ravel(), batch.get('label')) if self.task_type == 'regr' else F.cross_entropy(y_hat, batch.get('label'))
        return {
            'total_loss': prediction_loss
        }

    @torch.no_grad()
    def val_step(self, batch): 
        batch_x, batch_mask = batch.get('complete_input'), batch.get('mask')
        y_hat = self.forward(batch_x)
        prediction_loss = F.mse_loss(y_hat.ravel(), batch.get('label')) if self.task_type == 'regr' else F.cross_entropy(y_hat, batch.get('label'))
        return {
            'total_loss': prediction_loss
        }

# Proposed model 
class GraphImputer(nn.Module):
    def __init__(self, 
                num_features:int,
                graph_emb_dim:int, 
                num_labels:int,
                cat_vars_pos:list= [], 
                numeric_vars_pos:list= [],
                num_layers:int= 1,
                alpha:float= 3., 
                imp_loss_penalty:float = 1.,
                reg_loss_peanlty:float = 0.1,
                prior_adj = None, 
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                task_type: str = 'cls',
                init_impute = None
                ):
        super().__init__() 

        # layers 
        # graph sampling layer
        if prior_adj is None: 
            prior_adj = torch.ones((num_features, num_features), device= device)
        self.graph_sampling = GraphSamplingLayer(num_features, graph_emb_dim, alpha, prior_adj, device)
        
        # GraphAutoEncoder layer
        for i in range(num_layers-1):
            setattr(self, f'gc{i}', GraphConvolutionLayer(num_features, F.leaky_relu))
        setattr(self, f'gc{num_layers-1}', GraphConvolutionLayer(num_features, None))

        # prediction layer 
        self.prediction_head = PredictionHead(num_features, num_labels, numeric_vars_pos, cat_vars_pos, task_type)

        # loss
        self.mse_loss = nn.MSELoss()
        self.cls_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss()

        # attributes 
        self.num_features = num_features
        self.num_labels = num_labels 
        self.cat_vars_pos = cat_vars_pos 
        self.numeric_vars_pos = numeric_vars_pos
        self.alpha = alpha
        self.prior_adj = prior_adj.to(device)

        self.device = device 
        self.task_type = task_type

        self.num_layers = num_layers 
        self.imp_loss_penalty = imp_loss_penalty 
        self.reg_loss_peanlty = reg_loss_peanlty

        if init_impute is None:
            self.init_impute = SoftImpute(verbose= False)
        else: 
            self.init_impute = init_impute

    # batch {input, mask, label, complete_input}
    # input, mask = make_mask(complete_input)
    def forward(self, batch):
        # inputs
        batch_x, batch_mask = batch.get('input'), batch.get('mask')
        batch_x = torch.FloatTensor(self.init_impute.fit_transform(batch_x.detach().cpu())).to(self.device)

        # graph sampling and making an adjacency matrix
        logits = self.graph_sampling() # num_features x num_features 
        adj_mat = torch.sigmoid(logits)
        adj_mat_norm = self.norm_adj(adj_mat)
        
        for i in range(self.num_layers-1):
            batch_x_recon = getattr(self, f'gc{i}')(batch_x, adj_mat_norm)
        batch_x_recon = getattr(self, f'gc{self.num_layers-1}')(batch_x, adj_mat_norm)

        batch_x_recon_num = torch.tanh(batch_x_recon[:, self.numeric_vars_pos]) if len(self.numeric_vars_pos) > 0 else None
        batch_x_recon_cat = torch.sigmoid(batch_x_recon[:, self.cat_vars_pos]) if len(self.cat_vars_pos) > 0 else None 

        # trim the inputs
        assert batch_x_recon_cat is not None or batch_x_recon_num is not None, "Inputs should not be None, one of categorical- or numeric- variable should be a proper input."
        if batch_x_recon_cat is not None and batch_x_recon_num is not None:
            batch_x_hat = torch.cat([batch_x_recon_num, batch_x_recon_cat], dim= 1) 
        elif batch_x_recon_cat is not None and batch_x_recon_num is None: 
            batch_x_hat = batch_x_recon_cat 
        elif batch_x_recon_num is not None and batch_x_recon_cat is None: 
            batch_x_hat = batch_x_recon_num

        if self.training:
            y_hat = self.prediction_head(batch.get('complete_input')[:, self.numeric_vars_pos + self.cat_vars_pos]) 
            if self.task_type == 'regr': 
                y_hat = y_hat.ravel()
        else: 
            y_hat = self.prediction_head(batch_x_hat) 
            if self.task_type == 'regr': 
                y_hat = y_hat.ravel()            
        
        return {
            'x_recon_num': batch_x_recon_num,
            'x_recon_cat': batch_x_recon_cat,
            'adj_mat': adj_mat,
            'x_imputed': batch_x_hat,
            'y_hat': y_hat
        }
    
    @torch.no_grad()
    def get_adj(self): 
        logits = self.graph_sampling()
        adj_mat = torch.sigmoid(logits) 
        adj_mat = (adj_mat > 0.5) * 1. 
        return adj_mat.detach().cpu().numpy()

    def train_step(self, batch): 
        # returns the training loss 
        # (1) feed forward
        # with torch.set_grad_enabled(True)
        # out = {x_recon, adj_mat, x_imputed, y_hat}
        out = self.forward(batch)
        num, cat = out.get('x_recon_num'), out.get('x_recon_cat')
        
        # imputation loss 
        num_imp_loss = self.mse_loss(num, batch.get('complete_input')[:, self.numeric_vars_pos]) if num is not None else float('nan')
        cat_imp_loss = self.bce_loss(cat, batch.get('complete_input')[:, self.cat_vars_pos]) if cat is not None else float('nan')

        # prediction loss 
        prediction_loss = self.mse_loss(out.get('y_hat').ravel(), batch.get('label')) if self.task_type == 'regr' else self.cls_loss(out.get('y_hat'), batch.get('label'))
        
        # regularization loss 
        regularization_loss =  self.bce_loss(out['adj_mat'], self.prior_adj*1.) 

        return {
            'num_imp_loss': num_imp_loss,
            'cat_imp_loss': cat_imp_loss,
            'prediction_loss': prediction_loss,
            'regularization_loss': regularization_loss
        } 

    @torch.no_grad()
    def val_step(self, batch): 
        # with torch.no_grad()
        out = self.forward(batch)
        num, cat = out.get('x_recon_num'), out.get('x_recon_cat')
        
        # imputation loss 
        num_imp_loss = self.mse_loss(num, batch.get('complete_input')[:, self.numeric_vars_pos]) if num is not None else float('nan')
        cat_imp_loss = self.bce_loss(cat, batch.get('complete_input')[:, self.cat_vars_pos]) if cat is not None else float('nan')

        # prediction loss 
        prediction_loss = self.mse_loss(out.get('y_hat').ravel(), batch.get('label')) if self.task_type == 'regr' else self.cls_loss(out.get('y_hat'), batch.get('label'))

        # regularization loss 
        regularization_loss =  self.bce_loss(out['adj_mat'], self.prior_adj*1.) 

        # perf measure 
        perfs = get_perf_num(out.get('y_hat').ravel(), batch.get('label')) if self.task_type == 'regr'\
                                 else get_perf_cat(out.get('y_hat'), batch.get('label'), self.num_labels)
        
        perfs['num_imp_loss'] = num_imp_loss
        perfs['cat_imp_loss'] = cat_imp_loss
        perfs['prediction_loss'] = prediction_loss
        perfs['regularization_loss'] = regularization_loss
        
        return perfs

    @torch.no_grad()
    def test_step(self, batch): 
        # with torch.no_grad()
        out = self.forward(batch)
        num, cat = out.get('x_recon_num'), out.get('x_recon_cat')
        
        # imputation loss 
        num_imp_loss = self.mse_loss(num, batch.get('complete_input')[:, self.numeric_vars_pos]).detach().cpu().item() if num is not None else float('nan')
        cat_imp_loss = self.bce_loss(cat, batch.get('complete_input')[:, self.cat_vars_pos]).detach().cpu().item() if cat is not None else float('nan')

        # prediction loss 
        prediction_loss = self.mse_loss(out.get('y_hat').ravel(), batch.get('label')).detach().cpu().item() if self.task_type == 'regr' \
            else self.cls_loss(out.get('y_hat'), batch.get('label')).detach().cpu().item()

        # regularization loss 
        regularization_loss =  self.bce_loss(out['adj_mat'], self.prior_adj*1.).detach().cpu().item()
        # perf measure 
        perfs = get_perf_num(out.get('y_hat').ravel(), batch.get('label')) if self.task_type == 'regr'\
                                 else get_perf_cat(out.get('y_hat'), batch.get('label'), self.num_labels)
        
        perfs['num_imp_loss'] = num_imp_loss
        perfs['cat_imp_loss'] = cat_imp_loss
        perfs['prediction_loss'] = prediction_loss
        perfs['regularization_loss'] = regularization_loss

        return perfs

    def norm_adj(self, adj_mat): 
        return adj_mat/adj_mat.sum(dim=0, keepdim=True)

