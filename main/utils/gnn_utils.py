import torch 
import numpy as np

from sklearn.linear_model import LassoCV 
from sklearn.linear_model import LogisticRegressionCV

from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier

def get_prior_relation(X, 
                    numeric_vars_pos:list, cat_vars_pos:list, 
                    cv:int= 5, 
                    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')): 

    n, p = X.shape 
    adj_mat = np.zeros((p,p))

    for i in range(p): 
        col_index = np.full((p,), True) 
        col_index[i] = False
        if i in numeric_vars_pos:
            lasso = LassoCV(cv=cv, random_state=0)
            reg = lasso.fit(X[:, col_index], X[:, ~col_index].flatten())
        else: 
            clf = LogisticRegressionCV(cv=cv, random_state=0, max_iter= 1000)
            reg = clf.fit(X[:, col_index], X[:, ~col_index].flatten())            
        adj_mat[i, col_index] = (reg.coef_ > 1e-1) *1

    relation_index = (torch.nonzero(torch.LongTensor(adj_mat)).T).to(device)

    return relation_index

def get_prior_relation_by_tree(X, 
                    numeric_vars_pos:list, cat_vars_pos:list, 
                    cv:int= 5, 
                    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')): 

    n, p = X.shape 
    adj_mat = np.zeros((p,p))
    # select features that have importance greater than average
    # feature importance = mean decrease in impurity
    for i in range(p): 
        col_index = np.full((p,), True) 
        col_index[i] = False
        if i in numeric_vars_pos:
            selector = SelectFromModel(estimator=DecisionTreeRegressor()).fit(X[:, col_index], X[:, ~col_index].flatten())
        else: 
            selector = SelectFromModel(estimator=DecisionTreeClassifier()).fit(X[:, col_index], X[:, ~col_index].flatten())         
        adj_mat[i, col_index] = selector.get_support() * 1

    relation_index = (torch.nonzero(torch.LongTensor(adj_mat)).T).to(device)

    return relation_index

def get_prior_adj_by_tree(X, 
                    numeric_vars_pos:list, cat_vars_pos:list, 
                    cv:int= 5, 
                    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')): 

    n, p = X.shape 
    adj_mat = np.zeros((p,p))

    for i in range(p): 
        col_index = np.full((p,), True) 
        col_index[i] = False
        if i in numeric_vars_pos:
            selector = SelectFromModel(estimator=DecisionTreeRegressor()).fit(X[:, col_index], X[:, ~col_index].flatten())
        else: 
            selector = SelectFromModel(estimator=DecisionTreeClassifier()).fit(X[:, col_index], X[:, ~col_index].flatten())         
        adj_mat[i, col_index] = selector.get_support() * 1

    prior_adj = (torch.LongTensor(adj_mat).T).to(device)

    return prior_adj