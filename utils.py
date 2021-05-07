import torch 
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle


def import_data(path, sheet=""):
    """Import and clean from excel"""
    df = pd.read_excel(
        path,
        sheet_name=sheet
    )
    
    df.drop(0, inplace=True)
    df.rename(columns={"Unnamed: 0":"Date"}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df.set_index("Date", drop=True, inplace=True)
    return df

def split_tr_te(df, size=0.8, shuffle=True, y_name=""):
    """Split data into training and testing set"""
    if shuffle:
        df = df.sample(frac=1)
    
    df.reset_index(drop=True, inplace=True)
    split = int(df.shape[0] * size)
    train = df[:split].reset_index(drop=True)
    test = df[split:].reset_index(drop=True)
    
    X_tr = train.drop(columns=[y_name])
    X_te = test.drop(columns=[y_name])
    y_tr = train[y_name].values.ravel()
    y_te = test[y_name].values.ravel()
    return X_tr, X_te, y_tr, y_te


class StandardScaler(object):
    def __init__(self, X):
        self.X = X
        self.mean = self.X.mean(axis=0)
        self.std = self.X.std(axis=0)
    def transform(self, X):
        return (X - self.mean) / self.std
    def reverse_transform(self, X_sc):
        return X_sc * self.std + self.mean
    
class MinMaxScaler(object):
    def __init__(self, X, feature_range=(-1, 1)):
        assert feature_range[1] > feature_range[0]
        
        self.X = X
        self.maxRng = X.max(axis=0)
        self.minRng = X.min(axis=0)
        
        self.u = feature_range[1]
        self.d = feature_range[0]
    def transform(self, X):
        x_std = (X - self.minRng)/(self.maxRng-self.minRng)
        x_sc = x_std * (self.u - self.d) + self.d
        return x_sc 
    
    def reverse_transform(self, X_sc):
        x_std = (X_sc - self.d) / (self.u - self.d)
        x = x_std * (self.maxRng-self.minRng) +  self.minRng
        return x
    
class PCA(object):
    def __init__(self, X, comp_prop=1.0):       
        self.comp_prop = comp_prop
        
        if hasattr(X, "values"):
            self.X=X.values
        else:
            self.X=X
        
        self.n_comp = int(np.floor(self.comp_prop*self.X.shape[1]))
        self.cov = np.cov(X.T)
        self.eigVal, self.eigVec = np.linalg.eig(self.cov)
        self.W = (self.eigVec.T[:][:self.n_comp]).T
            
        
    def explained(self):
        # Calculating the explained variance on each component
        expl = self.eigVal / np.sum(self.eigVal) * 100
        cum_expl = np.cumsum(expl)

        fig, ax = plt.subplots()
        ax.plot(np.arange(0, self.X.shape[1]), cum_expl)
        ax.set_xlabel("N component")
        ax.set_ylabel("Explained variance")
        ax.title.set_text("Principal Component Analysis")
        plt.show()
        return cum_expl
    
    def transform(self, X):      
        if hasattr(X, "values"):
            X=X.values
        else:
            X=X
            
        X_pca = np.dot(X, self.W)
        colNames = [f"comp_{x}" for x in range(0, self.n_comp)]
        return pd.DataFrame(X_pca, columns=colNames)
    
    def reverse_transform(self, X_pca):
        if hasattr(X_pca, "values"):
            X_pca=X_pca.values
        else:
            X_pca=X_pca

        X = X_pca @ self.W @ self.W.T
        return X
            
    
class evaluateRegressor(object):
    """Give regression metrics numpy"""
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
    def MAE(self):
        return np.mean(np.abs(self.y_true - self.y_pred))
    def MSE(self):
        return np.mean((self.y_true-self.y_pred)**2)
    def RMSE(self):
        return np.sqrt(self.MSE())    
    def r2score(self):
        u = np.sum((self.y_true - self.y_pred) ** 2)
        d = np.sum((self.y_true - self.y_pred.mean()) ** 2)
        return 1 - u / d
    
    def toString(self):
        mae = self.MAE()
        mse = self.MSE()
        rmse = self.RMSE()
        r2 = self.r2score()
        print("-"*40)
        print(f"MAE \t| MSE \t| RMSE \t| r^2 \t|")
        print("-"*40)
        print(f"{mae:.4f}  {mse:.4f}  {rmse:.4f}  {r2:.4f}")
        print("-"*40)
        
        
def plot_training(metrics, log=False):
    fig, ax = plt.subplots(2, 2, figsize=(15,10))
    
    for axis, metric in zip(ax.flatten(), metrics.keys()):

        axis.plot(metrics[metric])
        if log:
            axis.set_yscale("log")
        axis.title.set_text(f"{metric} | log : {log}")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Metric")
        plt.tight_layout()
    plt.show()
    
    return