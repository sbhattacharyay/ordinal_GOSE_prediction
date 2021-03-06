# Fundamental libraries
import os
import sys
import time
import glob
import random
import datetime
import warnings
import itertools
import numpy as np
import pandas as pd
import pickle as cp
import seaborn as sns
import multiprocessing
from scipy import stats
from pathlib import Path
from ast import literal_eval
import matplotlib.pyplot as plt
from collections import Counter
warnings.filterwarnings(action="ignore")

# PyTorch, PyTorch.Text, and Lightning-PyTorch methods
import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import Vocab
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# SciKit-Learn methods
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

# deep learning CPM model
class CPM_deep(pl.LightningModule):
    def __init__(self,in_features,layers,neurons,dropout,output_activation,learning_rate,class_weights,targets):
        """
        Args:
            in_features (int): number of input features
            layers (int): number of hidden layers in feed forward neural network
            neurons (list of length layers): the number of neurons in each layer
            dropout (flaot): the proportion of each dense layer dropped out during training
            output_activation (string): 'softmax' for DeepMN or 'sigmoid' for DeepOR
            learning_rate (float): Learning rate for ADAM optimizer
            class_weights (boolean): identifies whether loss should be weighted against class frequency
            targets (NumPy array): if class_weights == True, provides the class labels of the training set
        """
        super(CPM_deep, self).__init__()
        
        self.save_hyperparameters()
        
        self.in_features = in_features
        self.layers = layers
        self.neurons = neurons
        self.dropout = dropout
        self.output_activation = output_activation
        self.learning_rate = learning_rate
        self.class_weights = class_weights
        self.targets = targets
        
        self.hidden_layers = nn.ModuleList()
        
        self.hidden_layers.append(nn.Linear(self.in_features, self.neurons[0]))
        self.hidden_layers.append(nn.Dropout(self.dropout))

        for i in range(1,self.layers):
            self.hidden_layers.append(nn.Linear(self.neurons[i-1],self.neurons[i]))
            self.hidden_layers.append(nn.Dropout(self.dropout))
        
        if self.output_activation == 'softmax': 
            self.hidden2gose = nn.Linear(self.neurons[-1], 7)
        elif self.output_activation == 'sigmoid': 
            self.hidden2gose = nn.Linear(self.neurons[-1],6)
        else:
            raise ValueError("Invalid output activation type. Must be 'softmax' or 'sigmoid'")
            
    def forward(self,x):
        
        for f in self.hidden_layers:
            x = f(F.relu(x.float()))
        
        if self.output_activation == 'softmax': 
            return self.hidden2gose(x)
        elif self.output_activation == 'sigmoid': 
            y_int = self.hidden2gose(x)
            mod_out = -F.relu(y_int.clone()[:,1:6])
            y_int[:,1:6] = mod_out
            return y_int.cumsum(dim=1)
        else:
            raise ValueError("Invalid output activation type. Must be 'softmax' or 'sigmoid'")
            
    def training_step(self, batch, batch_idx):
        
        # Get information from current batch
        x, y = batch
        
        # Collect current model state outputs for the batch
        yhat = self(x)
        
        # Calculate loss based on the output activation type
        if self.output_activation == 'softmax': 
            
            if self.class_weights:
                bal_weights = torch.from_numpy(compute_class_weight(class_weight='balanced',
                                                                    classes=np.sort(np.unique(self.targets)),
                                                                    y=self.targets)).type_as(yhat)
                loss = F.cross_entropy(yhat, y, weight = bal_weights)
            else:
                loss = F.cross_entropy(yhat, y)
                
        elif self.output_activation == 'sigmoid': 
            
            if self.class_weights:
                bal_weights = torch.from_numpy((self.targets.shape[0]
                                                - np.sum(self.targets, axis=0))
                                               / np.sum(self.targets,
                                                        axis=0)).type_as(yhat)
                
                loss = F.binary_cross_entropy_with_logits(yhat, y.type_as(yhat), pos_weight = bal_weights)
            else:
                loss = F.binary_cross_entropy_with_logits(yhat, y.type_as(yhat))
                
        else:
            raise ValueError("Invalid output activation type. Must be 'softmax' or 'sigmoid'")
        
        return {"loss": loss, "yhat": yhat, "true_y": y}
    
    def training_epoch_end(self, training_step_outputs):
        
        comp_loss = torch.tensor([output["loss"].detach() for output in training_step_outputs]).cpu().numpy().mean()
        comp_yhats = torch.vstack([output["yhat"].detach() for output in training_step_outputs])
        comp_true_y = torch.cat([output["true_y"].detach() for output in training_step_outputs]).cpu().numpy()
        
        if self.output_activation == 'softmax': 
            curr_train_probs = F.softmax(comp_yhats).cpu().numpy()
            aucs = []
            for ix, (a, b) in enumerate(itertools.combinations(np.sort(np.unique(comp_true_y)), 2)):
                a_mask = comp_true_y == a
                b_mask = comp_true_y == b
                ab_mask = np.logical_or(a_mask,b_mask)
                condit_probs = curr_train_probs[ab_mask,b]/(curr_train_probs[ab_mask,a]+curr_train_probs[ab_mask,b]) 
                condit_probs = np.nan_to_num(condit_probs,nan=.5,posinf=1,neginf=0)
                condit_labels = b_mask[ab_mask].astype(int)
                aucs.append(roc_auc_score(condit_labels,condit_probs))            
            train_AUROC = np.mean(aucs)
            
        elif self.output_activation == 'sigmoid': 
            curr_train_probs = F.sigmoid(comp_yhats).cpu().numpy()
            comp_true_y = comp_true_y.sum(1).astype(int)
            
            train_probs = np.empty([curr_train_probs.shape[0], curr_train_probs.shape[1]+1])
            train_probs[:,0] = 1 - curr_train_probs[:,0]
            train_probs[:,-1] = curr_train_probs[:,-1]

            for col_idx in range(1,(curr_train_probs.shape[1])):
                train_probs[:,col_idx] = curr_train_probs[:,col_idx-1] - curr_train_probs[:,col_idx]                
                
            aucs = []
            for ix, (a, b) in enumerate(itertools.combinations(np.sort(np.unique(comp_true_y)), 2)):
                a_mask = comp_true_y == a
                b_mask = comp_true_y == b
                ab_mask = np.logical_or(a_mask,b_mask)
                condit_probs = train_probs[ab_mask,b]/(train_probs[ab_mask,a]+train_probs[ab_mask,b]) 
                condit_probs = np.nan_to_num(condit_probs,nan=.5,posinf=1,neginf=0)
                condit_labels = b_mask[ab_mask].astype(int)
                aucs.append(roc_auc_score(condit_labels,condit_probs))            
            train_AUROC = np.mean(aucs)
                        
        self.log('train_AUROC', train_AUROC, prog_bar=True, logger=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log('train_loss', comp_loss, prog_bar=False, logger=True, sync_dist=True, on_step=False, on_epoch=True)
    
    def validation_step(self, batch, batch_idx):
        
        # Get information from current batch
        x, y = batch
        
        # Collect current model state outputs for the batch
        yhat = self(x)
        
        val_true_y = y.cpu().numpy()
        
        # Calculate loss based on the output activation type
        if self.output_activation == 'softmax': 
                                    
            curr_val_probs = F.softmax(yhat).cpu().numpy()
            
            val_loss = F.cross_entropy(yhat, y)
            
            aucs = []
            for ix, (a, b) in enumerate(itertools.combinations(np.sort(np.unique(val_true_y)), 2)):
                a_mask = val_true_y == a
                b_mask = val_true_y == b
                ab_mask = np.logical_or(a_mask,b_mask)
                condit_probs = curr_val_probs[ab_mask,b]/(curr_val_probs[ab_mask,a]+curr_val_probs[ab_mask,b]) 
                condit_probs = np.nan_to_num(condit_probs,nan=.5,posinf=1,neginf=0)
                condit_labels = b_mask[ab_mask].astype(int)
                aucs.append(roc_auc_score(condit_labels,condit_probs))            
            val_AUROC = np.mean(aucs)
                        
        elif self.output_activation == 'sigmoid': 
            
            curr_val_probs = F.sigmoid(yhat).cpu().numpy()
            val_true_y = val_true_y.sum(1).astype(int)
            
            val_probs = np.empty([curr_val_probs.shape[0], curr_val_probs.shape[1]+1])
            val_probs[:,0] = 1 - curr_val_probs[:,0]
            val_probs[:,-1] = curr_val_probs[:,-1]

            for col_idx in range(1,(curr_val_probs.shape[1])):
                val_probs[:,col_idx] = curr_val_probs[:,col_idx-1] - curr_val_probs[:,col_idx]                
                                    
            val_loss = F.binary_cross_entropy_with_logits(yhat, y.type_as(yhat))
            
            aucs = []
            for ix, (a, b) in enumerate(itertools.combinations(np.sort(np.unique(val_true_y)), 2)):
                a_mask = val_true_y == a
                b_mask = val_true_y == b
                ab_mask = np.logical_or(a_mask,b_mask)
                condit_probs = val_probs[ab_mask,b]/(val_probs[ab_mask,a]+val_probs[ab_mask,b]) 
                condit_probs = np.nan_to_num(condit_probs,nan=.5,posinf=1,neginf=0)
                condit_labels = b_mask[ab_mask].astype(int)
                aucs.append(roc_auc_score(condit_labels,condit_probs))            
            val_AUROC = np.mean(aucs)
                        
        else:
            raise ValueError("Invalid output layer type. Must be 'softmax' or 'sigmoid'")
        
        self.log('val_AUROC', val_AUROC, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_loss', val_loss, prog_bar=False, logger=True, sync_dist=True)

        return val_loss
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),lr=self.learning_rate)
        return optimizer