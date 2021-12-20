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

# DeepCENTERTBI
class deepCENTERTBI(pl.LightningModule):
    def __init__(self,n_tokens,layers,neurons,dropout,output_activation,learning_rate,class_weights,targets):
        """
        Args:
            n_tokens (int): Size of vocabulary
            layers (int): number of hidden layers in feed forward neural network
            neurons (list of length layers): the number of neurons in each layer
            dropout (flaot): the proportion of each dense layer dropped out during training
            output_activation (string): 'softmax' for DeepMN or 'sigmoid' for DeepOR
            learning_rate (float): Learning rate for ADAM optimizer
            class_weights (boolean): identifies whether loss should be weighted against class frequency
            targets (NumPy array): if class_weights == True, provides the class labels of the training set
        """
        super(deepCENTERTBI, self).__init__()
        
        self.save_hyperparameters()
        self.n_tokens = n_tokens
        self.layers = layers
        self.neurons = neurons
        self.dropout = dropout
        self.output_activation = output_activation
        self.learning_rate = learning_rate
        self.class_weights = class_weights
        self.targets = targets
        
        # Define embedding layers
        self.embedX = nn.Embedding(self.n_tokens, self.neurons[0])
        self.embedW = nn.Embedding(self.n_tokens, 1)
        self.embed_Dropout = nn.Dropout(p = self.dropout)
        
        # Define additional hidden layers if self.layers > 1
        
        if self.layers > 1:
            
            self.hidden_layers = nn.ModuleList()
            
            for i in range(1,self.layers):
                self.hidden_layers.append(nn.Linear(self.neurons[i-1],self.neurons[i]))
                self.hidden_layers.append(nn.Dropout(self.dropout))
                
        if self.output_activation == 'softmax': 
            self.hidden2gose = nn.Linear(self.neurons[-1], 7)
        elif self.output_activation == 'sigmoid': 
            self.hidden2gose = nn.Linear(self.neurons[-1],6)
        else:
            raise ValueError("Invalid output activation type. Must be 'softmax' or 'sigmoid'")
            
    def forward(self, idx_list, pt_offsets):

        idx_list = idx_list.to(torch.int64)
        pt_offsets = pt_offsets.to(torch.int64)
        
        # Embed token indices into vectors
        embeddedX = self.embedX(idx_list)
        
        # Constrain aggregation weights to be positive with exponentiation
        w = torch.exp(self.embedW(idx_list))
        
        # Iterate through individual bins and calculate weighted averages per bin
        embed_output = []
        for curr_off_idx in torch.arange(0, len(pt_offsets), dtype=torch.long):
            
            if curr_off_idx == (torch.LongTensor([len(pt_offsets) - 1])[0]):
                curr_pt_idx = torch.arange(pt_offsets[curr_off_idx], embeddedX.shape[0], dtype=torch.long)
            else:        
                curr_pt_idx = torch.arange(pt_offsets[curr_off_idx], pt_offsets[curr_off_idx+1], dtype=torch.long)
                
            embeddedX_avg = (embeddedX[curr_pt_idx,:] * w[curr_pt_idx]).sum(dim=0, keepdim=True) / (len(curr_pt_idx) + 1e-6)
            embed_output += [embeddedX_avg]
            
        x = torch.cat(embed_output, dim=0)
        x = self.embed_Dropout(F.relu(x))
        
        if self.layers > 1:                
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
        gupis, idx_list, y_list, pt_offsets = batch
        
        # Collect current model state outputs for the batch
        yhat = self(idx_list, pt_offsets)
        
        # Calculate loss based on the output activation type
        if self.output_activation == 'softmax': 
            
            if self.class_weights:
                bal_weights = torch.from_numpy(compute_class_weight(class_weight='balanced',
                                                                    classes=np.sort(np.unique(self.targets)),
                                                                    y=self.targets)).type_as(yhat)
                loss = F.cross_entropy(yhat, y_list, weight = bal_weights)
            else:
                loss = F.cross_entropy(yhat, y_list)
                
        elif self.output_activation == 'sigmoid': 
            
            if self.class_weights:
                bal_weights = torch.from_numpy((self.targets.shape[0]
                                                - np.sum(self.targets, axis=0))
                                               / np.sum(self.targets,
                                                        axis=0)).type_as(yhat)
                
                loss = F.binary_cross_entropy_with_logits(yhat, y_list.type_as(yhat), pos_weight = bal_weights)
            else:
                loss = F.binary_cross_entropy_with_logits(yhat, y_list.type_as(yhat))
                
        else:
            raise ValueError("Invalid output activation type. Must be 'softmax' or 'sigmoid'")
        
        return {"loss": loss, "yhat": yhat, "true_y":y_list}
                
    def training_epoch_end(self, training_step_outputs):
        
        comp_loss = torch.tensor([output["loss"].detach() for output in training_step_outputs]).cpu().numpy().mean()
        comp_yhats = torch.vstack([output["yhat"].detach() for output in training_step_outputs])
        comp_true_y = torch.cat([output["true_y"].detach() for output in training_step_outputs]).cpu().numpy()
        
        if self.output_activation == 'softmax': 
            curr_train_probs = F.softmax(comp_yhats).cpu().numpy()
            train_AUROC = roc_auc_score(comp_true_y, curr_train_probs, multi_class='ovo')
        elif self.output_activation == 'sigmoid': 
            curr_train_probs = F.sigmoid(comp_yhats).cpu().numpy()
            comp_true_y = comp_true_y.sum(1).astype(int)
            
            train_probs = np.empty([curr_train_probs.shape[0], curr_train_probs.shape[1]+1])
            train_probs[:,0] = 1 - curr_train_probs[:,0]
            train_probs[:,-1] = curr_train_probs[:,-1]

            for col_idx in range(1,(curr_train_probs.shape[1])):
                train_probs[:,col_idx] = curr_train_probs[:,col_idx-1] - curr_train_probs[:,col_idx]                
            
            train_AUROC = roc_auc_score(comp_true_y, train_probs, multi_class='ovo')
            
        self.log('train_AUROC', train_AUROC, prog_bar=True, logger=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log('train_loss', comp_loss, prog_bar=False, logger=True, sync_dist=True, on_step=False, on_epoch=True)
    
    def validation_step(self, batch, batch_idx):
        
        # Get information from current batch
        gupis, idx_list, y_list, pt_offsets = batch
        
        # Collect current model state outputs for the batch
        yhat = self(idx_list, pt_offsets)
        
        val_true_y = y_list.cpu().numpy()
        
        # Calculate loss based on the output activation type
        if self.output_activation == 'softmax': 
                                    
            curr_val_probs = F.softmax(yhat).cpu().numpy()
            
            val_loss = F.cross_entropy(yhat, y_list)
            
            val_AUROC = roc_auc_score(val_true_y, curr_val_probs, multi_class='ovo')
            
        elif self.output_activation == 'sigmoid': 
            
            curr_val_probs = F.sigmoid(yhat).cpu().numpy()
            val_true_y = val_true_y.sum(1).astype(int)
            
            val_probs = np.empty([curr_val_probs.shape[0], curr_val_probs.shape[1]+1])
            val_probs[:,0] = 1 - curr_val_probs[:,0]
            val_probs[:,-1] = curr_val_probs[:,-1]

            for col_idx in range(1,(curr_val_probs.shape[1])):
                val_probs[:,col_idx] = curr_val_probs[:,col_idx-1] - curr_val_probs[:,col_idx]                
                                    
            val_loss = F.binary_cross_entropy_with_logits(yhat, y_list.type_as(yhat))
                        
            val_AUROC = roc_auc_score(val_true_y, val_probs, multi_class='ovo')
            
        else:
            raise ValueError("Invalid output layer type. Must be 'softmax' or 'sigmoid'")
        
        self.log('val_AUROC', val_AUROC, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_loss', val_loss, prog_bar=False, logger=True, sync_dist=True)

        return val_loss
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),lr=self.learning_rate)
        return optimizer
    
# DeepCENTERTBI modification for SHAP calculation
class shapCENTERTBI(nn.Module):
    def __init__(self,vocab_embed_matrix,hidden2gose,prob=False,thresh=False):
        super(shapCENTERTBI, self).__init__()
        self.vocab_embed_matrix = vocab_embed_matrix
        self.hidden2gose = hidden2gose
        self.prob = prob
        self.thresh = thresh
        
    def forward(self, x):
        embed_x = F.relu(torch.div(torch.matmul(x,self.vocab_embed_matrix),x.sum(1)[:,None]))
        output = self.hidden2gose(embed_x)
        
        if self.thresh:
            
            prob_matrix = F.softmax(output)
            thresh_prob_matrix = (1 - prob_matrix.cumsum(1))[:,:-1]

            if self.prob:
                return thresh_prob_matrix
            else:
                return torch.special.logit(thresh_prob_matrix)
        else:
            if self.prob:
                return F.softmax(output)
            else:
                return output