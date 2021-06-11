import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import MSELoss, SmoothL1Loss, L1Loss
import numpy as np

class molecular_wise_contrastive_loss(torch.nn.Module):
    '''
    The Molecular Wise Contrastive Loss Using Tox 21 Datasets 
    Based on SimCLR (A Simple Framework for Contrastive Learning of Visual Representation, Chen et al., 2020)
    Key idea : In chemical space, each molecular similarity calculator
    '''
    def __init__(self):
        super(molecular_wise_contrastive_loss, self).__init__()
        self.cosine = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, predicition, label):
        
        pos_pred = predicition[(label == 1).nonzero(as_tuple=True)[0]]
        neg_pred = predicition[(label == 0).nonzero(as_tuple=True)[0]]
        
        pos_sim = self.cosine(pos_pred.unsqueeze(1), pos_pred.unsqueeze(0))
        exp_pos_sim = torch.exp(pos_sim) 
        batch, row, col = exp_pos_sim.size()
        
        net_sim = self.cosine(neg_pred.unsqueeze(1), neg_pred.unsqueeze(0))
        
        #labels = (label.unsqueeze(0) == label.unsqueeze(1)).float()
        
        return loss