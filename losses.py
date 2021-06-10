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
        self.cosine = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, predicition, label):
        
        positive_sim = self.cosine()
        
        return loss