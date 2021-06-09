import numpy as np
import ipdb
import rdkit
from torch._C import Value
import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
import ast
import os
import glob
from PIL import Image
import pandas as pd
import random
import torch.utils.data as data
import torchvision.transforms.functional as TF
import cv2
try :
    from rdkit import Chem
except :
    os.system("pip install rdkit-pypi")
    from rdkit import Chem
    

class tox_21(Dataset) :
    def __init__(self, file_root, mode="train", ratio=0.3, target="all") :
        self.file_root = file_root
        
        try :      
            self.data = pd.read_csv(self.file_root)
        except :
            raise ValueError("File root is not supported loacation")
                    
        mode = mode.lower()
        
        if self.mode not in ["train", "validation", "test", "t", "v"] :
            raise ValueError("Not supported mode")
        else :
            self.mode = mode 
        
        if type(ratio) != int :
            raise ValueError("The Ratio are must be INT type")
        else : 
            self.ratio = ratio 
        
        self.target = target  
        self.check_target(target)
            
        self.trn_file, self.val_file, self.tst_file = self.split_data()
        
        self.SMILES_CHARS = [' ',
                  '#', '%', '(', ')', '+', '-', '.', '/',
                  '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                  '=', '@',
                  'A', 'B', 'C', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P',
                  'R', 'S', 'T', 'V', 'X', 'Z',
                  '[', '\\', ']',
                  'a', 'b', 'c', 'e', 'g', 'i', 'l', 'n', 'o', 'p', 'r', 's',
                  't', 'u']
        
        self.smi2index = dict( (c,i) for i,c in enumerate( self.SMILES_CHARS ) )
        self.index2smi = dict( (i,c) for i,c in enumerate( self.SMILES_CHARS ) )
    
    def __len__(self) :
        
        if self.mode == "train" or self.mode == "t" :
            return len(self.trn_file) 
        elif self.mode == "validation" or self.mode=="v" :
            return len(self.val_file)
        elif self.mode == "test" :
            return len(self.tst_file) 
        else :
            return len(self.file_root)
        
    def check_target(self, target) :
        targets = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER",
                "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", 
                "SR-MMP", "SR-p53", "all"]
        
        targets = [i.lower() for i in targets] 
        tar = target.lower()

        if tar not in targets :
            raise ValueError("Not Support Target!") 
        
        return 0 
    
    def split_data(self) :
        ratio = self.ratio 
        
        tst_ratio = 1-ratio 
        tst_idx = int(len(self.data) * tst_ratio)
        tst_file = self.data[tst_idx:]
        tst_file = tst_file.reset_index(drop=True)
        
        tmp_file = self.data[:tst_idx]
        tmp_file = tmp_file.reset_index(drop=True)
        
        tmp = len(tmp_file)
        tmp_ratio = 0.8
        tmp_idx = int(tmp * tmp_ratio) 
        
        trn_file = tmp_file[:tmp_idx]
        trn_file = trn_file.reset_index(drop=True)
        
        val_file = tmp_file[tmp_idx:]
        val_file = val_file.reset_index(drop=True)
        
        return trn_file, val_file, tst_file 
    
    def smiles_encoder(self, smiles, maxlen=120 ):
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles( smiles ))
        X = np.zeros( ( maxlen, len( self.SMILES_CHARS ) ) )
        for i, c in enumerate( smiles ):
            X[i, self.smi2index[c] ] = 1
        return X
 
    def smiles_decoder(self, X ):
        smi = ''
        X = X.argmax( axis=-1 )
        for i in X:
            smi += self.index2smi[ i ]
        return smi
    
    def get_x_y(self, idx) :
        
        if self.mode == "train" :
            molecular = self.trn_file['smiles'].loc[idx]
            labels = list(self.trn_file.drop(['smiles'], axis=1).loc[idx].values)
            
        elif self.mode == "validation" :
            molecular = self.val_file['smiles'].loc[idx]
            labels = list(self.val_file.drop(['smiles'], axis=1).loc[idx].values)
            
        else : #self.mode == "test" :
            molecular = self.tst_file['smiles'].loc[idx]
            labels = list(self.tst_file.drop(['smiles'], axis=1).loc[idx].values)
        
        return molecular, labels
    
    def __getitem__(self, idx):
        
        molecular, labels = self.get_x_y(idx)
        
        target_label = self.target.lower()
        
        targets = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER",
                "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", 
                "SR-MMP", "SR-p53"]
        
        targets = [i.lower() for i in targets] 
        
        if target_label != "all" :
            index = int(targets.index(target_label)) 
            labels = labels[index]
            
        molecular = self.smiles_encoder(molecular)
        
        return molecular, labels 

if __name__=='__main__':

    import ipdb; ipdb.set_trace()
