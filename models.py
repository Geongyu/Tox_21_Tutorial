from scipy.special.orthogonal import chebyc
import torch
import torch.nn as nn
import torch.nn.functional as F

# Seqeunce Modeling 

class BESTox(nn.Module) :
    def __init__(self, input_shape, padding, momentum=0.9) :
        '''
        input_shape : input data size 
        padding : padding size
        momentum : BatchNorm Momentum 
        '''
        super(BESTox, self).__init__()
        heights, width = input_shape
        self.padding = padding 
        
        self.conv1 = nn.Conv1d(in_channels=heights, out_channels=512, kernel_size=1)
        self.avg_pool1 = nn.AvgPool1d(2)
        self.batchnorm1 = nn.BatchNorm1d(512)
        
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=1)
        self.avg_pool2 = nn.AvgPool1d(2) 
        self.batchnorm2 = nn.BatchNorm1d(1024) 
        
        self.max_pool = nn.MaxPool1d(2)
        
        self.fc1 = nn.Linear(7*1024, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        
        self.output = nn.Linear(256, 1)
        
        self.relu = nn.ReLU()
        
    def forward(self, x) :
        
        #import ipdb; ipdb.set_trace()
        x = self.conv1(x)
        x = self.relu(self.avg_pool1(x))
        x = self.batchnorm1(x)
        
        x = self.conv2(x) 
        x = self.relu(self.avg_pool2(x))
        x = self.batchnorm2(x) 
        
        x = self.max_pool(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x)) 
        
        output = self.output(x)
        
        return output 

if __name__ == '__main__':
    #from torchsummary import summary
    #net = BESTox(input_shape = (140, 56), padding=1, momentum=0.1).cpu()
    #summary(net.cpu(), (140, 56), device="cpu")
    
    from dataloader import tox_21
    file_root = "/home/deepbio/Desktop/ADMET_code/Data/tox21.csv"
    train_set = tox_21(file_root, mode="train", ratio=0.2, target="NR-AR") 
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=0)
    
    net = BESTox(input_shape = (200, 56), padding=1, momentum=0.1).cpu().double()
    
    for idx, (data, label) in enumerate(train_loader) :
        out = net(data.double())
        import ipdb; ipdb.set_trace()
