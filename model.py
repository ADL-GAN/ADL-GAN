import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CONTEXT_SIZE= 64

class Down(nn.Module):
    """Down sampling"""
    def __init__(self, in_channel ,out_channel, kernel, stride, padding):
        super(Down, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=padding)
        self.norm = nn.InstanceNorm1d(out_channel)
        self.relu = nn.LeakyReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x
            
class Up(nn.Module):
    """Up sampling"""
    def __init__(self, in_channel ,out_channel, kernel, stride, padding):
        super(Up, self).__init__()
        self.dconv = nn.ConvTranspose1d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=padding)
        self.norm = nn.InstanceNorm1d(out_channel)
        self.relu = nn.LeakyReLU()
         
    def forward(self, x):
        x = self.dconv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out,context):
        super(ResidualBlock, self).__init__()
        self.context=context
        self.main = nn.Sequential(
            nn.Conv1d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm1d(dim_out, affine=True, track_running_stats=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm1d(dim_out, affine=True, track_running_stats=True))
    def forward(self, x):
        return x[:,:-self.context,:] + self.main(x)

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.query_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1) #

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width)
        out = self.gamma*out + x
        return out

class Generator(nn.Module):
    """Generator of ADL"""
    def __init__(self,context):
        super(Generator, self).__init__()
        self.downsample = nn.Sequential(
            Down(19, 32, (7), (1), (3)),
            Down(32, 64, (4), (2), (1)),
            Down(64, 128, (4), (2), (1)),
            Down(128, 128, (4), (2), (1))
        )

        self.backbone = ResidualBlock(dim_in=128+context, dim_out=128,context=context)
        self.attn = Self_Attn(128, 'relu')

        self.upsample = nn.Sequential(
            Up(128, 128, (4), (2), (1)),
            Up(128, 64, (4), (2), (1)),
            Up(64, 32, (4), (2), (1)),
            Up(32, 19, (9), (1), (2))
        )
        
    def forward(self, x, c):
        #downsample
        x = self.downsample(x)
        #backbone part
        c = c.view(c.size(0), c.size(1), 1)
        c = c.repeat(1, 1, x.size(2))
        for i in range(3):           
            x = torch.cat([x, c], dim=1)
            x = self.backbone(x)
        x = self.attn(x)
        #upsample
        x = self.upsample(x)
        return x

class ADLClassifier(nn.Module):
    """ADLClassifier."""
    def __init__(self,context):
        super(ADLClassifier, self).__init__()
        self.main = nn.Sequential(
            Down(19, 32, (4), (2), (1)),
            Down(32, 64, (4), (2), (1)),
            Down(64, 64, (4), (2), (1)),
            Down(64, 128, (4), (2), (1)),
            Self_Attn(128, 'relu'),
            nn.MaxPool1d((5)),
            nn.Flatten(),
            nn.Linear(384,64),
            nn.LeakyReLU(),
            nn.Linear(64,context),
            nn.LogSoftmax()
        )
        
    def forward(self, x):
       return self.main(x)

class Discriminator(nn.Module):
    """ADLClassifier."""
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            Down(19, 32, (4), (2), (1)),
            Down(32, 64, (4), (2), (1)),
            Down(64, 64, (4), (2), (1)),
            Down(64, 128, (4), (2), (1)),
            Self_Attn(128, 'relu'),
            nn.MaxPool1d((5)),
            nn.Flatten(),
            nn.Linear(384,64),
            nn.LeakyReLU(),
            nn.Linear(64,1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
       return self.main(x)
