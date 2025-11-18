import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NLinear(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, configs, device):
        super(NLinear, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = configs.enc_in
        self.individual = False
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Channel, Input length]
        B, C, L = x.size()
        
        # Transpose to [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)  # [B, L, C]
        
        # Normalization: subtract the last value
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last
        
        if self.individual:
            output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            # Apply linear layer: [B, L, C] -> permute to [B, C, L] -> linear -> [B, C, pred_len] -> permute to [B, pred_len, C]
            x = self.Linear(x.permute(0,2,1)).permute(0,2,1)  # [B, pred_len, C]
        
        # Denormalization: add back the last value
        x = x + seq_last
        
        # Transpose back to [Batch, Channel, Output length]
        x = x.permute(0, 2, 1)  # [B, C, pred_len]
        
        return x

