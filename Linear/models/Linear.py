import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Module):
    """
    Linear model for time series forecasting
    """
    def __init__(self, configs, device):
        super(Linear, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        
        # Simple linear projection from seq_len to pred_len
        self.linear = nn.Linear(self.seq_len, self.pred_len)
        
    def norm(self, x, dim=0, means=None, stdev=None):
        if means is not None:
            return x * stdev + means
        else:
            means = x.mean(dim, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=dim, keepdim=True, unbiased=False) + 1e-5).detach()
            x /= stdev
            return x, means, stdev
    
    def forward(self, x):
        # x shape: [batch, channels, seq_len]
        B, C, L = x.size()
        
        # Normalize input
        x, means, stdev = self.norm(x, dim=2)
        
        # Apply linear projection: [batch, channels, seq_len] -> [batch, channels, pred_len]
        # Reshape to [batch * channels, seq_len] for linear layer
        x = x.view(B * C, L)
        output = self.linear(x)  # [batch * channels, pred_len]
        
        # Reshape back to [batch, channels, pred_len]
        output = output.view(B, C, self.pred_len)
        
        # Denormalize
        output = self.norm(output, means=means, stdev=stdev)
        
        return output

