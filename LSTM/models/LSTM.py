import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    """
    LSTM model for time series forecasting
    """
    def __init__(self, configs, device):
        super(LSTM, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        
        # LSTM parameters
        self.hidden_size = getattr(configs, 'hidden_size', 128)
        self.num_layers = getattr(configs, 'num_layers', 2)
        self.dropout = getattr(configs, 'dropout', 0.2)
        self.bidirectional = bool(getattr(configs, 'bidirectional', 0))
        
        # Input projection
        self.input_proj = nn.Linear(self.enc_in, self.hidden_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional,
            batch_first=True
        )
        
        # Output projection
        lstm_output_size = self.hidden_size * (2 if self.bidirectional else 1)
        # Use a decoder to generate predictions
        self.decoder = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(lstm_output_size, self.c_out * self.pred_len)
        )
        
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
        
        # Transpose to [batch, seq_len, channels]
        x = x.permute(0, 2, 1)  # [B, L, C]
        
        # Project input
        x = self.input_proj(x)  # [B, L, hidden_size]
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)  # [B, L, hidden_size * (2 if bidirectional else 1)]
        
        # Use the last hidden state for prediction
        # h_n shape: [num_layers * (2 if bidirectional else 1), B, hidden_size]
        # Take the last layer's hidden state
        last_hidden = h_n[-1]  # [B, hidden_size * (2 if bidirectional else 1)]
        
        # Decode to predictions
        output = self.decoder(last_hidden)  # [B, c_out * pred_len]
        
        # Reshape to [B, c_out, pred_len]
        output = output.view(B, self.c_out, self.pred_len)
        
        # Denormalize
        output = self.norm(output, means=means, stdev=stdev)
        
        return output

