import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, latent_dim=16, drop_prob=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, dropout=drop_prob, num_layers=1)
        self.mean_fc = nn.Linear(hidden_dim, latent_dim)
        self.log_var_fc = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        # x: (B, T, input_dim)
        _, (hidden, cell) = self.lstm(x)
        cell = cell.squeeze(0)  # (L, B, hidden_dim)
        
        mean = self.mean_fc(cell)           # (B, latent_dim)
        log_var = self.log_var_fc(cell)     # (B, latent_dim)
        return mean, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim=16, hidden_dim=32, output_dim=2, seq_length=50, drop_prob=0.3):
        super().__init__()
        self.seq_length = seq_length
        self.lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True, dropout=drop_prob, num_layers=1)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, z, seq_length=None):
        if seq_length is None:
            seq_length = self.seq_length
                
        # z: (B, latent_dim)
        # Repeat z for each time step: (B, T, latent_dim)
        z_expanded = z.unsqueeze(1).repeat(1, seq_length, 1)
        
        # h: (B, T, hidden_dim)
        h, _ = self.lstm(z_expanded)

        # out: (B, T, output_dim)
        out = self.fc_out(h)
        return out

class SeqVAE(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, latent_dim=16, seq_length=50):
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_length = seq_length
        
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, drop_prob=0)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, seq_length, drop_prob=0)
        
    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def encode(self, x):
        mean, log_var = self.encoder(x)
        return self.reparameterize(mean, log_var)
    
    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        out = self.decoder(z, seq_length=x.shape[1])
        return out, mean, log_var
    
    def sample(self, num_samples, z=None, seq_length=None, device='cpu'):
        if z == None:
            z = torch.randn(num_samples, self.latent_dim)
        z = z.to(device)
        
        with torch.no_grad():
            generated = self.decoder(z, seq_length=seq_length)
        return generated