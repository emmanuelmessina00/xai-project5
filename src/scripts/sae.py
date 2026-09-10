import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 512, hidden_dim: int = 2048):
        super(SparseAutoencoder, self).__init__()
        self.b_dec = nn.Parameter(torch.zeros(input_dim))
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim,bias=False)
    
    def normalize_decoder_weights(self):
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, p=2, dim=0)

    def forward(self, x: torch.Tensor):
        x_centered= x-self.b_dec
        z = F.relu(self.encoder(x_centered))
        x_hat = self.decoder(z) + self.b_dec
        
        return x_hat, z

def sae_loss_function(x: torch.Tensor, x_hat: torch.Tensor, z: torch.Tensor, l1_lambda: float = 2e-3):
    mse_loss = F.mse_loss(x_hat, x)
    l1_loss = z.abs().sum(dim=-1).mean()
    total_loss = mse_loss + l1_lambda * l1_loss
    
    return total_loss, mse_loss, l1_loss

def l2_normalize(embeddings: torch.Tensor) -> torch.Tensor:
    # F.normalize divide ogni vettore per la sua norma L2
    return F.normalize(embeddings, p=2, dim=1)

if __name__=="__main__":
    sae = SparseAutoencoder(input_dim=512, hidden_dim=2048)
    print(sae)