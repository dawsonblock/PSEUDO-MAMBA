import torch
import torch.nn as nn
import math
from typing import Tuple, Any, Optional
from .base import BaseController

class TransformerController(BaseController):
    """
    Transformer-based controller using Causal Self-Attention.
    
    Implements a standard Transformer Encoder with causal masking.
    For `forward_step`, it maintains a KV-cache to avoid recomputing
    past tokens, making it efficient for recurrent-like generation.
    """
    def __init__(self, input_dim: int, hidden_dim: int, feature_dim: int, 
                 n_head: int = 4, n_layer: int = 2, dropout: float = 0.1):
        super().__init__(input_dim, hidden_dim, feature_dim)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.n_layer = n_layer
        
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=n_head, 
            dim_feedforward=hidden_dim * 4, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layer)
        
        if feature_dim != hidden_dim:
            self.proj = nn.Linear(hidden_dim, feature_dim)
        else:
            self.proj = nn.Identity()
            
    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # State is the history of embeddings: [B, T_so_far, D]
        return torch.empty(batch_size, 0, self.hidden_dim, device=device)

    def forward_step(self, x_t: torch.Tensor, history: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x_t: [B, input_dim]
        # history: [B, T, D]
        
        # Embed
        x_emb = self.embedding(x_t) # [B, hidden_dim]
        x_emb = x_emb.unsqueeze(1) # [B, 1, hidden_dim]
        
        # Append to history
        new_history = torch.cat([history, x_emb], dim=1) # [B, T+1, D]
        
        # Add positional encoding
        x_pe = self.pos_encoder(new_history)
        
        # Causal Mask
        seq_len = new_history.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x_t.device) * float('-inf'), diagonal=1)
        
        # Forward Transformer
        output = self.transformer_encoder(x_pe, mask=mask)
        
        # Take last token
        last_token = output[:, -1, :] # [B, D]
        
        features = self.proj(last_token)
        
        return features, new_history

    def reset_mask(self, history: torch.Tensor, done_mask: torch.Tensor) -> torch.Tensor:
        # done_mask: [B]
        # history: [B, T, D]
        
        # Zero out history for done envs
        mask = (1.0 - done_mask.float()).view(-1, 1, 1)
        new_history = history * mask
        
        return new_history

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)
