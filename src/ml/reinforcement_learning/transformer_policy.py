import math

import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = th.zeros(max_len, d_model)
        position = th.arange(0, max_len, dtype=th.float).unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: th.Tensor) -> th.Tensor:
        return x + self.pe[:, :x.size(1)]

class TransformerSingularityExtractor(BaseFeaturesExtractor):
    """
    Advanced Transformer feature extractor for multi-modal market data.
    Implements multi-head self-attention with learned positional encodings and layer-norm bottlenecks.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512, d_model: int = 256, nhead: int = 8, num_layers: int = 4):
        super().__init__(observation_space, features_dim)
        
        # Determine shape [window_size, input_dim]
        if len(observation_space.shape) == 2:
            self.window_size, self.input_dim = observation_space.shape
        else:
            # Flattened or unexpected shape
            self.window_size = 1
            self.input_dim = observation_space.shape[0]

        self.input_projection = nn.Linear(self.input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max(500, self.window_size))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=0.1, 
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 🚀 SOTA: Learnable query-based pooling (attention pooling)
        self.query = nn.Parameter(th.randn(1, 1, d_model))
        self.attn_pool = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, features_dim),
            nn.LayerNorm(features_dim),
            nn.GELU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # observations: [batch, window, dim] or [batch, dim]
        if observations.dim() == 2:
            observations = observations.unsqueeze(1)
            
        x = self.input_projection(observations)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        # Query-based attention pooling: 
        # The 'query' attends to the transformer sequence to produce a fixed-size representation.
        # [batch, 1, d_model]
        query = self.query.expand(x.size(0), -1, -1)
        pooled, _ = self.attn_pool(query, x, x)
        
        return self.output_projection(pooled.squeeze(1))

from stable_baselines3.td3.policies import TD3Policy


class TransformerTD3Policy(TD3Policy):
    """
    Policy for TD3 using the Transformer Singularity Extractor.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, 
            **kwargs, 
            features_extractor_class=TransformerSingularityExtractor,
            features_extractor_kwargs=dict(features_dim=512, d_model=256, nhead=8, num_layers=4)
        )