import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    SOTA: Attention-based feature extractor for temporal market state.
    Uses multi-head attention and attention pooling to maintain temporal locality.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self.window_size, self.input_dim = observation_space.shape
        
        self.input_projection = nn.Linear(self.input_dim, 128)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 🚀 OPTIMIZATION: Attention-based pooling instead of flattening
        self.attention_pooling = nn.Sequential(
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        
        self.output_projection = nn.Linear(128, features_dim)
        self.layer_norm = nn.LayerNorm(features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # observations: [batch, window, dim]
        x = self.input_projection(observations)
        x = self.transformer(x) # [batch, window, 128]
        
        # Calculate attention weights over time steps
        weights = self.attention_pooling(x) # [batch, window, 1]
        pooled = th.sum(x * weights, dim=1) # [batch, 128]
        
        return self.layer_norm(self.output_projection(pooled))

from stable_baselines3.td3.policies import TD3Policy

class TransformerTD3Policy(TD3Policy):
    """
    Integrated Transformer policy for TD3.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, 
            **kwargs, 
            features_extractor_class=TransformerFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256)
        )
