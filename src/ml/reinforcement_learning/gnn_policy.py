import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.td3.policies import TD3Policy

class GATFeaturesExtractor(BaseFeaturesExtractor):
    """
    Advanced Graph Attention Network (GAT) Extractor for stable-baselines3.
    Constructs features from option surface topology.
    """
    def __init__(self, observation_space, features_dim: int = 64, heads: int = 4):
        # features_dim is the output dimension after pooling
        super().__init__(observation_space, features_dim)
        
        # We assume node features are fixed-size (e.g. price, strike, greeks)
        # state_dim is inferred from observation_space if possible, or hardcoded for the extractor
        self.input_dim = 100 # Example dimension for option nodes
        
        self.conv1 = GATConv(self.input_dim, 64, heads=heads, dropout=0.1)
        self.conv2 = GATConv(64 * heads, 64, heads=heads, dropout=0.1)
        self.conv3 = GATConv(64 * heads, features_dim, heads=1, concat=False)
        
        self.layer_norm = nn.LayerNorm(features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Hot path: Converts flat observations into graph and passes through GAT.
        Note: In a pure graph env, edge_index would be part of the observation.
        For now, we infer adjacencies (strike/expiry neighbors).
        """
        # 🚀 TOPOLOGICAL INFERENCE: Construct graph from flat state
        # num_options = observations.shape[1] // self.input_dim
        # x = observations.view(-1, num_options, self.input_dim)
        
        # Mocking graph construction for SB3 integration compatibility
        # In a real GraphEnv, we use a custom SB3 policy that takes dict observations
        x = observations # Placeholder
        edge_index = self._get_static_edge_index(x.device) 
        
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        
        return self.layer_norm(x)

    def _get_static_edge_index(self, device):
        """Build edges between strike/expiry neighbors."""
        # Simple adjacency for 10 nodes (OPT_0 to OPT_9)
        edges = []
        for i in range(9):
            edges.append([i, i+1]) # Strike adjacency
            edges.append([i+1, i])
        return torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)

class GATTD3Policy(TD3Policy):
    """TD3 Policy with GAT topological extractor."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, 
                         features_extractor_class=GATFeaturesExtractor,
                         features_extractor_kwargs={"features_dim": 64})
