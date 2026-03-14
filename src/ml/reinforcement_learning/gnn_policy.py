import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.td3.policies import TD3Policy
from torch_geometric.nn import GATConv


class GATFeaturesExtractor(BaseFeaturesExtractor):
    """
    Advanced Graph Attention Network (GAT) Extractor for stable-baselines3.
    Constructs features from option surface topology.
    """

    def __init__(self, observation_space, features_dim: int = 64, heads: int = 4):
        super().__init__(observation_space, features_dim)
        self.input_dim = 100

        self.conv1 = GATConv(self.input_dim, 64, heads=heads, dropout=0.1)
        self.conv2 = GATConv(64 * heads, 64, heads=heads, dropout=0.1)
        self.conv3 = GATConv(64 * heads, features_dim, heads=1, concat=False)

        self.layer_norm = nn.LayerNorm(features_dim)
        self._cached_edge_index = None

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Standard path for training with CACHED edge index."""
        if self._cached_edge_index is None or self._cached_edge_index.device != observations.device:
            self._cached_edge_index = self._get_static_edge_index(observations.device)

        return self.forward_jit(observations, self._cached_edge_index)

    @torch.jit.export
    def forward_jit(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """SILICON PATH: High-performance JIT-friendly forward pass."""
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return self.layer_norm(x)

    def _get_static_edge_index(self, device: torch.device):
        """Build edges between strike/expiry neighbors."""
        # Simple adjacency for 10 nodes (OPT_0 to OPT_9)
        edges = []
        for i in range(9):
            edges.append([i, i + 1])  # Strike adjacency
            edges.append([i + 1, i])
        return torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)


class GATTD3Policy(TD3Policy):
    """TD3 Policy with GAT topological extractor."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=GATFeaturesExtractor,
            features_extractor_kwargs={"features_dim": 64},
        )


GNNFeatureExtractor = GATFeaturesExtractor
SACGNNPolicy = GATTD3Policy
