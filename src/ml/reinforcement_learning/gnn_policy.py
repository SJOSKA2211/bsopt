import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class GNNFeatureExtractor(nn.Module):
    """
    Advanced Graph Attention Network (GAT) for option surface topology.
    Models complex strike/expiry interactions with multi-head attention.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 32, heads: int = 4):
        super().__init__()
        # Use GAT layers for edge-aware attention
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.1)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=0.1)
        self.conv3 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False)
        
        self.fc = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch vector [num_nodes]
        """
        # 1. Attentional Message Passing
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        
        x = self.conv3(x, edge_index)
        
        # 2. Global Topological Pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
            
        return self.fc(x)

class SACGNNPolicy(nn.Module):
    """
    Soft Actor-Critic policy using GAT features.
    """
    def __init__(self, state_dim: int, action_dim: int, gnn_output_dim: int = 32):
        super().__init__()
        self.gnn = GNNFeatureExtractor(input_dim=state_dim, output_dim=gnn_output_dim)
        
        self.actor = nn.Sequential(
            nn.Linear(gnn_output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim * 2) 
        )
        
    def forward(self, x, edge_index):
        features = self.gnn(x, edge_index)
        return self.actor(features)
