import torch

from src.ml.reinforcement_learning.gnn_policy import GNNFeatureExtractor


def test_gnn_feature_extractor_forward_pass():
    # Setup extractor
    input_dim = 16
    output_dim = 64
    extractor = GNNFeatureExtractor(input_dim=input_dim, output_dim=output_dim)

    # Create mock graph data
    num_nodes = 10
    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4, 5], [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]],
        dtype=torch.long,
    )

    # Run forward pass
    features = extractor(x, edge_index)

    # Verify output
    assert features.shape == (num_nodes, output_dim)
    assert not torch.isnan(features).any()


def test_gnn_feature_extractor_with_different_sizes():
    extractor = GNNFeatureExtractor(input_dim=8, output_dim=32)
    x = torch.randn(5, 8)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

    features = extractor(x, edge_index)
    assert features.shape == (5, 32)
