import pytest
import torch
from unittest.mock import MagicMock, patch
from src.ml.training.train_v2 import TransformerAdapter, train_neural_network

def test_transformer_adapter_forward():
    model = TransformerAdapter(input_dim=10, hidden_dim=64, output_dim=1)
    x = torch.randn(32, 10)
    output = model(x)
    assert output.shape == (32, 1)

@patch("src.ml.training.train_v2.Trainer")
def test_train_neural_network(mock_trainer):
    mock_instance = MagicMock()
    mock_trainer.return_value = mock_instance
    mock_instance.output_dir = MagicMock()
    mock_instance.output_dir.__truediv__.return_value = "best_model.pt"
    
    path = train_neural_network(n_samples=100, epochs=1)
    assert str(path) == "best_model.pt"
    assert mock_instance.fit.called
