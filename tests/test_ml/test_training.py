from unittest.mock import MagicMock, patch

from src.ml.training.train_v2 import train_neural_network

@patch("src.ml.training.train_v2.Trainer")
@patch("src.ml.training.train_v2.get_dataloaders")
@patch("src.ml.training.train_v2.torch")
@patch("src.ml.training.train_v2.TransformerAdapter")  # Patch the class to avoid instantiation
def test_train_neural_network(mock_adapter, mock_torch, mock_get_dl, mock_trainer):
    # Setup Dataloaders mock
    mock_get_dl.return_value = (MagicMock(), MagicMock())

    # Setup Trainer mock
    mock_instance = MagicMock()
    mock_trainer.return_value = mock_instance
    mock_instance.output_dir = MagicMock()
    mock_instance.output_dir.__truediv__.return_value = "best_model.pt"

    path = train_neural_network(n_samples=100, epochs=1)
    assert str(path) == "best_model.pt"
    assert mock_instance.fit.called
    assert mock_adapter.called  # Ensure Adapter was initialized
