import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from services.ml.reinforcement_learning.transformer_policy import DecisionTransformer
from services.ml.trainer_v2 import Trainer


def test_decision_transformer_forward():
    model = DecisionTransformer(state_dim=10, act_dim=2, hidden_size=64)
    batch_size = 4
    seq_len = 20

    states = torch.randn(batch_size, seq_len, 10)
    actions = torch.randn(batch_size, seq_len, 2)
    returns = torch.randn(batch_size, seq_len, 1)
    timesteps = torch.randint(0, 100, (batch_size, seq_len))

    s_preds, a_preds, r_preds = model(states, actions, returns, timesteps)

    assert s_preds.shape == (batch_size, seq_len, 10)
    assert a_preds.shape == (batch_size, seq_len, 2)
    assert r_preds.shape == (batch_size, seq_len, 1)


def test_trainer_v2_fit(tmp_path):
    model = nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        output_dir=str(tmp_path),
        experiment_name="Test_Exp",
    )

    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=10)

    trainer.fit(loader, loader, epochs=2)

    assert (tmp_path / "best_model.pt").exists()
    assert (tmp_path / "metrics.json").exists()
    assert len(trainer.history["train_loss"]) == 2
