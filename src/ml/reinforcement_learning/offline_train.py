import pickle

import structlog
import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.ml.reinforcement_learning.decision_transformer import DecisionTransformer

logger = structlog.get_logger()


class TrajectoryDataset(Dataset):
    def __init__(self, trajectories):
        self.trajectories = trajectories

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        # traj: {states, actions, rewards, returns_to_go, timesteps}
        return {
            "states": th.FloatTensor(traj["states"]),
            "actions": th.FloatTensor(traj["actions"]),
            "rtg": th.FloatTensor(traj["returns_to_go"]),
            "timesteps": th.LongTensor(traj["timesteps"]),
        }


def train_offline(dataset_path: str, epochs: int = 100, batch_size: int = 64):
    """
    Advanced Offline training for Decision Transformer.
    """
    logger.info("offline_training_started", dataset=dataset_path)

    with open(dataset_path, "rb") as f:
        trajectories = pickle.load(f)

    dataset = TrajectoryDataset(trajectories)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DecisionTransformer(state_dim=100, action_dim=10)
    optimizer = th.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in loader:
            optimizer.zero_grad()

            # Predict actions based on state and return-to-go
            # model(states, actions, returns_to_go, timesteps)
            _, action_preds, _ = model(
                batch["states"],
                th.zeros_like(batch["actions"]),  # Masked actions for prediction
                batch["rtg"],
                batch["timesteps"],
            )

            loss = criterion(action_preds, batch["actions"])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        logger.info("epoch_completed", epoch=epoch, loss=epoch_loss / len(loader))

    th.save(model.state_dict(), "models/decision_transformer_offline.pt")
    logger.info(
        "offline_training_completed", path="models/decision_transformer_offline.pt"
    )
    return model


if __name__ == "__main__":
    train_offline("data/trajectories.pkl")
