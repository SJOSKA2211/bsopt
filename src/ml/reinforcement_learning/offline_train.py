import pickle  # nosec B403
import time

import mlflow
import mlflow.pytorch
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
    Advanced Offline training for Decision Transformer (v2).
    Uses AMP, torch.compile, and detailed MLflow tracking.
    """
    logger.info("offline_training_started_v2", dataset=dataset_path)

    with open(dataset_path, "rb") as f:
        trajectories = pickle.load(f)  # nosec B301

    dataset = TrajectoryDataset(trajectories)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model = DecisionTransformer(state_dim=100, action_dim=10).to(device)

    # 🔥 ACCELERATION: torch.compile (requires PyTorch 2.0+)
    if hasattr(th, "compile") and device.type == "cuda":
        try:
            model = th.compile(model)
            logger.info("model_compiled_successfully")
        except Exception as e:
            logger.warning("torch_compile_failed_falling_back", error=str(e))

    optimizer = th.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = nn.MSELoss()
    scaler = th.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    with mlflow.start_run(run_name="DT_v2_Offline_Training"):
        mlflow.log_params({"epochs": epochs, "batch_size": batch_size, "device": str(device)})

        model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            start_time = time.time()

            for batch in loader:
                # Move to device
                states = batch["states"].to(device)
                actions = batch["actions"].to(device)
                rtg = batch["rtg"].to(device)
                timesteps = batch["timesteps"].to(device)

                optimizer.zero_grad(set_to_none=True)

                # 🔥 AMP: Automatic Mixed Precision
                with th.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    state_preds, action_preds, return_preds = model(
                        states,
                        actions,
                        rtg,
                        timesteps,
                    )
                    
                    # 1. Action Prediction Loss (P0)
                    loss_action = criterion(action_preds, actions)
                    
                    # 2. Auxiliary Losses for Stability (P1)
                    loss_state = criterion(state_preds[:, :-1, :], states[:, 1:, :])
                    loss_rtg = criterion(return_preds[:, :-1, :], rtg[:, 1:, :])
                    
                    loss = loss_action + 0.1 * loss_state + 0.1 * loss_rtg

                scaler.scale(loss).backward()

                # Gradient Clipping for stability
                scaler.unscale_(optimizer)
                grad_norm = th.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()

            duration = time.time() - start_time
            avg_loss = epoch_loss / len(loader)

            # Log advanced metrics to MLflow
            mlflow.log_metric("loss", avg_loss, step=epoch)
            mlflow.log_metric("grad_norm", grad_norm.item(), step=epoch)
            mlflow.log_metric("epoch_duration", duration, step=epoch)
            
            # Periodically log weight distributions
            if epoch % 10 == 0:
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        mlflow.log_metric(f"weight_std_{name.replace('.', '_')}", param.std().item(), step=epoch)

            logger.info("epoch_completed", epoch=epoch, loss=avg_loss, duration=round(duration, 2))

        mlflow.pytorch.log_model(model, "decision_transformer_v2")
        th.save(model.state_dict(), "models/decision_transformer_v2.pt")
        logger.info("offline_training_completed_v2", path="models/decision_transformer_v2.pt")

    return model


if __name__ == "__main__":
    train_offline("data/trajectories.pkl")
