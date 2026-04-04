import time
from typing import Any, cast

import mlflow
import mlflow.pytorch
import structlog
import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.ml.reinforcement_learning.decision_transformer import (
    DecisionTransformer,
    QNetwork,
    ValueNetwork,
)

logger = structlog.get_logger()


class TrajectoryDataset(Dataset[dict[str, th.Tensor]]):  # type: ignore
    def __init__(self, trajectories: list[dict[str, Any]]) -> None:
        self.trajectories = trajectories

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> dict[str, th.Tensor]:
        traj = self.trajectories[idx]
        return {
            "states": th.tensor(traj.get("states", []), dtype=th.float32),
            "actions": th.tensor(traj.get("actions", []), dtype=th.float32),
            "rtg": th.tensor(traj.get("rtg", []), dtype=th.float32),
            "timesteps": th.tensor(traj.get("timesteps", []), dtype=th.long),
        }


def expectile_loss(diff: th.Tensor, tau: float = 0.7) -> th.Tensor:
    weight = th.where(diff > 0, tau, 1 - tau)
    return weight * (diff**2)


def convert_pkl_to_parquet(pkl_path: str, parquet_path: str) -> None:
    raise RuntimeError("Insecure pickle loading is deprecated. Use convert_json_to_parquet instead.")

def convert_json_to_parquet(json_path: str, parquet_path: str) -> None:
    """
     OPTIMIZATION: Convert bulky serialized trajectories to compressed Parquet.
    Enables zero-copy reading and sharding for Ray Data.
    """
    import json
    import pandas as pd

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df.to_parquet(parquet_path, compression="snappy")
        logger.info("trajectories_converted_to_parquet", path=parquet_path)
    except Exception as e:
        logger.error("parquet_conversion_failed", error=str(e))


def _log_gradient_flow(model: nn.Module, step: int) -> None:
    """
    High-Performance: Monitor gradient flow across deep transformer layers.
    Helps detect vanishing/exploding gradients in real-time.
    """
    avg_grads = []
    max_grads = []
    layers = []
    for n, p in model.named_parameters():
        if p.requires_grad and ("bias" not in n) and p.grad is not None:
            layers.append(n)
            avg_grads.append(float(p.grad.abs().mean().item()))
            max_grads.append(float(p.grad.abs().max().item()))

    if avg_grads:
        mlflow.log_metric("grad_avg_mean", sum(avg_grads) / len(avg_grads), step=step)
        mlflow.log_metric("grad_max_mean", sum(max_grads) / len(max_grads), step=step)


def train_offline(
    dataset_path: str,
    epochs: int = 100,
    batch_size: int = 64,
    iql_beta: float = 3.0,
    iql_tau: float = 0.7,
) -> None:
    """
    Advanced Offline training for Decision Transformer (v2) with IQL integration.
    OPTIMIZED: AMP, torch.compile, Cosine Annealing, Gradient Flow Monitoring.
    """
    logger.info("offline_training_started_v2_iql", dataset=dataset_path)

    # 1. ⚡ DATA LOADING OPTIMIZATION
    if dataset_path.endswith(".parquet"):
        import pandas as pd

        df = pd.read_parquet(dataset_path)
        trajectories = cast(list[dict[str, Any]], df.to_dict("records"))
    elif dataset_path.endswith(".json"):
        import json
        with open(dataset_path, "r") as f:
            trajectories = cast(list[dict[str, Any]], json.load(f))
    else:
        raise RuntimeError("Insecure pickle loading is deprecated. Please migrate legacy .pkl files to .json or .parquet.")

    dataset = TrajectoryDataset(trajectories)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    # DT-v2 Model
    model = cast(nn.Module, DecisionTransformer(state_dim=128, action_dim=10).to(device))

    # IQL Components
    q_net = cast(nn.Module, QNetwork(state_dim=128, action_dim=10).to(device))
    v_net = cast(nn.Module, ValueNetwork(state_dim=128).to(device))
    target_q_net = cast(nn.Module, QNetwork(state_dim=128, action_dim=10).to(device))
    target_q_net.load_state_dict(q_net.state_dict())

    #  ACCELERATION: torch.compile
    if hasattr(th, "compile") and device.type == "cuda":
        try:
            model = th.compile(model)
            q_net = th.compile(q_net)
            v_net = th.compile(v_net)
            logger.info("models_compiled_successfully")
        except Exception as e:
            logger.warning("torch_compile_failed", error=str(e))

    optimizer = th.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    th.optim.Adam(q_net.parameters(), lr=3e-4)
    v_optimizer = th.optim.Adam(v_net.parameters(), lr=3e-4)

    #  SCHEDULING
    scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    criterion = nn.MSELoss()
    scaler = th.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    with mlflow.start_run(run_name="DT_v2_IQL_High_Performance"):
        mlflow.log_params(
            {
                "epochs": epochs,
                "batch_size": batch_size,
                "iql_beta": iql_beta,
                "iql_tau": iql_tau,
                "model": "DT-v2",
                "precision": "AMP",
            }
        )

        global_step = 0
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            start_time = time.time()

            for batch in loader:
                states = batch["states"].to(device)
                actions = batch["actions"].to(device)
                rtg = batch["rtg"].to(device)
                timesteps = batch["timesteps"].to(device)

                # 1. Update Value Network (Expectile Regression)
                with th.no_grad():
                    q1, q2 = cast(Any, target_q_net)(states, actions)
                    target_q = th.min(q1, q2)

                v = cast(Any, v_net)(states)
                v_loss = expectile_loss(target_q - v, tau=iql_tau).mean()

                v_optimizer.zero_grad(set_to_none=True)
                v_loss.backward()
                v_optimizer.step()

                # 2. Update Decision Transformer (Policy) with AWR Weighting
                with th.no_grad():
                    v_val = cast(Any, v_net)(states)
                    q1, q2 = cast(Any, q_net)(states, actions)
                    advantage = th.min(q1, q2) - v_val
                    # Advantage Normalization
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                    exp_adv = th.exp(iql_beta * advantage).clamp(max=100.0)

                optimizer.zero_grad(set_to_none=True)
                with th.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    state_preds, action_preds, return_preds = cast(Any, model)(
                        states,
                        actions,
                        rtg,
                        timesteps,
                    )

                    # Weighted imitation loss
                    loss_action = (exp_adv * (action_preds - actions) ** 2).mean()

                    # Auxiliary losses: predict next state from current seq
                    loss_state = criterion(state_preds[:, :-1, :], states[:, 1:, :])
                    loss = loss_action + 0.1 * loss_state

                scaler.scale(loss).backward()

                # 📊 Gradient Flow Monitoring (Periodic)
                if global_step % 100 == 0:
                    _log_gradient_flow(model, global_step)

                scaler.step(optimizer)
                scaler.update()

                epoch_loss += float(loss.item())
                global_step += 1

            # Step scheduler
            scheduler.step()

            # Target Q soft update
            with th.no_grad():
                for p, p_t in zip(q_net.parameters(), target_q_net.parameters()):
                    p_t.data.copy_(0.005 * p.data + (1 - 0.005) * p_t.data)

            duration = time.time() - start_time
            avg_loss = epoch_loss / len(loader)
            mlflow.log_metric("loss", avg_loss, step=epoch)
            mlflow.log_metric("lr", scheduler.get_last_lr()[0], step=epoch)
            logger.info("epoch_completed", epoch=epoch, loss=avg_loss, duration=round(duration, 2))

        # Log weight distributions at the end
        for name, param in model.named_parameters():
            if param.requires_grad:
                mlflow.log_param(
                    f"weight_norm_{name.replace('.', '_')}", float(param.norm().item())
                )

        mlflow.pytorch.log_model(model, "decision_transformer_v2_god_mode")
        th.save(model.state_dict(), "models/dt_v2_final.pt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Offline DT Training")
    parser.add_argument("--dataset", type=str, default="data/trajectories.parquet")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--study_name", type=str, default="offline_dt_v1")
    parser.add_argument("--tracking_uri", type=str, default=None)

    args = parser.parse_args()

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    train_offline(
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        iql_beta=args.beta,
        iql_tau=args.tau,
    )
