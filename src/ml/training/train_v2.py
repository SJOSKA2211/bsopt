import logging
import os
from datetime import datetime

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.ml.architectures.neural_network import OptionPricingNN
from src.ml.training.train import load_or_collect_data

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Hyperparameters from user request
EPOCHS = 10
LR = 1e-4
BATCH_SIZE = 32
OPTIMIZER_TYPE = "AdamW"
LOSS_TYPE = "CrossEntropy"
SAVE_FREQ = 2
EVAL_FREQ = 1


def prepare_classification_data(X, feature_names):
    """
    Convert regression data to classification (ITM status).
    ITM = 1 if (Call and Spot > Strike) or (Put and Spot < Strike) else 0
    """
    spot_idx = feature_names.index("underlying_price")
    strike_idx = feature_names.index("strike")
    is_call_idx = feature_names.index("is_call")

    spots = X[:, spot_idx]
    strikes = X[:, strike_idx]
    is_calls = X[:, is_call_idx]

    # ITM logic
    itm = ((is_calls == 1) & (spots > strikes)) | ((is_calls == 0) & (spots < strikes))
    return itm.astype(np.int64)


def train_v2():
    # 1. Load Data
    logger.info("Loading data...")
    X, y_reg, feature_names, _ = load_or_collect_data(
        use_real_data=False, n_samples=50000
    )

    # Prepare targets for CrossEntropy
    y = prepare_classification_data(X, feature_names)

    # 2. Preprocess
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Convert to Tensors
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 3. Model, Optimizer, Loss
    model = OptionPricingNN(input_dim=len(feature_names), num_classes=2)

    if OPTIMIZER_TYPE == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=LR)
    else:
        optimizer = optim.Adam(model.parameters(), lr=LR)

    criterion = nn.CrossEntropyLoss()

    # 4. MLflow Tracking
    tracking_uri = "file://" + os.path.abspath("mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    os.makedirs("mlruns", exist_ok=True)
    mlflow.set_experiment("Option_ITM_Classification")

    with mlflow.start_run(
        run_name=f"training-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    ):
        mlflow.log_params(
            {
                "epochs": EPOCHS,
                "lr": LR,
                "batch_size": BATCH_SIZE,
                "optimizer": OPTIMIZER_TYPE,
                "loss": LOSS_TYPE,
                "save_freq": SAVE_FREQ,
                "eval_freq": EVAL_FREQ,
            }
        )

        logger.info("Starting training loop...")
        for epoch in range(1, EPOCHS + 1):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

            train_acc = correct / total
            train_loss = running_loss / len(train_loader)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc, step=epoch)

            logger.info(
                f"Epoch {epoch}/{EPOCHS} - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}"
            )

            # Evaluation
            if epoch % EVAL_FREQ == 0:
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for batch_X, batch_y in test_loader:
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += batch_y.size(0)
                        val_correct += (predicted == batch_y).sum().item()

                val_acc = val_correct / val_total
                val_loss_avg = val_loss / len(test_loader)
                mlflow.log_metric("val_loss", val_loss_avg, step=epoch)
                mlflow.log_metric("val_acc", val_acc, step=epoch)
                logger.info(
                    f"  Validation - Loss: {val_loss_avg:.4f}, Acc: {val_acc:.4f}"
                )

            # Saving
            if epoch % SAVE_FREQ == 0:
                checkpoint_path = f"models/model_epoch_{epoch}.pth"
                os.makedirs("models", exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
                mlflow.log_artifact(checkpoint_path)
                logger.info(f"  Checkpoint saved: {checkpoint_path}")

        # Final Model Save
        final_path = "models/itm_classifier_final.pth"
        torch.save(model.state_dict(), final_path)
        mlflow.pytorch.log_model(model, "itm_classifier")
        logger.info(f"Final model saved: {final_path}")


if __name__ == "__main__":
    train_v2()
