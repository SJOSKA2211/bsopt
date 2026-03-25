import re

with open("src/ml/distributed_training.py", "r") as f:
    content = f.read()

fallback = '''        import json

        with open("data/trajectories.json", "r") as f:
            trajectories = json.load(f)  # nosec B301
        dataset = TrajectoryDataset(trajectories)
        loader = DataLoader(dataset, batch_size=config.get("batch_size", 64), shuffle=True)
        sharded_loader = ray.train.torch.prepare_data_loader(loader)'''
content = re.sub(r'        import pickle\s*#.*?\n\n        with open\("data/trajectories\.pkl", "rb"\) as f:\n            trajectories = pickle\.load\(f\)\s*#.*?\n        dataset = TrajectoryDataset\(trajectories\)\n        loader = DataLoader\(dataset, batch_size=config\.get\("batch_size", 64\), shuffle=True\)\n        sharded_loader = ray\.train\.torch\.prepare_data_loader\(loader\)', fallback, content, flags=re.DOTALL)

with open("src/ml/distributed_training.py", "w") as f:
    f.write(content)
