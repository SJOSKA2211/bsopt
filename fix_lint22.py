with open("src/ml/aiops/anomaly_detector.py", "r") as f:
    c = f.read()

c = c.replace('F = None\n    TensorDataset = None\n    DataLoader = None\n', 'F = None\nTensorDataset = None\nDataLoader = None\n')
with open("src/ml/aiops/anomaly_detector.py", "w") as f:
    f.write(c)
