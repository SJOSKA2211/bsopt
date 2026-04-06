with open("src/ml/aiops/anomaly_detector.py", "r") as f:
    c = f.read()

c = c.replace('    class Nn:\n        class Module:\n            pass\n\n    F = None', '    class Nn:\n        class Module: pass\n    F = None')

with open("src/ml/aiops/anomaly_detector.py", "w") as f:
    f.write(c)
