with open("src/ml/aiops/anomaly_detector.py", "r") as f:
    c = f.read()

c = c.replace('    # Define stubs so code doesn\'t crash on class definitions\n    class Nn:\n        class Module: pass\n    F = None', '    # Define stubs so code doesn\'t crash on class definitions\nclass Nn:\n    class Module:\n        pass\n\nF = None')

with open("src/ml/aiops/anomaly_detector.py", "w") as f:
    f.write(c)

with open("src/ml/main.py", "r") as f:
    c = f.read()
if "from fastapi import Depends" not in c and "from fastapi import FastAPI, Request, Depends" not in c:
    c = c.replace('from fastapi import FastAPI', 'from fastapi import FastAPI, Depends')

with open("src/ml/main.py", "w") as f:
    f.write(c)

with open("src/ml/aiops/backtest_evaluator.py", "r") as f:
    c = f.read()
if "from typing import cast" not in c:
    c = c.replace('from typing import TypedDict', 'from typing import TypedDict, cast')

with open("src/ml/aiops/backtest_evaluator.py", "w") as f:
    f.write(c)
