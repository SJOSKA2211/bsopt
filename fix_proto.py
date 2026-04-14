import re

file_path = "src/protos/auth_pb2_grpc.py"
try:
    with open(file_path, "r") as f:
        content = f.read()

    # Mypy fails because the generated auth_pb2_grpc.py code might have `def ValidateToken(` without `self` or some invalid Python syntax.
    # Actually wait, let's see what's wrong with it.
    pass
except FileNotFoundError:
    pass

file_path = "src/protos/inference_pb2_grpc.py"
try:
    with open(file_path, "r") as f:
        content = f.read()
    pass
except FileNotFoundError:
    pass
