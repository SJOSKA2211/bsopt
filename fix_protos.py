import os
import glob

def fix_imports(filepath):
    with open(filepath, "r") as f:
        content = f.read()

    # Replace bad imports with their correct versions
    content = content.replace("from google.protobuf from . from . from . from . from . import empty_pb2", "from google.protobuf import empty_pb2")
    content = content.replace("from google.protobuf from . from . from . from . from . import timestamp_pb2", "from google.protobuf import timestamp_pb2")
    content = content.replace("from . from . from . from . from . from . import auth_pb2", "from . import auth_pb2")
    content = content.replace("from . from . from . from . from . from . import pricing_pb2", "from . import pricing_pb2")
    content = content.replace("from google.protobuf from . from . from . from . from . import struct_pb2", "from google.protobuf import struct_pb2")
    content = content.replace("from . from . from . from . from . from . import inference_pb2", "from . import inference_pb2")
    content = content.replace("from . from . from . from . from . from . import market_data_pb2", "from . import market_data_pb2")
    content = content.replace("from . from . from . from . from . from . import common_pb2", "from . import common_pb2")

    with open(filepath, "w") as f:
        f.write(content)

if __name__ == "__main__":
    proto_files = glob.glob("src/shared/protos/*.py")
    for f in proto_files:
        fix_imports(f)
