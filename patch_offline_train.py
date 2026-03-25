import re

with open("src/ml/reinforcement_learning/offline_train.py", "r") as f:
    content = f.read()

# Replace import pickle
content = re.sub(r'import pickle\s*#.*', 'import json', content)

# Modify convert_pkl_to_parquet to raise RuntimeError and add convert_json_to_parquet
convert_pkl = '''def convert_pkl_to_parquet(pkl_path: str, parquet_path: str) -> None:
    """
    DEPRECATED: Pickle is strictly prohibited.
    """
    raise RuntimeError("Insecure pickle usage is prohibited. Use convert_json_to_parquet instead.")

def convert_json_to_parquet(json_path: str, parquet_path: str) -> None:
    """
     OPTIMIZATION: Convert bulky serialized trajectories to compressed Parquet.
    Enables zero-copy reading and sharding for Ray Data.
    """
    import pandas as pd

    try:
        import json
        with open(json_path, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df.to_parquet(parquet_path, compression="snappy")
        logger.info("trajectories_converted_to_parquet", path=parquet_path)
    except Exception as e:
        logger.error("parquet_conversion_failed", error=str(e))'''

content = re.sub(r'def convert_pkl_to_parquet.*?except Exception as e:\n        logger\.error\("parquet_conversion_failed", error=str\(e\)\)', convert_pkl, content, flags=re.DOTALL)

# Modify fallback in train_offline
train_offline_fallback = '''    else:
        import json
        with open(dataset_path, "r") as f:
            trajectories = cast(list[dict[str, Any]], json.load(f))'''
content = re.sub(r'    else:\n        with open\(dataset_path, "rb"\) as f:\n            trajectories = cast\(list\[dict\[str, Any\]\], pickle\.load\(f\)\)\s*#.*', train_offline_fallback, content)

with open("src/ml/reinforcement_learning/offline_train.py", "w") as f:
    f.write(content)
