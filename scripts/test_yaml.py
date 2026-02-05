import yaml

with open('.github/workflows/mlops-pipeline.yml') as f:
    try:
        data = yaml.safe_load(f)
        print("YAML Loaded successfully")
        print(f"Keys: {list(data.keys())}")
    except Exception as e:
        print(f"Error: {e}")
