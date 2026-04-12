import yaml

try:
    yaml.safe_load(open('.github/workflows/equaflow-institutional.yml'))
    print("equaflow OK")
except Exception as e:
    print("equaflow Error:", e)

try:
    yaml.safe_load(open('.github/workflows/blue-green-deploy.yml'))
    print("blue-green OK")
except Exception as e:
    print("blue-green Error:", e)
