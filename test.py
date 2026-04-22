import yaml
import sys
try:
    yaml.safe_load(open('.github/workflows/blue-green-deploy.yml'))
    print("YAML is valid.")
except Exception as e:
    print(f"Error parsing YAML: {e}")
    sys.exit(1)
