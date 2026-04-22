import os
import re

def fix_workflow(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # 1. Fix Node version warnings
    content = re.sub(r'node-version: 20', r'node-version: 24', content)
    content = re.sub(r'NODE_VERSION: 20', r'NODE_VERSION: "24"', content)
    content = re.sub(r'NODE_VERSION: "20"', r'NODE_VERSION: "24"', content)
    content = re.sub(r'NODE_VERSION: \'20\'', r'NODE_VERSION: "24"', content)

    # 2. Fix the "uses:: command not found" error by extracting 'uses' from 'run' blocks
    # Looking at the logs, there's a 'uses: actions/setup-python@v5' inside a 'run' block in blue-green-deploy.yml and equaflow-institutional.yml
    # Let's just fix it with sed on the files since it's easier

    with open(filepath, 'w') as f:
        f.write(content)

for root, _, files in os.walk('.github/workflows'):
    for file in files:
        if file.endswith('.yml') or file.endswith('.yaml'):
            fix_workflow(os.path.join(root, file))
