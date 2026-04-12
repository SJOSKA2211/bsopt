import yaml

with open('.github/workflows/equaflow-institutional.yml') as f:
    data = yaml.safe_load(f)

# Find the unit-tests job
for job_name, job_data in data['jobs'].items():
    if job_name == 'unit-tests':
        steps = job_data['steps']
        new_steps = []
        for step in steps:
            if step.get('name') == 'Backend Tests':
                new_steps.append({
                    'name': 'Setup Python for Backend',
                    'if': "matrix.layer == 'backend'",
                    'uses': 'actions/setup-python@v5',
                    'with': {'python-version': '${{ env.PYTHON_VERSION }}'}
                })
                new_steps.append({
                    'name': 'Run Backend Tests',
                    'if': "matrix.layer == 'backend'",
                    'run': "pip install .[dev,api,ml]\npytest tests/unit\n"
                })
            elif step.get('name') == 'Frontend Tests':
                new_steps.append({
                    'name': 'Setup Node for Frontend',
                    'if': "matrix.layer == 'frontend'",
                    'uses': 'actions/setup-node@v4',
                    'with': {'node-version': '${{ env.NODE_VERSION }}'}
                })
                new_steps.append({
                    'name': 'Run Frontend Tests',
                    'if': "matrix.layer == 'frontend'",
                    'run': "cd src/frontend\nnpm install\nnpm run test\n"
                })
            else:
                new_steps.append(step)
        job_data['steps'] = new_steps

with open('.github/workflows/equaflow-institutional.yml', 'w') as f:
    yaml.dump(data, f, sort_keys=False, default_flow_style=False)
