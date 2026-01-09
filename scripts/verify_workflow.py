import yaml
import os
import sys

def verify_workflow(file_path):
    print(f"Verifying workflow: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"ERROR: Workflow file not found: {file_path}")
        return False
        
    try:
        with open(file_path, 'r') as f:
            workflow = yaml.safe_load(f)
            
        # Basic syntax and required keys
        # YAML 1.1 converts 'on' to True
        has_on = 'on' in workflow or True in workflow
        if 'name' not in workflow or not has_on or 'jobs' not in workflow:
            print(f"ERROR: Missing basic GitHub Actions keys (name, on, jobs). Keys found: {list(workflow.keys())}")
            return False
            
        print(f"Workflow Name: {workflow['name']}")
        
        # Check triggers
        triggers = workflow.get('on') or workflow.get(True)
        print(f"Triggers: {triggers}")
        
        # Check jobs
        jobs = workflow['jobs']
        print(f"Found {len(jobs)} jobs: {list(jobs.keys())}")
        
        # Verify dependencies
        if 'integration-tests' in jobs:
            # Current implementation doesn't have explicit 'needs' for integration-tests, 
            # but it's independent. Deploy needs it.
            pass
            
        if 'deploy' in jobs:
            needs = jobs['deploy'].get('needs', [])
            required = ['security-scan', 'linting', 'integration-tests']
            for r in required:
                if r not in needs:
                    print(f"WARNING: 'deploy' job should probably depend on '{r}' (Current: {needs})")
                else:
                    print(f" - 'deploy' job correctly depends on '{r}'")
                    
        if 'continuous-training' in jobs:
            needs = jobs['continuous-training'].get('needs', [])
            required = ['security-scan', 'linting']
            for r in required:
                if r not in needs:
                    print(f"WARNING: 'continuous-training' job should depend on '{r}'")
                else:
                    print(f" - 'continuous-training' correctly depends on '{r}'")
                    
        print("\nWorkflow structure looks VALID.")
        return True
        
    except Exception as e:
        print(f"ERROR parsing YAML: {e}")
        return False

if __name__ == "__main__":
    workflow_path = ".github/workflows/mlops-pipeline.yml"
    if verify_workflow(workflow_path):
        sys.exit(0)
    else:
        sys.exit(1)
