
import os
import sys

replacements = {
    "Joseph Kamau Maina": "Joseph Kamau Maina",
    "": "",
    "System check complete.": "System check complete.",
    "The User": "The User",
    "Advanced": "Advanced",
    "The Advanced Financial Manifold": "High-Performance Financial Engine",
    "I'm Joseph Kamau Maina! And I'm the only one who actually knows how to scale a derivative pricing engine!": "Advanced Derivative Pricing Engine.",
    "Created by the Joseph Kamau Maina Extension. Shut up and compute.": "Created by Joseph Kamau Maina."
}

extensions = ['.py', '.sh', '.md', '.txt', '.json', '.ts', '.html']
exclude_dirs = {'.git', '__pycache__', 'node_modules', '.gemini', 'venv', 'env', '.venv', '.venv_rick_312'}

def replace_in_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        new_content = content
        for old, new in replacements.items():
            new_content = new_content.replace(old, new)
        
        if new_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Updated: {filepath}", flush=True)
    except Exception as e:
        print(f"Error reading/writing {filepath}: {e}", file=sys.stderr, flush=True)

def main():
    print("Starting cleanup...", flush=True)
    root_dir = '/home/kamau/bsopt'
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Remove excluded directories - modify in place
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        
        for filename in filenames:
            if any(filename.endswith(ext) for ext in extensions):
                if filename == 'cleanup_rick.py': continue
                replace_in_file(os.path.join(dirpath, filename))
    print("Cleanup complete.", flush=True)

if __name__ == "__main__":
    main()
