import os
import subprocess
import sys


def check():
    print(f"Python: {sys.version}")
    print(f"CWD: {os.getcwd()}")
    print(f"ENVIRONMENT: {os.environ.get('ENVIRONMENT')}")
    print(f"BSOPT_ALLOW_WEAK_SECRETS: {os.environ.get('BSOPT_ALLOW_WEAK_SECRETS')}")
    
    try:
        import lightning
        print(f"Lightning: {lightning.__version__}")
    except ImportError:
        print("Lightning: NOT FOUND")
        
    try:
        res = subprocess.run(["whoami"], capture_output=True, text=True)
        print(f"Whoami: {res.stdout.strip()}")
    except Exception as e:
        print(f"Whoami error: {e}")

if __name__ == "__main__":
    check()
