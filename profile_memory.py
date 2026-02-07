import subprocess
import sys
import time

import psutil


def monitor_pytest():
    # Start pytest in a subprocess
    process = subprocess.Popen(
        [sys.executable, "-m", "pytest", "-v", "tests"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    ps_process = psutil.Process(process.pid)
    
    print(f"Monitoring pytest (PID: {process.pid})")
    
    max_rss = 0
    try:
        while process.poll() is None:
            try:
                # Get memory info of the main process and all its children
                mem_info = ps_process.memory_info()
                rss = mem_info.rss
                
                # Check children (important for xdist or multiprocessing)
                for child in ps_process.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                
                max_rss = max(max_rss, rss)
                print(f"Current Memory Usage: {rss / 1024 / 1024:.2f} MB | Max: {max_rss / 1024 / 1024:.2f} MB")
                
                # Read some output to keep it moving
                # line = process.stdout.readline()
                # if line:
                #     print(line.strip())
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            time.sleep(1)
            
    except KeyboardInterrupt:
        process.terminate()
        
    stdout, stderr = process.communicate()
    print(f"Pytest finished with exit code {process.returncode}")
    print(f"Max Memory Usage: {max_rss / 1024 / 1024:.2f} MB")
    
    if process.returncode == 137:
        print("DETECTED OOM (Exit Code 137)")
    
    with open("memory_profile.log", "w") as f:
        f.write(f"Max RSS: {max_rss / 1024 / 1024:.2f} MB\n")
        f.write(f"Exit Code: {process.returncode}\n")
        f.write("STDOUT:\n")
        f.write(stdout[-2000:]) # Last 2000 chars
        f.write("\nSTDERR:\n")
        f.write(stderr)

if __name__ == "__main__":
    monitor_pytest()
