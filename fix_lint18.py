with open("src/ml/main.py", "r") as f:
    c = f.read()

# Make sure Depends is imported
c = c.replace('from fastapi import FastAPI, Request', 'from fastapi import FastAPI, Request, Depends')
# De-duplicate Depends
c = c.replace('from fastapi import FastAPI, Request, Depends, Depends', 'from fastapi import FastAPI, Request, Depends')

with open("src/ml/main.py", "w") as f:
    f.write(c)


with open("src/ml/aiops/anomaly_detector.py", "r") as f:
    c = f.read()
# Try removing unused ray
c = c.replace('try:\n    import ray\n    RAY_AVAILABLE = True\nexcept ImportError:\n    pass', '')
# Ensure RAY_AVAILABLE is false then
if 'RAY_AVAILABLE = False\n' not in c:
    c = c.replace('TORCH_AVAILABLE = False', 'TORCH_AVAILABLE = False\nRAY_AVAILABLE = False')
with open("src/ml/aiops/anomaly_detector.py", "w") as f:
    f.write(c)

with open("src/workers/tasks/email_tasks.py", "r") as f:
    c = f.read()
c = c.replace('        except BaseException as e:', '        except Exception as e:')
c = c.replace('        except BaseException:', '        except Exception as e:')
c = c.replace('        except Exception:', '        except Exception as e:')
with open("src/workers/tasks/email_tasks.py", "w") as f:
    f.write(c)
