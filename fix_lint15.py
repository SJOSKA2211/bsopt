import re

# Fix main.py imports
with open("src/ml/main.py", "r") as f:
    c = f.read()

c = c.replace('from fastapi import FastAPI, Request', 'from fastapi import FastAPI, Request, Depends')
c = c.replace('from fastapi import FastAPI, Request, Depends, Depends', 'from fastapi import FastAPI, Request, Depends')

with open("src/ml/main.py", "w") as f:
    f.write(c)
