import re

# Fix email_tasks exception correctly this time.
with open("src/workers/tasks/email_tasks.py", "r") as f:
    c = f.read()

# Let's fix the inner except
c = re.sub(r'        except Exception:', r'        except Exception as e:', c)
with open("src/workers/tasks/email_tasks.py", "w") as f:
    f.write(c)

with open("src/auth/auth.py", "r") as f:
    c = f.read()

# Move the auth core imports
match = re.search(r'from src.auth.core.hashing import hasher.*?from src.auth.core.webauthn import webauthn_service', c, re.DOTALL)
if match:
    imports = match.group(0)
    c = c.replace(imports, '')
    c = c.replace('from fastapi.security import APIKeyHeader', 'from fastapi.security import APIKeyHeader\n' + imports)

with open("src/auth/auth.py", "w") as f:
    f.write(c)
