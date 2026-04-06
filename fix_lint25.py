with open("src/ml/aiops/backtest_evaluator.py", "r") as f:
    c = f.read()
if 'cast(' in c and 'from typing import cast' not in c:
    c = c.replace('from typing import TypedDict, cast', 'from typing import TypedDict, cast')
    c = c.replace('from typing import TypedDict', 'from typing import TypedDict, cast')
with open("src/ml/aiops/backtest_evaluator.py", "w") as f:
    f.write(c)

with open("src/workers/tasks/email_tasks.py", "r") as f:
    c = f.read()
c = c.replace('        except Exception:', '        except Exception as e:')
with open("src/workers/tasks/email_tasks.py", "w") as f:
    f.write(c)

with open("src/shared/config.py", "r") as f:
    lines = f.readlines()

out = []
skip = False
for line in lines:
    if 'def tracking_uri(self) -> str:' in line:
        skip = True
        out.pop() # remove @property
    if skip:
        if line.strip() == '' or line.startswith('    @') or line.startswith('    def ') or line.startswith('    class '):
            if not line.startswith('    def tracking_uri(self) -> str:'):
                skip = False

    if not skip:
        out.append(line)

with open("src/shared/config.py", "w") as f:
    f.write("".join(out))
