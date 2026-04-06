with open("src/ml/aiops/backtest_evaluator.py", "r") as f:
    c = f.read()

c = c.replace('from typing import TypedDict, cast, cast', 'from typing import TypedDict, cast')
if 'from typing import cast' not in c and 'cast' in c and 'from typing import TypedDict, cast' not in c:
    c = c.replace('from typing import TypedDict', 'from typing import TypedDict, cast')

with open("src/ml/aiops/backtest_evaluator.py", "w") as f:
    f.write(c)

with open("src/workers/tasks/email_tasks.py", "r") as f:
    c = f.read()
import re
c = re.sub(r'        except BaseException as e:', r'        except Exception as e:', c)
c = re.sub(r'        except BaseException:', r'        except Exception as e:', c)
c = re.sub(r'        except Exception:', r'        except Exception as e:', c)
with open("src/workers/tasks/email_tasks.py", "w") as f:
    f.write(c)
