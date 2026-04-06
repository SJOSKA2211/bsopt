import re

# Fix email_tasks exception correctly this time.
with open("src/workers/tasks/email_tasks.py", "r") as f:
    c = f.read()

# Let's fix the inner except
c = c.replace('        except Exception as e:\n        logger.error', '    except Exception as e:\n        logger.error')
c = c.replace('        except Exception as e:', '    except Exception as e:')

with open("src/workers/tasks/email_tasks.py", "w") as f:
    f.write(c)
