import re

# email_tasks.py
with open("src/workers/tasks/email_tasks.py", "r") as f:
    c = f.read()

# Let's fix the inner except
c = re.sub(r'        except Exception as e as e:', r'        except Exception as e:', c)
c = re.sub(r'        except BaseException as e as e:', r'        except Exception as e:', c)
c = re.sub(r'        except Exception:\n            logger.error\("audit_log_persistence_failed", error=str\(e\)\)', r'        except Exception as e:\n            logger.error("audit_log_persistence_failed", error=str(e))', c)
c = re.sub(r'        except Exception:\n', r'        except Exception as e:\n', c)
with open("src/workers/tasks/email_tasks.py", "w") as f:
    f.write(c)
