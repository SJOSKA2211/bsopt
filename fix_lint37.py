with open("src/workers/tasks/email_tasks.py", "r") as f:
    c = f.read()
import re
c = re.sub(r'        except BaseException as e:', r'        except Exception as e:', c)
c = re.sub(r'        except BaseException:', r'        except Exception as e:', c)
c = re.sub(r'        except Exception:', r'        except Exception as e:', c)
c = re.sub(r'        except Exception as e as e:', r'        except Exception as e:', c)
with open("src/workers/tasks/email_tasks.py", "w") as f:
    f.write(c)
