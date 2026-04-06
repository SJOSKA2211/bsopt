with open("src/workers/tasks/email_tasks.py", "r") as f:
    c = f.read()
import re
c = re.sub(r'        except BaseException as e:', '        except Exception as e:', c)
c = re.sub(r'        except BaseException:', '        except Exception as e:', c)
c = re.sub(r'        except Exception:', '        except Exception as e:', c)
c = re.sub(r'        except Exception as e as e:', '        except Exception as e:', c)

with open("src/workers/tasks/email_tasks.py", "w") as f:
    f.write(c)
