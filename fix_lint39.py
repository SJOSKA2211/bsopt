with open("src/workers/tasks/email_tasks.py", "r") as f:
    c = f.read()

c = c.replace('            async with get_async_db_context() as db:\n                await db.execute(\n                    update(EmailLog)\n                    .where(EmailLog.id == log_id)\n                    .values(status="failed", error_message=str(e))', '            async with get_async_db_context() as db:\n                await db.execute(\n                    update(EmailLog)\n                    .where(EmailLog.id == log_id)\n                    .values(status="failed", error_message=str(exc))')

c = c.replace('        async def _log_error():\n', '        async def _log_error(exc):\n')
c = c.replace('        loop.run_until_complete(_log_error())\n', '        loop.run_until_complete(_log_error(e))\n')

with open("src/workers/tasks/email_tasks.py", "w") as f:
    f.write(c)
