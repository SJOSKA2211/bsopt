-- This query helps identify queries that are using temporary files, which
-- is an indicator that `work_mem` may be too small.
--
-- Run this query in your database to identify candidates for `work_mem` tuning.
-- If you see queries consistently writing a large amount of temporary blocks,
-- consider increasing `work_mem` for the session or globally.

SELECT
    (total_time / 1000 / 60) as total_minutes,
    (total_time/calls) as average_time,
    query,
    calls,
    temp_blks_read,
    temp_blks_written
FROM pg_stat_statements
WHERE temp_blks_written > 0
ORDER BY temp_blks_written DESC
LIMIT 20;
