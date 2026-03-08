"""
Centralized storage for Redis LUA scripts.
Optimizes performance by reducing network overhead and ensuring atomicity.
"""

# Sliding Window Rate Limiter
# Keys: [rate_limit_key]
# Args: [window_size_ms, limit, current_timestamp_ms, request_id]
SLIDING_WINDOW_RL = """
local key = KEYS[1]
local window = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local request_id = ARGV[4]

local window_start = now - window

-- 1. Remove old requests outside the window
redis.call('ZREMRANGEBYSCORE', key, 0, window_start)

-- 2. Count current requests in window
local current_count = redis.call('ZCARD', key)

if current_count < limit then
    -- 3. Add current request
    redis.call('ZADD', key, now, request_id)
    -- 4. Refresh TTL
    redis.call('PEXPIRE', key, window)
    return {1, current_count + 1}
else
    return {0, current_count}
end
"""
