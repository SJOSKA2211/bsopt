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

# Distributed Delta Risk Check & State Commit
# Keys: [risk_state_key]
# Args: [trade_delta, max_net_delta]
DISTRIBUTED_RISK_CHECK = """
local key = KEYS[1]
local trade_delta = tonumber(ARGV[1])
local max_delta = tonumber(ARGV[2])

local current_delta = tonumber(redis.call('GET', key) or "0")
local new_delta = current_delta + trade_delta

if math.abs(new_delta) <= max_delta then
    redis.call('SET', key, tostring(new_delta))
    return {1, new_delta}
else
    return {0, current_delta}
end
"""

# Advanced Greeks Matrix & Safety Sync
# Keys: [risk_state_hash, global_kill_switch, blockchain_breaker]
# Args: [d_delta, d_gamma, d_vega, max_d, max_g, max_v]
ADVANCED_RISK_MATRIX = """
local risk_key = KEYS[1]
local kill_switch = KEYS[2]
local breaker_key = KEYS[3]

local d_delta = tonumber(ARGV[1])
local d_gamma = tonumber(ARGV[2])
local d_vega = tonumber(ARGV[3])
local max_d = tonumber(ARGV[4])
local max_g = tonumber(ARGV[5])
local max_v = tonumber(ARGV[6])

-- 1. Global Kill-Switch Check
if redis.call('GET', kill_switch) == '1' then
    return {0, 'KILL_SWITCH_ACTIVE'}
end

-- 2. Blockchain Breaker Check
if redis.call('GET', breaker_key) == 'OPEN' then
    return {0, 'BLOCKCHAIN_CIRCUIT_OPEN'}
end

-- 3. Fetch Current Greeks
local state = redis.call('HMGET', risk_key, 'delta', 'gamma', 'vega')
local curr_d = tonumber(state[1] or '0')
local curr_g = tonumber(state[2] or '0')
local curr_v = tonumber(state[3] or '0')

-- 4. Validate Limits
local new_d = curr_d + d_delta
local new_g = curr_g + d_gamma
local new_v = curr_v + d_vega

if math.abs(new_d) <= max_d and math.abs(new_g) <= max_g and math.abs(new_v) <= max_v then
    -- Commit Matrix
    redis.call('HMSET', risk_key, 'delta', new_d, 'gamma', new_g, 'vega', new_v, 'ts', redis.call('TIME')[1])
    return {1, new_d, new_g, new_v}
else
    return {0, 'LIMIT_EXCEEDED', curr_d, curr_g, curr_v}
end
"""
