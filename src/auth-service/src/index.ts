import 'dotenv/config'
import { serve } from '@hono/node-server'
import { Hono } from 'hono'
import { cors } from 'hono/cors'
import { logger } from 'hono/logger'
import { secureHeaders } from 'hono/secure-headers'
import { compress } from 'hono/compress'
import Redis from 'ioredis'
import { v4 as uuidv4 } from 'uuid'
import { auth } from './auth'
import { openAPI } from "better-auth/plugins"

export const app = new Hono()

const redis = new Redis(process.env.REDIS_URL || 'redis://redis:6379')

// LUA script for atomic sliding window rate limiting
const SLIDING_WINDOW_RL_SCRIPT = `
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
`;

// PII Masking Logger
const piiMaskingLogger = (str: string) => {
  // Simple masking for IPs and Emails in logs
  const masked = str
    .replace(/\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b/g, 'XXX.XXX.XXX.XXX')
    .replace(/\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b/g, 'masked@email.com');
  console.log(masked);
};

// Security & Logging Middleware
app.use('*', logger(piiMaskingLogger))
app.use('*', compress())
app.use('*', secureHeaders())
app.use('*', cors({
  origin: process.env.CORS_ORIGIN || '*',
  allowMethods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowHeaders: ['Content-Type', 'Authorization', 'X-CSRF-Token'],
  exposeHeaders: ['Content-Length'],
  maxAge: 600,
  credentials: true,
}))

// Custom Redis Rate Limiting for Auth
app.use('/api/auth/*', async (c, next) => {
  const ip = c.req.header('x-forwarded-for') || 'anonymous';
  const nowMs = Date.now();
  const limit = 10; // 10 requests
  const windowMs = 60000; // per minute
  const requestId = uuidv4();
  const key = `rate_limit:auth:${ip}`;

  try {
    const result = await redis.eval(
      SLIDING_WINDOW_RL_SCRIPT,
      1,
      key,
      windowMs,
      limit,
      nowMs,
      requestId
    ) as [number, number];

    const [allowed, currentCount] = result;

    if (!allowed) {
      return c.json({ error: 'Too many requests' }, 429);
    }

    // Set headers similar to Python backend
    c.header('X-RateLimit-Limit', limit.toString());
    c.header('X-RateLimit-Remaining', Math.max(0, limit - currentCount).toString());
  } catch (err) {
    console.error('Rate limiting error:', err);
    // Fail open if Redis is down
  }

  await next();
});

// Health check
app.get('/', (c) => c.text('Better Auth Service Running '))

// OpenAPI Schema
app.get('/openapi.json', async (c) => {
  // @ts-ignore
  const openAPISchema = await auth.api.generateOpenAPISchema();
  return c.json(openAPISchema);
});

// Auth Middleware/Handler
app.all('/api/auth/*', async (c) => {
  // Internal rewrite for convenience
  if (c.req.path === '/api/auth/login' && c.req.method === 'POST') {
    const url = new URL(c.req.url);
    url.pathname = '/api/auth/sign-in/email';
    return auth.handler(new Request(url.toString(), c.req.raw));
  }

  return auth.handler(c.req.raw);
});

if (process.env.NODE_ENV !== 'test') {
  const port = Number(process.env.PORT) || 3001
  console.log(`Server is running on port ${port}`)

  serve({
    fetch: app.fetch,
    port
  })
}
