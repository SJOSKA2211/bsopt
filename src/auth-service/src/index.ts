import 'dotenv/config'
import { serve } from '@hono/node-server'
import { Hono } from 'hono'
import { cors } from 'hono/cors'
import { logger } from 'hono/logger'
import { secureHeaders } from 'hono/secure-headers'
import { auth } from './auth'

export const app = new Hono()

// Security & Logging Middleware
app.use('*', logger())
app.use('*', secureHeaders())
app.use('*', cors({
  origin: process.env.CORS_ORIGIN || '*',
  allowMethods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowHeaders: ['Content-Type', 'Authorization', 'X-CSRF-Token'],
  exposeHeaders: ['Content-Length'],
  maxAge: 600,
  credentials: true,
}))

// Custom Rate Limiting for Auth
const rateLimitMap = new Map<string, { count: number, reset: number }>();

app.use('/api/auth/*', async (c, next) => {
  const ip = c.req.header('x-forwarded-for') || 'anonymous';
  const now = Date.now();
  const limit = 10; // 10 requests
  const window = 60000; // per minute

  const record = rateLimitMap.get(ip) || { count: 0, reset: now + window };

  if (now > record.reset) {
    record.count = 0;
    record.reset = now + window;
  }

  record.count++;
  rateLimitMap.set(ip, record);

  if (record.count > limit) {
    return c.json({ error: 'Too many requests' }, 429);
  }

  await next();
});

// Health check
app.get('/', (c) => c.text('Better Auth Service Running '))

// OpenAPI Schema
app.get('/openapi.json', async (c) => {
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
