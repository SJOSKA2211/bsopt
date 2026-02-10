import 'dotenv/config'
import { serve } from '@hono/node-server'
import { Hono } from 'hono'
import { auth } from './auth'

export const app = new Hono()

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
