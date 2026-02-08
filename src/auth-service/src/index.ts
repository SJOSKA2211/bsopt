import 'dotenv/config'
import { serve } from '@hono/node-server'
import { Hono } from 'hono'
import { auth } from './auth'

export const app = new Hono()

app.on(['GET', 'POST'], '/api/auth/**', async (c) => {
  let req = c.req.raw;
  
  if (c.req.path === '/api/auth/login' && c.req.method === 'POST') {
    const url = new URL(req.url);
    url.pathname = '/api/auth/sign-in/email';
    req = new Request(url.toString(), req);
  }

  return auth.handler(req);
});

app.get('/', (c) => c.text('Better Auth Service Running 🥒'))

app.get('/openapi.json', async (c) => {
    const openAPISchema = await auth.api.generateOpenAPISchema();
    return c.json(openAPISchema);
});

const port = 3001
console.log(`Server is running on port ${port}`)

serve({
  fetch: app.fetch,
  port
})
