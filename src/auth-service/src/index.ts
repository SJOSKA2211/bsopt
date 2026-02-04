import 'dotenv/config'
import { serve } from '@hono/node-server'
import { Hono } from 'hono'
import { auth } from './auth'

const app = new Hono()
const authApp = new Hono() // NEW LINE

authApp.all('*', async (c) => { // NEW BLOCK START
  return auth.handler(c.req.raw);
}); // NEW BLOCK END

app.route('/api/auth', authApp) // NEW LINE

app.get('/', (c) => c.text('Better Auth Service Running 🥒'))

app.get('/openapi.json', async (c) => {
    const openAPISchema = await auth.api.generateOpenAPISchema();
    return c.json(openAPISchema);
});

const port = 4000
console.log(`Server is running on port ${port}`)

serve({
  fetch: app.fetch,
  port
})
