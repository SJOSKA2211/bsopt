import 'dotenv/config'
import fastify from 'fastify'
import cors from '@fastify/cors'
import helmet from '@fastify/helmet'
import compress from '@fastify/compress'
import rateLimit from '@fastify/rate-limit'
import Redis from 'ioredis'
import { v4 as uuidv4 } from 'uuid'
import { auth } from './auth'

const port = Number(process.env.PORT) || 3001
const redisUrl = process.env.REDIS_URL || 'redis://redis:6379'
const redis = new Redis(redisUrl)

async function start() {
  const app = fastify({
    logger: {
      level: process.env.LOG_LEVEL || 'info',
      serializers: {
        req(request) {
          return {
            method: request.method,
            url: request.url,
            hostname: request.hostname,
            remoteAddress: request.ip,
          }
        },
      },
    },
    disableRequestLogging: process.env.NODE_ENV === 'production',
  })

  // 1. Register Plugins
  await app.register(helmet, {
    contentSecurityPolicy: process.env.NODE_ENV === 'production',
  })
  await app.register(cors, {
    origin: process.env.CORS_ORIGIN || '*',
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization', 'X-CSRF-Token'],
    exposedHeaders: ['Content-Length'],
    credentials: true,
  })
  await app.register(compress)

  // 2. Rate Limiting (Using @fastify/rate-limit with Redis)
  await app.register(rateLimit, {
    redis,
    max: 100,
    timeWindow: '1 minute',
    keyGenerator: (request) => request.headers['x-forwarded-for'] as string || request.ip,
    errorResponseBuilder: (request, context) => ({
      statusCode: 429,
      error: 'Too Many Requests',
      message: `Rate limit exceeded. Try again in ${context.after}`,
    }),
  })

  // 3. Health Check
  app.get('/health', async () => {
    return {
      status: 'operational',
      service: 'auth-service',
      timestamp: new Date().toISOString(),
    }
  })

  app.get('/', async () => {
    return 'Better Auth Service (Fastify) Running '
  })

  // 4. OpenAPI Schema
  app.get('/openapi.json', async (request, reply) => {
    // @ts-ignore
    const openAPISchema = await auth.api.generateOpenAPISchema()
    return reply.send(openAPISchema)
  })

  // 5. Better Auth Handler
  app.all('/api/auth/*', async (request, reply) => {
    // Ported internal rewrite logic
    if (request.url === '/api/auth/login' && request.method === 'POST') {
      const url = new URL(request.url, `http://${request.hostname}`)
      url.pathname = '/api/auth/sign-in/email'
      const rewrittenRequest = new Request(url.toString(), {
        method: request.method,
        headers: request.headers as HeadersInit,
        body: JSON.stringify(request.body),
      })
      const response = await auth.handler(rewrittenRequest)
      return reply.send(response)
    }

    // Standard handler
    const response = await auth.handler(request.raw)
    return reply.send(response)
  })

  // 6. Graceful Shutdown
  const signals: NodeJS.Signals[] = ['SIGTERM', 'SIGINT']
  signals.forEach((signal) => {
    process.on(signal, async () => {
      app.log.info({ signal }, 'Closing Auth Service...')
      try {
        await app.close()
        await redis.quit()
        process.exit(0)
      } catch (err) {
        app.log.error(err, 'Error during shutdown')
        process.exit(1)
      }
    })
  })

  // 7. Start Server
  try {
    await app.listen({ port, host: '0.0.0.0' })
    app.log.info(`Auth Service ready at http://0.0.0.0:${port}`)
  } catch (err) {
    app.log.error(err)
    process.exit(1)
  }
}

start()
