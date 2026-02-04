const Fastify = require('fastify');
const proxy = require('@fastify/reply-from');
const cors = require('@fastify/cors');
const helmet = require('@fastify/helmet');
const pino = require('pino');

const logger = pino({ level: process.env.LOG_LEVEL || 'info' });

const subgraphs = {
  options: process.env.OPTIONS_URL || 'http://localhost:8000',
  pricing: process.env.PRICING_URL || 'http://localhost:8001',
  ml: process.env.ML_URL || 'http://localhost:8002',
  auth: process.env.AUTH_URL || 'http://localhost:3001'
};

const app = Fastify({ logger: false });

async function start() {
  await app.register(helmet, { contentSecurityPolicy: false });
  await app.register(cors);
  
  // Register proxy targets
  for (const [name, url] of Object.entries(subgraphs)) {
    await app.register(proxy, {
      name,
      base: url,
      prefix: `/api/v1/${name}`
    });
  }

  // Unified Route Handler
  app.all('/api/v1/:service/*', (request, reply) => {
    const { service } = request.params;
    if (subgraphs[service]) {
      reply.from(request.url.replace(`/api/v1/${service}`, ''));
    } else {
      reply.code(404).send({ error: 'Service not found' });
    }
  });

  const port = process.env.PORT || 4000;
  await app.listen({ port, host: '0.0.0.0' });
  logger.info(`🚀 Singularity Gateway ready at http://localhost:${port}/`);
}

start().catch(err => {
  logger.error(err);
  process.exit(1);
});