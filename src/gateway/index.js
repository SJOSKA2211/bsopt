'use strict';

const fastify = require('fastify');
const mercurius = require('mercurius');
const Piscina = require('piscina');
const path = require('path');

// Configuration
const port = parseInt(process.env.PORT || '4000', 10);
const subgraphs = [
  { name: 'api', url: process.env.API_URL || 'http://api:8000/graphql' },
  { name: 'neural-pricing', url: process.env.PRICING_URL || 'http://neural-pricing:8000/graphql' },
];

// Initialize Piscina worker pool
const piscina = new Piscina({
  filename: path.resolve(__dirname, 'worker.js'),
  minThreads: 2,
  maxThreads: 8,
});

// Institutional Trace ID Generator
const { v4: uuidv4 } = require('uuid');
const Opossum = require('opossum');

// Circuit Breaker Options
const breakerOptions = {
  timeout: 3000,
  errorThresholdPercentage: 50,
  resetTimeout: 30000
};

async function start() {
  const app = fastify({
    logger: {
      level: process.env.LOG_LEVEL || 'info',
    },
    disableRequestLogging: process.env.NODE_ENV === 'production',
  });

  // 1. Institutional Tracing (X-Request-ID) & Versioning
  app.addHook('onRequest', async (request, reply) => {
    request.headers['x-request-id'] = request.headers['x-request-id'] || uuidv4();
    reply.header('x-request-id', request.headers['x-request-id']);
    reply.header('x-api-version', 'v1');
  });

  // 2. Register Standard Plugins
  await app.register(require('@fastify/helmet'), {
    contentSecurityPolicy: process.env.NODE_ENV === 'production',
  });
  await app.register(require('@fastify/cors'));
  await app.register(require('@fastify/compress'));

  // 3. Circuit Breaker Initialization for Upstreams
  const breakers = subgraphs.map(s => {
    const breaker = new Opossum(async (payload) => {
      // Logic for proxied GraphQL calls
      return { status: 'ok' };
    }, breakerOptions);
    
    breaker.on('open', () => app.log.warn(`Circuit Breaker OPEN for ${s.name}`));
    breaker.on('halfOpen', () => app.log.info(`Circuit Breaker HALF-OPEN for ${s.name}`));
    breaker.on('close', () => app.log.info(`Circuit Breaker CLOSED for ${s.name}`));
    
    return { name: s.name, breaker };
  });

  // Register Mercurius Gateway
  await app.register(mercurius, {
    gateway: {
      src: subgraphs,
      pollingInterval: process.env.NODE_ENV === 'production' ? 60 : 10,
    },
    graphiql: process.env.NODE_ENV !== 'production',
    jit: 1, // Enable Just-In-Time optimization
    errorFormatter: (execution, context) => {
      app.log.error(execution.errors, 'Mercurius Execution Errors');
      return mercurius.defaultErrorFormatter(execution, context);
    },
    context: async (request) => {
      // Use Piscina to transform incoming headers or metadata if needed
      const headers = await piscina.run({
        type: 'PROCESS_DATA',
        payload: request.headers,
      });
      return { headers };
    },
  });

  // Health Check
  app.get('/health', async () => {
    return {
      status: 'operational',
      service: 'gateway',
      piscina: {
        threads: piscina.threads.length,
        queueSize: piscina.queueSize,
      },
      timestamp: new Date().toISOString(),
    };
  });

  // Graceful Shutdown
  const signals = ['SIGTERM', 'SIGINT'];
  for (const signal of signals) {
    process.on(signal, async () => {
      app.log.info({ signal }, 'Closing gateway...');
      try {
        await app.close();
        process.exit(0);
      } catch (err) {
        app.log.error(err, 'Error during shutdown');
        process.exit(1);
      }
    });
  }

  // Start Server
  try {
    await app.listen({ port, host: '0.0.0.0' });
    app.log.info(`High-Performance Mercurius Gateway ready at http://0.0.0.0:${port}/graphql`);
  } catch (err) {
    app.log.error(err);
    process.exit(1);
  }
}

start();
