const { ApolloServer } = require('@apollo/server');
const { fastifyApolloHandler, fastifyApolloDrainPlugin } = require('@as-integrations/fastify');
const { ApolloGateway, IntrospectAndCompose, RemoteGraphQLDataSource } = require('@apollo/gateway');
const Fastify = require('fastify');
const cors = require('@fastify/cors');
const helmet = require('@fastify/helmet');
const compress = require('@fastify/compression');
const pino = require('pino');
const responseCachePlugin = require('@apollo/server-plugin-response-cache').default;
const cluster = require('cluster');
const os = require('os');

const logger = pino({
  level: process.env.LOG_LEVEL || 'info',
  transport: process.env.NODE_ENV !== 'production' ? {
    target: 'pino-pretty',
    options: { colorize: true }
  } : undefined
});

class AuthenticatedDataSource extends RemoteGraphQLDataSource {
  willSendRequest({ request, context }) {
    if (context.headers && context.headers['authorization']) {
      request.http.headers.set('authorization', context.headers['authorization']);
    }
    request.http.headers.set('X-SSL-Client-Verify', 'SUCCESS');
    request.http.headers.set('X-SSL-Client-S-DN', 'CN=api-gateway');
    
    if (context.headers && context.headers['x-user-id']) {
      request.http.headers.set('X-User-Id', context.headers['x-user-id']);
    }
  }
}

const fs = require('fs');

const gatewayConfig = process.env.SUPERGRAPH_SDL_PATH && fs.existsSync(process.env.SUPERGRAPH_SDL_PATH)
  ? { supergraphSdl: fs.readFileSync(process.env.SUPERGRAPH_SDL_PATH, 'utf-8') }
  : {
      supergraphSdl: new IntrospectAndCompose({
        subgraphs: [
          { name: 'options', url: process.env.OPTIONS_URL || 'http://localhost:8000/graphql' },
          { name: 'pricing', url: process.env.PRICING_URL || 'http://localhost:8001/graphql' },
          { name: 'ml', url: process.env.ML_URL || 'http://localhost:8002/graphql' },
          { name: 'portfolio', url: process.env.PORTFOLIO_URL || 'http://localhost:8003/graphql' },
          { name: 'marketdata', url: process.env.MARKETDATA_URL || 'http://localhost:8004/graphql' },
        ],
      }),
    };

const gateway = new ApolloGateway({
  ...gatewayConfig,
  buildService({ url }) {
    return new AuthenticatedDataSource({ url });
  },
});

async function startServer() {
  const fastify = Fastify({ 
    logger: false, // Apollo has its own logging or we use pino directly
    disableRequestLogging: true
  });

  // Fastify optimized plugins
  await fastify.register(helmet, { contentSecurityPolicy: false });
  await fastify.register(cors);
  await fastify.register(compress);
  
  // 🚀 SOTA: Auth Proxy Singularity
  await fastify.register(require('@fastify/reply-from'), {
    base: process.env.AUTH_URL || 'http://localhost:3001'
  });

  await gateway.load();

  const server = new ApolloServer({
    gateway,
    plugins: [
      responseCachePlugin(),
      fastifyApolloDrainPlugin(fastify),
    ],
  });

  await server.start();

  // Proxy Better Auth paths
  fastify.all('/api/auth/*', (request, reply) => {
    reply.from(request.url);
  });

  fastify.route({
    method: ['GET', 'POST', 'OPTIONS'],
    url: '/',
    handler: fastifyApolloHandler(server, {
      context: async (request) => ({ headers: request.headers }),
    }),
  });

  const port = process.env.PORT || 4000;
  await fastify.listen({ port, host: '0.0.0.0' });
  logger.info(`🚀 Gateway worker ${process.pid} ready at http://localhost:${port}/`);
}

if (process.env.NODE_ENV === 'production' && cluster.isPrimary) {
  const numCPUs = os.cpus().length;
  logger.info(`Primary ${process.pid} is running. Forking ${numCPUs} workers...`);
  for (let i = 0; i < numCPUs; i++) cluster.fork();
  cluster.on('exit', (worker) => {
    logger.warn(`Worker ${worker.process.pid} died. Forking a new one...`);
    cluster.fork();
  });
} else {
  startServer().catch(err => {
    logger.error(err, 'failed_to_start_server');
    process.exit(1);
  });
}
