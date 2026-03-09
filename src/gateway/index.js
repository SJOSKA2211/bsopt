'use strict';

const { ApolloGateway, IntrospectAndCompose } = require('@apollo/gateway');
const { ApolloServer } = require('@apollo/server');
const fastify = require('fastify');
const { fastifyApolloHandler, fastifyApolloDrainPlugin } = require('@as-integrations/fastify');

// Configuration
const port = parseInt(process.env.PORT || '4000', 10);
const subgraphs = [
  { name: 'api', url: process.env.API_URL || 'http://api:8000/graphql' },
  { name: 'portfolio', url: process.env.PORTFOLIO_URL || 'http://portfolio:8000/graphql' },
  { name: 'neural-pricing', url: process.env.PRICING_URL || 'http://neural-pricing:8000/graphql' },
];

async function start() {
  // 1. Initialize Fastify with Pino logger
  const app = fastify({
    logger: {
      level: process.env.LOG_LEVEL || 'info',
      serializers: {
        req(request) {
          return {
            method: request.method,
            url: request.url,
            hostname: request.hostname,
          };
        },
      },
    },
    disableRequestLogging: process.env.NODE_ENV === 'production',
  });

  // 2. Register standard plugins
  await app.register(require('@fastify/helmet'), {
    contentSecurityPolicy: process.env.NODE_ENV === 'production',
  });
  await app.register(require('@fastify/cors'));
  await app.register(require('@fastify/compress'));

  // 3. Initialize Apollo Gateway
  const gateway = new ApolloGateway({
    supergraphSdl: new IntrospectAndCompose({
      subgraphs,
      pollIntervalInMs: process.env.NODE_ENV === 'production' ? 60000 : 10000,
    }),
    buildService({ url }) {
      return new (require('@apollo/gateway').RemoteGraphQLDataSource)({
        url,
        willSendRequest({ request, context }) {
          request.http.headers.set('user-agent', 'ApolloGateway/2.0');
          if (context && context.headers && context.headers.authorization) {
            request.http.headers.set('authorization', context.headers.authorization);
          }
        },
      });
    },
    debug: process.env.DEBUG === 'true',
  });

  // 4. Initialize Apollo Server
  const server = new ApolloServer({
    gateway,
    plugins: [
      fastifyApolloDrainPlugin(app),
      require('@apollo/server-plugin-response-cache').default(),
    ],
    introspection: process.env.NODE_ENV !== 'production',
  });

  try {
    await server.start();
    
    // 5. Register Apollo Handler
    app.route({
      method: ['GET', 'POST', 'OPTIONS'],
      url: '/graphql',
      handler: fastifyApolloHandler(server, {
        context: async (request) => ({
          headers: request.headers,
        }),
      }),
    });
  } catch (err) {
    app.log.error(err, 'Failed to start Apollo Server / Gateway. GraphQL endpoint will be unavailable.');
  }

  // 6. Standardized Health Check
  app.get('/health', async () => {
    return {
      status: 'operational',
      service: 'gateway',
      timestamp: new Date().toISOString(),
    };
  });

  // 7. Graceful Shutdown
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

  // 8. Start Server
  try {
    await app.listen({ port, host: '0.0.0.0' });
    app.log.info(`High-Performance Federated Gateway (Fastify) ready at http://0.0.0.0:${port}/graphql`);
  } catch (err) {
    app.log.error(err);
    process.exit(1);
  }
}

start();
