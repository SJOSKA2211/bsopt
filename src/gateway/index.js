'use strict';

const http = require('http');
const { ApolloGateway, IntrospectAndCompose } = require('@apollo/gateway');
const { ApolloServer } = require('@apollo/server');
const { startStandaloneServer } = require('@apollo/server/standalone');
const pino = require('pino');

const logger = pino({ level: process.env.LOG_LEVEL || 'info' });

const subgraphs = [
  { name: 'api', url: process.env.API_URL || 'http://api:8000/graphql' },
  { name: 'portfolio', url: process.env.PORTFOLIO_URL || 'http://portfolio:8000/graphql' },
  { name: 'neural-pricing', url: process.env.PRICING_URL || 'http://neural-pricing:8000/graphql' },
];

const gateway = new ApolloGateway({
  supergraphSdl: new IntrospectAndCompose({
    subgraphs,
    pollIntervalInMs: 10000,
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

const server = new ApolloServer({
  gateway,
  logger,
});

async function start() {
  const port = parseInt(process.env.PORT || '4000', 10);

  // Start the Apollo standalone server on port 4000
  const { url } = await startStandaloneServer(server, {
    context: async ({ req }) => ({
      headers: req.headers,
    }),
    listen: { port, host: '0.0.0.0' },
  });

  logger.info(`God Mode Federated Gateway ready at ${url}`);

  // Lightweight health check on the same port via a separate HTTP server on /health
  // Since startStandaloneServer owns the port, we run health on port+1 internally
  // and expose it via docker healthcheck on port 4000 (Apollo handles /health natively
  // through its introspection on that port, but a dedicated path is cleaner).
  // Instead: attach a /health route using Apollo's built-in landing page bypass trick.
  // Simpler: run a tiny HTTP server on 4001 for the healthcheck probe.
  const healthServer = http.createServer((req, res) => {
    if (req.url === '/health') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ status: 'gateway_operational' }));
    } else {
      res.writeHead(404);
      res.end();
    }
  });

  healthServer.listen(4001, '0.0.0.0', () => {
    logger.info('Health check server listening on port 4001');
  });
}

start().catch(err => {
  logger.error(err);
  process.exit(1);
});
