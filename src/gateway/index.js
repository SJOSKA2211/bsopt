const Fastify = require('fastify');
const { ApolloGateway, IntrospectAndCompose } = require('@apollo/gateway');
const { ApolloServer } = require('@apollo/server');
const fastifyApollo = require('@as-integrations/fastify');
const cors = require('@fastify/cors');
const helmet = require('@fastify/helmet');
const pino = require('pino');

const logger = pino({ level: process.env.LOG_LEVEL || 'info' });

const subgraphs = [
  { name: 'api', url: process.env.API_URL || 'http://api:8000/graphql' },
  { name: 'portfolio', url: process.env.PORTFOLIO_URL || 'http://portfolio:8000/graphql' },
  { name: 'neural-pricing', url: process.env.PRICING_URL || 'http://neural-pricing:8000/graphql' }
];

const gateway = new ApolloGateway({
  supergraphSdl: new IntrospectAndCompose({
    subgraphs,
    pollIntervalInMs: 10000
  }),
  debug: process.env.DEBUG === 'true'
});

const server = new ApolloServer({
  gateway,
  logger
});

const app = Fastify();

async function start() {
  await app.register(helmet, { contentSecurityPolicy: false });
  await app.register(cors);
  
  await server.start();
  
  // High-performance Apollo integration
  await app.register(fastifyApollo(server), {
    path: '/graphql'
  });

  // Health check
  app.get('/health', async () => ({ status: 'gateway_operational' }));

  const port = process.env.PORT || 4000;
  await app.listen({ port, host: '0.0.0.0' });
  logger.info(` God Mode Federated Gateway ready at http://localhost:${port}/graphql`);
}

start().catch(err => {
  logger.error(err);
  process.exit(1);
});
