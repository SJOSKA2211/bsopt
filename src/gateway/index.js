const { ApolloServer } = require('@apollo/server');
const { expressMiddleware } = require('@apollo/server/express4');
const { ApolloGateway, IntrospectAndCompose, RemoteGraphQLDataSource } = require('@apollo/gateway');
const { createServer } = require('http');
const { WebSocketServer } = require('ws');
const { useServer } = require('graphql-ws/use/ws');
const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const compression = require('compression');
const helmet = require('helmet');
const responseTime = require('response-time');
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
    if (context.headers && context.headers['x-user-role']) {
      request.http.headers.set('X-User-Role', context.headers['x-user-role']);
    }
  }
}

const gateway = new ApolloGateway({
  supergraphSdl: new IntrospectAndCompose({
    subgraphs: [
      { name: 'options', url: process.env.OPTIONS_URL || 'http://localhost:8000/graphql' },
      { name: 'pricing', url: process.env.PRICING_URL || 'http://localhost:8001/graphql' },
      { name: 'ml', url: process.env.ML_URL || 'http://localhost:8002/graphql' },
      { name: 'portfolio', url: process.env.PORTFOLIO_URL || 'http://localhost:8003/graphql' },
      { name: 'marketdata', url: process.env.MARKETDATA_URL || 'http://localhost:8004/graphql' },
    ],
  }),
  buildService({ url }) {
    return new AuthenticatedDataSource({ url });
  },
});

async function startServer() {
  const app = express();
  const httpServer = createServer(app);

  // Middlewares for performance and security
  app.use(helmet({ contentSecurityPolicy: false })); // CSP disabled for GraphQL Playground if needed
  app.use(compression());
  app.use(responseTime((req, res, time) => {
    logger.info({
      method: req.method,
      url: req.url,
      status: res.statusCode,
      duration: `${time.toFixed(2)}ms`
    }, 'request_completed');
  }));

  await gateway.load();

  const server = new ApolloServer({
    gateway,
    plugins: [
      responseCachePlugin(),
      {
        async serverWillStart() {
          return {
            async drainServer() {
              await serverCleanup.dispose();
            },
          };
        },
      },
    ],
  });

  await server.start();

  app.use(
    '/',
    cors(),
    bodyParser.json({ limit: '1mb' }),
    expressMiddleware(server, {
      context: async ({ req }) => ({ headers: req.headers }),
    }),
  );

  const wsServer = new WebSocketServer({
    server: httpServer,
    path: '/',
  });

  const serverCleanup = useServer({ schema: gateway.schema }, wsServer);

  const port = process.env.PORT || 4000;
  httpServer.listen(port, () => {
    logger.info(`🚀 Gateway worker ${process.pid} ready at http://localhost:${port}/`);
  });
}

if (process.env.NODE_ENV === 'production' && cluster.isPrimary) {
  const numCPUs = os.cpus().length;
  logger.info(`Primary ${process.pid} is running. Forking ${numCPUs} workers...`);

  for (let i = 0; i < numCPUs; i++) {
    cluster.fork();
  }

  cluster.on('exit', (worker, code, signal) => {
    logger.warn(`Worker ${worker.process.pid} died. Forking a new one...`);
    cluster.fork();
  });
} else {
  startServer().catch(err => {
    logger.error(err, 'failed_to_start_server');
    process.exit(1);
  });
}