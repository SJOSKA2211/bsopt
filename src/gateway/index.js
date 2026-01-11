const { ApolloServer } = require('@apollo/server');
const { expressMiddleware } = require('@apollo/server/express4');
const { ApolloGateway, IntrospectAndCompose, RemoteGraphQLDataSource } = require('@apollo/gateway');
const { createServer } = require('http');
const { WebSocketServer } = require('ws');
const { useServer } = require('graphql-ws/use/ws');
const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const responseCachePlugin = require('@apollo/server-plugin-response-cache').default;

class AuthenticatedDataSource extends RemoteGraphQLDataSource {
  willSendRequest({ request, context }) {
    if (context.headers && context.headers['authorization']) {
      request.http.headers.set('authorization', context.headers['authorization']);
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
  buildService({ name, url }) {
    return new AuthenticatedDataSource({ url });
  },
});

const app = express();
const httpServer = createServer(app);

// Gateway startup
gateway.load().then(async () => {
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
    bodyParser.json(),
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
    console.log(`🚀 Gateway ready at http://localhost:${port}/`);
  });
});

