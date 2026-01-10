const { ApolloServer } = require('@apollo/server');
const { startStandaloneServer } = require('@apollo/server/standalone');
const { ApolloGateway, IntrospectAndCompose } = require('@apollo/gateway');

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
});

const server = new ApolloServer({
  gateway,
});

const port = process.env.PORT || 4000;

startStandaloneServer(server, {
  listen: { port: Number(port) },
}).then(({ url }) => {
  console.log(`🚀  Gateway ready at ${url}`);
}).catch(err => {
  console.error(err);
});
