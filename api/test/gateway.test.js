const { ApolloGateway, IntrospectAndCompose } = require('@apollo/gateway');

// Mock dependencies
jest.mock('@apollo/gateway', () => {
  const mGateway = {
    load: jest.fn().mockResolvedValue({}),
    schema: {}
  };
  return {
    ApolloGateway: jest.fn(() => mGateway),
    IntrospectAndCompose: jest.fn(),
    RemoteGraphQLDataSource: class MockRemoteGraphQLDataSource {}
  };
});

jest.mock('@apollo/server', () => ({
  ApolloServer: jest.fn(() => ({
    start: jest.fn().mockResolvedValue({}),
  }))
}));

jest.mock('@apollo/server/express4', () => ({
    expressMiddleware: jest.fn()
}));

jest.mock('graphql-ws/use/ws', () => ({
    useServer: jest.fn()
}));

jest.mock('ws', () => ({
    WebSocketServer: jest.fn(() => ({
        on: jest.fn()
    }))
}));

jest.mock('@apollo/server-plugin-response-cache', () => ({
    default: jest.fn()
}));

describe('Gateway Configuration', () => {
  let originalEnv;

  beforeAll(() => {
    originalEnv = process.env;
    process.env = { ...originalEnv, 
        OPTIONS_URL: 'http://test-options:8000',
        PRICING_URL: 'http://test-pricing:8001',
        ML_URL: 'http://test-ml:8002',
        PORTFOLIO_URL: 'http://test-portfolio:8003',
        MARKETDATA_URL: 'http://test-marketdata:8004'
    };
  });

  afterAll(() => {
    process.env = originalEnv;
  });

  test('should configure gateway with correct subgraphs', () => {
    // We need to require the file to trigger execution
    // But index.js starts the server immediately which might fail due to mocks or network
    // So we should probably export the config or logic to test it.
    // However, given the file structure, we can just load it and check the mocks.
    
    // To prevent actual server startup issues during require, we might need to mock express/http too
    jest.mock('express', () => {
        const mApp = { use: jest.fn() };
        return jest.fn(() => mApp);
    });
    jest.mock('http', () => ({
        createServer: jest.fn(() => ({ listen: jest.fn() }))
    }));

    require('../index');

    expect(ApolloGateway).toHaveBeenCalled();
    expect(IntrospectAndCompose).toHaveBeenCalledWith({
      subgraphs: [
        { name: 'options', url: 'http://test-options:8000' },
        { name: 'pricing', url: 'http://test-pricing:8001' },
        { name: 'ml', url: 'http://test-ml:8002' },
        { name: 'portfolio', url: 'http://test-portfolio:8003' },
        { name: 'marketdata', url: 'http://test-marketdata:8004' },
      ],
    });
  });
});
