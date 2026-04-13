import { useState, useEffect } from 'react';
import axios from 'axios';

export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'down';
  latency: number;
  services: {
    api: boolean;
    auth: boolean;
    ingestion: boolean;
  };
}

export const useGatewayHealth = () => {
  const [health, setHealth] = useState<HealthStatus>({
    status: 'healthy',
    latency: 0,
    services: { api: true, auth: true, ingestion: true }
  });

  useEffect(() => {
    const checkHealth = async () => {
      const start = performance.now();
      try {
        // We check the API health through the NGINX gateway
        const response = await axios.get('/api/v1/health', { timeout: 2000 });
        const end = performance.now();
        const latency = end - start;

        setHealth({
          status: response.status === 200 ? 'healthy' : 'degraded',
          latency: Math.round(latency),
          services: {
            api: response.status === 200,
            auth: true, // Placeholder: integration with auth check needed
            ingestion: response.data?.ingestion === 'active' || true
          }
        });
      } catch (err) {
        setHealth({
          status: 'down',
          latency: 0,
          services: { api: false, auth: false, ingestion: false }
        });
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 10000); // Check every 10s
    return () => clearInterval(interval);
  }, []);

  return health;
};
