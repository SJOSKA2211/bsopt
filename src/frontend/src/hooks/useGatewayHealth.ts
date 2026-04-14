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
        const [apiRes, authRes] = await Promise.all([
          axios.get('/api/v1/health', { timeout: 2000 }).catch(e => ({ status: 500, data: {} })),
          axios.get('/api/auth/health', { timeout: 2000 }).catch(e => ({ status: 500, data: {} }))
        ]);
        
        const end = performance.now();
        const latency = end - start;

        const isHealthy = apiRes.status === 200 && authRes.status === 200;

        setHealth({
          status: isHealthy ? 'healthy' : 'degraded',
          latency: Math.round(latency),
          services: {
            api: apiRes.status === 200,
            auth: authRes.status === 200,
            ingestion: apiRes.data?.ingestion === 'active' || true
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
