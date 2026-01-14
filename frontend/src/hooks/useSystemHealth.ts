import { useState, useEffect } from 'react';
import { getHealth, HealthResponse } from '@/api/endpoints/health';

export const useSystemHealth = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isOnline, setIsOnline] = useState<boolean>(true);

  const checkHealth = async () => {
    try {
      const data = await getHealth();
      setHealth(data);
      setIsOnline(data.status === 'healthy' || data.status === 'degraded');
    } catch (err) {
      setHealth(null);
      setIsOnline(false);
    }
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 30000); // Check every 30s
    return () => clearInterval(interval);
  }, []);

  return { health, isOnline, checkHealth };
};
