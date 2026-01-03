import apiClient from '@/utils/apiClient';
import { DataResponse } from '@/types/common';

export interface HealthResponse {
  status: string;
  version: string;
  timestamp: string;
  checks: Record<string, { status: string; latency_ms?: number; error?: string }>;
}

export const getHealth = async (): Promise<HealthResponse> => {
  const response = await apiClient.get<DataResponse<HealthResponse>>('/health');
  return response.data.data;
};
