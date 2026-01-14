import apiClient from '@/utils/apiClient';
import {
  PriceRequest,
  PriceResponse,
  BatchPriceRequest,
  BatchPriceResponse,
  GreeksRequest,
  GreeksResponse,
  ImpliedVolatilityRequest,
  ImpliedVolatilityResponse,
} from '@/types/pricing';
import { DataResponse } from '@/types/common';

/**
 * Pricing API Endpoints
 */

export const priceOption = async (data: PriceRequest): Promise<PriceResponse> => {
  const response = await apiClient.post<DataResponse<PriceResponse>>('/pricing/price', data);
  return response.data.data;
};

export const batchPriceOptions = async (data: BatchPriceRequest): Promise<BatchPriceResponse> => {
  const response = await apiClient.post<DataResponse<BatchPriceResponse>>('/pricing/batch', data);
  return response.data.data;
};

export const calculateGreeks = async (data: GreeksRequest): Promise<GreeksResponse> => {
  const response = await apiClient.post<DataResponse<GreeksResponse>>('/pricing/greeks', data);
  return response.data.data;
};

export const calculateIV = async (data: ImpliedVolatilityRequest): Promise<ImpliedVolatilityResponse> => {
  const response = await apiClient.post<DataResponse<ImpliedVolatilityResponse>>('/pricing/implied-volatility', data);
  return response.data.data;
};
