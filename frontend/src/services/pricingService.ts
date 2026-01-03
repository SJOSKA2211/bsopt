import apiClient from '../utils/apiClient';
import { 
    PriceRequest, 
    PriceResponse, 
    GreeksRequest, 
    GreeksResponse, 
    ImpliedVolatilityRequest, 
    ImpliedVolatilityResponse, 
    ExoticPriceRequest, 
    ExoticPriceResponse, 
    BatchPriceRequest, 
    BatchPriceResponse 
} from '../types/pricing';
import { DataResponse } from '../types/common';

export const getOptionPrice = async (request: PriceRequest): Promise<PriceResponse> => {
    // Note: The backend returns the object directly, not wrapped in 'data' usually for this setup, 
    // but the original service code expected DataResponse wrapper.
    // Based on endpoints/pricing.ts, it seems the backend returns the object directly.
    // I will adjust to match the likely actual backend behavior (direct return) 
    // BUT if the backend wraps it in a standard response, we need to handle that.
    // Looking at the endpoint file, it returns response.data directly.
    const response = await apiClient.post<PriceResponse>('/pricing/price', request);
    return response.data;
};

export const getBatchOptionPrices = async (request: BatchPriceRequest): Promise<BatchPriceResponse> => {
    const response = await apiClient.post<BatchPriceResponse>('/pricing/batch', request);
    return response.data;
};

export const getGreeks = async (request: GreeksRequest): Promise<GreeksResponse> => {
    const response = await apiClient.post<GreeksResponse>('/pricing/greeks', request);
    return response.data;
};

export const getImpliedVolatility = async (request: ImpliedVolatilityRequest): Promise<ImpliedVolatilityResponse> => {
    const response = await apiClient.post<ImpliedVolatilityResponse>('/pricing/implied-volatility', request);
    return response.data;
};

export const getExoticOptionPrice = async (request: ExoticPriceRequest): Promise<ExoticPriceResponse> => {
    const response = await apiClient.post<ExoticPriceResponse>('/pricing/exotic', request);
    return response.data;
};