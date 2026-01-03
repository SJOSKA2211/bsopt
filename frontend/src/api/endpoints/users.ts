import apiClient from '@/utils/apiClient';
import { UserResponse, UserUpdateRequest, UserStatsResponse, APIKeyCreateRequest, APIKeyResponse } from '@/types/user';
import { DataResponse, SuccessResponse } from '@/types/common';

/**
 * User API Endpoints
 */

export const getCurrentUser = async (): Promise<UserResponse> => {
  const response = await apiClient.get<DataResponse<UserResponse>>('/users/me');
  return response.data.data;
};

export const updateCurrentUser = async (data: UserUpdateRequest): Promise<UserResponse> => {
  const response = await apiClient.patch<DataResponse<UserResponse>>('/users/me', data);
  return response.data.data;
};

export const getUserStats = async (): Promise<UserStatsResponse> => {
  const response = await apiClient.get<DataResponse<UserStatsResponse>>('/users/me/stats');
  return response.data.data;
};

export const listAPIKeys = async (): Promise<APIKeyResponse[]> => {
  const response = await apiClient.get<DataResponse<APIKeyResponse[]>>('/users/me/keys');
  return response.data.data;
};

export const createAPIKey = async (data: APIKeyCreateRequest): Promise<APIKeyResponse> => {
  const response = await apiClient.post<DataResponse<APIKeyResponse>>('/users/me/keys', data);
  return response.data.data;
};

export const revokeAPIKey = async (keyId: string): Promise<SuccessResponse> => {
  const response = await apiClient.delete<SuccessResponse>(`/users/me/keys/${keyId}`);
  return response.data;
};

export const deleteAccount = async (): Promise<SuccessResponse> => {
  const response = await apiClient.delete<SuccessResponse>('/users/me');
  return response.data;
};