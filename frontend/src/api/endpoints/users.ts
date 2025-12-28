import apiClient from '@/utils/apiClient';

/**
 * User Types
 */

export interface UserResponse {
  id: string;
  email: string;
  full_name: string | null;
  tier: string;
  is_active: boolean;
  is_verified: boolean;
  is_mfa_enabled: boolean;
  created_at: string;
  last_login: string | null;
}

export interface UserUpdateRequest {
  full_name?: string;
  email?: string;
}

export interface UserStatsResponse {
  total_requests: number;
  requests_today: number;
  requests_this_month: number;
  rate_limit_remaining: number;
  rate_limit_reset: string;
}

export interface SuccessResponse {
  message: string;
}

/**
 * User API Endpoints
 */

export const getCurrentUser = async (): Promise<UserResponse> => {
  const response = await apiClient.get<UserResponse>('/users/me');
  return response.data;
};

export const updateCurrentUser = async (data: UserUpdateRequest): Promise<UserResponse> => {
  const response = await apiClient.patch<UserResponse>('/users/me', data);
  return response.data;
};

export const getUserStats = async (): Promise<UserStatsResponse> => {
  const response = await apiClient.get<UserStatsResponse>('/users/me/stats');
  return response.data;
};

export const deleteCurrentUserAccount = async (): Promise<SuccessResponse> => {
  const response = await apiClient.delete<SuccessResponse>('/users/me');
  return response.data;
};

export const getUserById = async (userId: string): Promise<UserResponse> => {
  const response = await apiClient.get<UserResponse>(`/users/${userId}`);
  return response.data;
};
