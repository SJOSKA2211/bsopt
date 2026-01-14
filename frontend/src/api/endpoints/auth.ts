import apiClient from '@/utils/apiClient';
import {
  LoginRequest,
  LoginResponse,
  RegisterRequest,
  RegisterResponse,
  TokenResponse,
  RefreshTokenRequest,
  PasswordResetRequest,
  PasswordResetConfirm,
  PasswordChangeRequest,
  MFASetupResponse,
  MFAVerifyRequest,
  EmailVerificationRequest,
} from '@/types/auth';
import { SuccessResponse, DataResponse } from '@/types/common';

/**
 * Authentication API Endpoints
 */

export const login = async (data: LoginRequest): Promise<LoginResponse> => {
  const response = await apiClient.post<DataResponse<LoginResponse>>('/auth/login', data);
  return response.data.data;
};

export const register = async (data: RegisterRequest): Promise<RegisterResponse> => {
  const response = await apiClient.post<DataResponse<RegisterResponse>>('/auth/register', data);
  return response.data.data;
};

export const logout = async (): Promise<SuccessResponse> => {
  const response = await apiClient.post<SuccessResponse>('/auth/logout');
  return response.data;
};

export const refreshToken = async (data: RefreshTokenRequest): Promise<TokenResponse> => {
  const response = await apiClient.post<DataResponse<TokenResponse>>('/auth/refresh', data);
  return response.data.data;
};

export const verifyEmail = async (data: EmailVerificationRequest): Promise<SuccessResponse> => {
  const response = await apiClient.post<SuccessResponse>('/auth/verify-email', data);
  return response.data;
};

export const requestPasswordReset = async (data: PasswordResetRequest): Promise<SuccessResponse> => {
  const response = await apiClient.post<SuccessResponse>('/auth/password-reset', data);
  return response.data;
};

export const confirmPasswordReset = async (data: PasswordResetConfirm): Promise<SuccessResponse> => {
  const response = await apiClient.post<SuccessResponse>('/auth/password-reset/confirm', data);
  return response.data;
};

export const changePassword = async (data: PasswordChangeRequest): Promise<SuccessResponse> => {
  const response = await apiClient.post<SuccessResponse>('/auth/password-change', data);
  return response.data;
};

export const setupMFA = async (): Promise<MFASetupResponse> => {
  const response = await apiClient.post<DataResponse<MFASetupResponse>>('/auth/mfa/setup');
  return response.data.data;
};

export const verifyMFA = async (data: MFAVerifyRequest): Promise<SuccessResponse> => {
  const response = await apiClient.post<SuccessResponse>('/auth/mfa/verify', data);
  return response.data;
};

export const disableMFA = async (data: MFAVerifyRequest): Promise<SuccessResponse> => {
  const response = await apiClient.post<SuccessResponse>('/auth/mfa/disable', data);
  return response.data;
};
