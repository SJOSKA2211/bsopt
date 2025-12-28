import apiClient from '@/utils/apiClient';

/**
 * Authentication Types
 */

export interface LoginRequest {
  email: string;
  password: string;
  remember_me?: boolean;
  mfa_code?: string;
}

export interface LoginResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
  user_id: string;
  email: string;
  tier: string;
  requires_mfa: boolean;
}

export interface RegisterRequest {
  email: string;
  password: string;
  password_confirm: string;
  full_name?: string;
  accept_terms: boolean;
}

export interface RegisterResponse {
  user_id: string;
  email: string;
  message: string;
  verification_required: boolean;
}

export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export interface RefreshTokenRequest {
  refresh_token: string;
}

export interface PasswordResetRequest {
  email: string;
}

export interface PasswordResetConfirm {
  token: string;
  new_password: string;
  new_password_confirm: string;
}

export interface PasswordChangeRequest {
  current_password: string;
  new_password: string;
  new_password_confirm: string;
}

export interface MFASetupResponse {
  secret: string;
  qr_code_uri: string;
  backup_codes: string[];
}

export interface MFAVerifyRequest {
  code: string;
}

export interface EmailVerificationRequest {
  token: string;
}

export interface SuccessResponse {
  message: string;
}

/**
 * Authentication API Endpoints
 */

export const login = async (data: LoginRequest): Promise<LoginResponse> => {
  const response = await apiClient.post<LoginResponse>('/auth/login', data);
  return response.data;
};

export const register = async (data: RegisterRequest): Promise<RegisterResponse> => {
  const response = await apiClient.post<RegisterResponse>('/auth/register', data);
  return response.data;
};

export const logout = async (): Promise<SuccessResponse> => {
  const response = await apiClient.post<SuccessResponse>('/auth/logout');
  return response.data;
};

export const refreshToken = async (data: RefreshTokenRequest): Promise<TokenResponse> => {
  const response = await apiClient.post<TokenResponse>('/auth/refresh', data);
  return response.data;
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
  const response = await apiClient.post<MFASetupResponse>('/auth/mfa/setup');
  return response.data;
};

export const verifyMFA = async (data: MFAVerifyRequest): Promise<SuccessResponse> => {
  const response = await apiClient.post<SuccessResponse>('/auth/mfa/verify', data);
  return response.data;
};

export const disableMFA = async (data: MFAVerifyRequest): Promise<SuccessResponse> => {
  const response = await apiClient.post<SuccessResponse>('/auth/mfa/disable', data);
  return response.data;
};
