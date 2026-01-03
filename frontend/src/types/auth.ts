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
