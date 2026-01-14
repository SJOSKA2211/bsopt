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

export interface APIKeyCreateRequest {
  name: string;
}

export interface APIKeyResponse {
  id: string;
  name: string;
  prefix: string;
  created_at: string;
  last_used_at: string | null;
  raw_key?: string;
}
