// frontend/src/types/websocket.d.ts

// Assuming UserStatsResponse structure from backend
export interface UserStats {
  total_requests: number;
  requests_today: number;
  requests_this_month: number;
  rate_limit_remaining: number;
  rate_limit_reset: string; // ISO format string
}

// Define types for expected WebSocket messages
export interface PriceUpdatePayload {
  instrument: {
    id: string;
    params: {
      spot: number;
      strike: number;
      time_to_expiry: number;
      volatility: number;
      rate: number;
      dividend: number;
      option_type: string;
    };
  };
  price: number;
  greeks: {
    delta: number;
    gamma: number;
    theta: number;
    vega: number;
    rho: number;
  };
  calculated_at: string;
  computation_time_ms: number;
  source: string;
}

export interface UserProfileUpdatePayload {
  id: string;
  email: string;
  full_name: string;
  tier: string;
  is_active: boolean;
  is_verified: boolean;
  is_mfa_enabled: boolean;
  created_at: string;
  last_login: string | null;
}

export interface UserStatusUpdatePayload {
  user_id: string;
  is_active: boolean;
  status: string;
}

export interface SystemStatsPayload {
  cpu_usage: number;
  memory_usage: number;
  active_connections: number;
}

export interface ClientInteractionPayload {
  action: string;
  timestamp: string;
}

// Define a union type for all possible WebSocket message payloads
export type WebSocketMessagePayload =
  | { type: 'user_stats_update'; payload: UserStats }
  | { type: 'price_update'; payload: PriceUpdatePayload }
  | { type: 'batch_price_update'; payload: PriceUpdatePayload[] } // Assuming batch is an array of PriceUpdatePayload
  | { type: 'system_stats_update'; payload: SystemStatsPayload }
  | { type: 'user_profile_update'; payload: UserProfileUpdatePayload }
  | { type: 'user_status_update'; payload: UserStatusUpdatePayload }
  | { type: 'client_interaction'; payload: ClientInteractionPayload }
  | { type: string; payload: unknown }; // Fallback for unknown message types
