export interface Position {
  contract_symbol: string;
  quantity: number;
  current_pnl: number;
  expiry?: string;
  underlying_price?: number;
  strike?: number;
  implied_volatility?: number;
  option_type?: 'call' | 'put';
  theor_greeks?: {
    delta: number;
    gamma: number;
    vega: number;
    theta: number;
    rho: number;
  };
}

export interface PortfolioData {
  balance: number;
  frozen_capital: number;
  risk_score: number;
  positions: Position[];
  totalValue: number;
  dailyPnL: number;
  dailyPnLPercent: number;
  positionsCount: number;
}
