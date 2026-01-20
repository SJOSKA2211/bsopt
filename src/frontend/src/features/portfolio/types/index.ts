export interface Position {
  contract_symbol: string;
  quantity: number;
  current_pnl: number;
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
