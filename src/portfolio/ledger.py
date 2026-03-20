from datetime import datetime

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class PortfolioLedger:
    """
    Institutional Portfolio Ledger.
    Tracks historical trades, cash flows, and realized/unrealized P&L.
    """

    def __init__(self):
        self.trades = []
        self.cash_balance = 0.0

    def record_trade(self, symbol: str, quantity: float, price: float, side: str):
        """Record a single entry in the ledger."""
        timestamp = datetime.now().isoformat()
        entry = {
            "timestamp": timestamp,
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "side": side,  # 'buy' or 'sell'
            "status": "finalized",
        }
        self.trades.append(entry)

        # Update cash (simplified)
        cost = quantity * price
        if side == "buy":
            self.cash_balance -= cost
        else:
            self.cash_balance += cost

        logger.info("trade_recorded", **entry)

    def get_holdings(self) -> pd.DataFrame:
        """Calculate current net positions from ledger."""
        if not self.trades:
            return pd.DataFrame()
        df = pd.DataFrame(self.trades)
        df["net_qty"] = df.apply(
            lambda x: x["quantity"] if x["side"] == "buy" else -x["quantity"], axis=1
        )
        return df.groupby("symbol")["net_qty"].sum().reset_index()
