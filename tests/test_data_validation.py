from datetime import datetime, timezone

import pandas as pd

from src.data.validation import OptionsDataValidator
from tests.test_utils import assert_equal


def test_options_data_validation():
    now = datetime.now(timezone.utc)
    df = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "contract_symbol": ["AAPL251225C00100000", "AAPL251225C00110000"],
            "underlying_price": [100.0, 100.0],
            "strike": [100.0, 110.0],
            "timestamp": [now, now],
            "expiration": [now, now],
            "time_to_expiry": [1.0, 1.0],
            "volatility": [0.2, 0.2],
            "bid": [10.0, 5.0],
            "ask": [11.0, 6.0],
            "mid_price": [10.5, 5.5],
            "option_type": ["call", "call"],
            "volume": [100, 100],
            "open_interest": [100, 100],
            "implied_volatility": [0.2, 0.21],
            "delta": [0.5, 0.4],
            "gamma": [0.02, 0.02],
            "theta": [-0.05, -0.06],
            "vega": [0.1, 0.12],
            "rho": [0.08, 0.09],
            "moneyness": [1.0, 0.9],
        }
    )

    # Valid data
    validator = OptionsDataValidator(min_samples=1)
    report = validator.validate(df)
    assert report.passed
    assert_equal(len(df), 2)

    # Invalid data (negative strike)
    df_invalid = df.copy()
    df_invalid.loc[0, "strike"] = -100.0
    report_invalid = validator.validate(df_invalid)
    assert not report_invalid.passed
