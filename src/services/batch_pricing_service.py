import sys
import pandas as pd
from pathlib import Path
from typing import Callable, Optional, List, Dict
import structlog
from src.pricing.black_scholes import BlackScholesEngine, BSParameters
from src.utils.filesystem import sanitize_path

logger = structlog.get_logger(__name__)

class BatchPricingService:
    def process_batch(
        self,
        input_file: Path,
        output_file: Path,
        method: str = 'bs',
        compute_greeks: bool = True,
        progress_setup: Optional[Callable[[int], None]] = None,
        progress_advance: Optional[Callable[[int], None]] = None
    ) -> List[Dict]:
        """
        Process a batch of options from a CSV file.
        """
        
        # Load input CSV
        safe_input_file = sanitize_path(Path.cwd(), input_file)
        df = pd.read_csv(safe_input_file)

        # Validate required columns
        required_cols = ['symbol', 'spot', 'strike', 'maturity', 'volatility',
                        'rate', 'dividend', 'option_type']
        missing_cols = set(required_cols) - set(df.columns)

        if missing_cols:
             raise ValueError(f"Missing columns: {missing_cols}")

        if progress_setup:
            progress_setup(len(df))

        # Process each option
        results = []
        engine = BlackScholesEngine()

        for idx, row in df.iterrows():
            try:
                params = BSParameters(
                    spot=row['spot'],
                    strike=row['strike'],
                    maturity=row['maturity'],
                    volatility=row['volatility'],
                    rate=row['rate'],
                    dividend=row['dividend']
                )

                # Currently only supporting BS in this service refactor for simplicity
                # If other methods (FDM, MC) are needed, they should be instantiated similarly
                if method == 'bs':
                    price = engine.price(params, row['option_type'])
                    greeks = None
                    if compute_greeks:
                        greeks = engine.calculate_greeks(params, row['option_type'])
                else:
                    # Fallback or raise for other methods if not yet migrated to service
                    # For now, we assume 'bs' as primary
                    price = engine.price(params, row['option_type'])
                    greeks = None if not compute_greeks else engine.calculate_greeks(params, row['option_type'])

                result_row = {
                    'symbol': row['symbol'],
                    'price': price,
                    'method': method,
                    'computation_time_ms': 0 
                }

                if greeks:
                    result_row.update({
                        'delta': greeks.delta,
                        'gamma': greeks.gamma,
                        'vega': greeks.vega,
                        'theta': greeks.theta,
                        'rho': greeks.rho
                    })

                results.append(result_row)

            except Exception as e:
                logger.warning("pricing_failed", symbol=row['symbol'], error=str(e))
                results.append({
                    'symbol': row['symbol'],
                    'price': None,
                    'error': str(e)
                })

            if progress_advance:
                progress_advance(1)

        # Save results
        results_df = pd.DataFrame(results)
        safe_output_file = sanitize_path(Path.cwd(), str(output_file))
        results_df.to_csv(safe_output_file, index=False)
        
        return results
