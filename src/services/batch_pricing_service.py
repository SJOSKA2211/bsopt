import sys
import pandas as pd
from pathlib import Path
from typing import Callable, Optional, List, Dict
import structlog
from src.pricing.black_scholes import BlackScholesEngine, BSParameters
from src.utils.filesystem import sanitize_path

logger = structlog.get_logger(__name__)

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Callable, Optional, List, Dict
import structlog
import time
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
        Process a batch of options from a CSV file using vectorized engines.
        """
        start_time = time.perf_counter()
        
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

        # Vectorized Processing for Black-Scholes
        if method.lower() in ('bs', 'black_scholes'):
            # Extract arrays
            spots = df['spot'].values
            strikes = df['strike'].values
            maturities = df['maturity'].values
            vols = df['volatility'].values
            rates = df['rate'].values
            divs = df['dividend'].values
            types = df['option_type'].values

            # Pre-allocate output buffer for zero-allocation JIT
            out_prices = np.empty(len(df), dtype=np.float64)
            
            # Parallel Vectorized Pricing
            BlackScholesEngine.price_options(
                spot=spots, strike=strikes, maturity=maturities, 
                volatility=vols, rate=rates, dividend=divs, 
                option_type=types, out=out_prices
            )
            
            df['price'] = out_prices
            df['method'] = 'black_scholes'
            
            if compute_greeks:
                # Pre-allocate Greek buffers
                out_delta = np.empty(len(df), dtype=np.float64)
                out_gamma = np.empty(len(df), dtype=np.float64)
                out_vega = np.empty(len(df), dtype=np.float64)
                out_theta = np.empty(len(df), dtype=np.float64)
                out_rho = np.empty(len(df), dtype=np.float64)
                
                # Parallel Vectorized Greeks
                BlackScholesEngine.calculate_greeks_batch(
                    spot=spots, strike=strikes, maturity=maturities, 
                    volatility=vols, rate=rates, dividend=divs, 
                    option_type=types,
                    out_delta=out_delta, out_gamma=out_gamma, out_vega=out_vega,
                    out_theta=out_theta, out_rho=out_rho
                )
                
                df['delta'] = out_delta
                df['gamma'] = out_gamma
                df['vega'] = out_vega
                df['theta'] = out_theta
                df['rho'] = out_rho

            if progress_advance:
                progress_advance(len(df))
        else:
            # Fallback for non-vectorized methods (Monte Carlo, etc.)
            # This remains row-by-row until those engines support batching
            results = []
            engine = BlackScholesEngine() # Placeholder engine for factory-like logic
            # In a real setup, we'd use PricingEngineFactory.get_strategy(method)
            
            for idx, row in df.iterrows():
                # ... (rest of sequential logic if needed)
                pass
            raise NotImplementedError(f"Vectorized batching not yet implemented for {method}")

        # Finalize computation time
        duration_ms = (time.perf_counter() - start_time) * 1000
        df['computation_time_ms'] = duration_ms / len(df) # Average per row

        # Save results
        safe_output_file = sanitize_path(Path.cwd(), str(output_file))
        df.to_csv(safe_output_file, index=False)
        
        logger.info("batch_processing_complete", 
                    count=len(df), 
                    total_time_ms=duration_ms, 
                    method=method)
        
        return df.to_dict('records')
