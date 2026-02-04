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
    async def process_batch(
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
        Benefits from ProcessPoolExecutor and SharedMemory for large batches.
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

        pricing_service = PricingService()
        
        # ZERO-ALLOCATION PATH: Extract arrays directly (Task 4)
        spots = df['spot'].to_numpy(dtype=np.float64)
        strikes = df['strike'].to_numpy(dtype=np.float64)
        maturities = df['maturity'].to_numpy(dtype=np.float64)
        vols = df['volatility'].to_numpy(dtype=np.float64)
        rates = df['rate'].to_numpy(dtype=np.float64)
        dividends = df['dividend'].to_numpy(dtype=np.float64)
        option_types = df['option_type'].to_numpy(dtype=object)
        symbols = df['symbol'].to_numpy(dtype=object)
        models = np.full(len(df), 'black_scholes' if method.lower() in ('bs', 'black_scholes') else method.lower(), dtype=object)

        # Call optimized array-based pricing
        if method.lower() in ('bs', 'black_scholes', 'heston') and len(df) > 1000:
            # ULTIMATE ZERO-ALLOCATION PATH (Task 4)
            n = len(df)
            shm_in_name = shm_manager.acquire()
            shm_out_name = shm_manager.acquire()
            
            if shm_in_name and shm_out_name:
                try:
                    shm_in = shm_manager.get_segment(shm_in_name)
                    shm_out = shm_manager.get_segment(shm_out_name)
                    
                    # Columns for SHM layout
                    cols = ['spot', 'strike', 'maturity', 'volatility', 'rate', 'dividend']
                    if method.lower() == 'heston':
                        # Heston expects 10 columns: [spot, strike, T, r, v0, kappa, theta, sigma, rho, is_call]
                        # This is a bit more complex as we need parameters. 
                        # Fallback to price_batch_arrays for Heston for now.
                        out_prices = await pricing_service.price_batch_arrays(
                            spots, strikes, maturities, vols, rates, dividends, option_types, models, symbols
                        )
                    else:
                        # BS: [spot, strike, T, vol, r, q, is_call] (7 columns)
                        input_shm = np.ndarray((n, 7), dtype=np.float64, buffer=shm_in.buf)
                        for i, col in enumerate(cols):
                            input_shm[:, i] = df[col].to_numpy()
                        input_shm[:, 6] = (df['option_type'].str.lower() == 'call').astype(np.float64)
                        
                        success = await pricing_service.price_batch_shm(
                            shm_in_name, shm_out_name, (n, 7), model=method.lower()
                        )
                        
                        if success is True:
                            out_prices = np.ndarray((n,), dtype=np.float64, buffer=shm_out.buf).copy()
                        else:
                            out_prices = await pricing_service.price_batch_arrays(
                                spots, strikes, maturities, vols, rates, dividends, option_types, models, symbols
                            )
                finally:
                    shm_manager.release(shm_in_name)
                    shm_manager.release(shm_out_name)
            else:
                out_prices = await pricing_service.price_batch_arrays(
                    spots, strikes, maturities, vols, rates, dividends, option_types, models, symbols
                )
        else:
            out_prices = await pricing_service.price_batch_arrays(
                spots, strikes, maturities, vols, rates, dividends, option_types, models, symbols
            )
        
        df['price'] = out_prices
        df['method'] = method.lower()
        
        if compute_greeks:
            # Fallback to standard greeks for now, or implement calculate_greeks_batch_arrays
            from src.api.schemas.pricing import PriceRequest
            options_data = []
            for i in range(len(df)):
                options_data.append(PriceRequest(
                    spot=spots[i], strike=strikes[i], time_to_expiry=maturities[i],
                    volatility=vols[i], rate=rates[i], dividend_yield=dividends[i],
                    option_type=option_types[i], model=models[i], symbol=symbols[i]
                ))
            greeks_response = await pricing_service.calculate_greeks_batch(options_data)
            df['delta'] = [r.delta for r in greeks_response.results]
            df['gamma'] = [r.gamma for r in greeks_response.results]
            df['vega'] = [r.vega for r in greeks_response.results]
            df['theta'] = [r.theta for r in greeks_response.results]
            df['rho'] = [r.rho for r in greeks_response.results]

        if progress_advance:
            progress_advance(len(df))

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
