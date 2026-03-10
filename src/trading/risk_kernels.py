import numpy as np
import structlog
from numba import njit

try:
    import bsopt_core

    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False

logger = structlog.get_logger(__name__)


@njit(cache=True, fastmath=True)
def _validate_order_kernel(
    price: float,
    quantity: int,
    side: int,
    max_qty: int = 1000,
    min_price: float = 0.01,
    max_price: float = 10000.0,
) -> int:
    """
    Sub-microsecond silicon risk check.
    Returns: 1 if OK, 0 if VETO.
    """
    # 1. Fat-finger Price Protection
    if price < min_price or price > max_price:
        return 0

    # 2. Quantity Protection
    if quantity <= 0 or quantity > max_qty:
        return 0

    # 3. Side Integrity (Must be 1 or -1)
    if side != 1 and side != -1:
        return 0

    return 1


@njit(cache=True, fastmath=True)
def _validate_delta_kernel(
    state_arr: np.ndarray, trade_delta: float, max_net_delta: float = 10000.0
) -> int:
    """
    O(1) incremental delta check.
    Returns: 1 if OK else 0.
    """
    new_net_delta = state_arr[0] + trade_delta
    if abs(new_net_delta) > max_net_delta:
        return 0

    # Commit state
    state_arr[0] = new_net_delta
    return 1


@njit(cache=True, fastmath=True)
def _full_risk_check_kernel(
    price: float,
    quantity: int,
    side: int,
    trade_delta: float,
    state_arr: np.ndarray,
    max_qty: int = 1000,
    min_price: float = 0.01,
    max_price: float = 10000.0,
    max_net_delta: float = 10000.0,
) -> int:
    """
    Combined God-Tier Risk Kernel: Sub-300ns execution.
    Handles fat-finger protection and incremental delta validation in one pass.
    """
    # 1. Base Silicon Checks
    if (
        price < min_price
        or price > max_price
        or quantity <= 0
        or quantity > max_qty
        or (side != 1 and side != -1)
    ):
        return 0

    # 2. Incremental Delta Check
    new_net_delta = state_arr[0] + trade_delta
    if abs(new_net_delta) > max_net_delta:
        return 0

    # 3. State Commit
    state_arr[0] = new_net_delta
    return 1


class RiskVectorTracker:
    """
    Multi-Dimensional stateful tracker for portfolio-wide risk exposure.
    OPTIMIZED: Maintains O(1) running totals for Delta, Gamma, and Vega.
    """

    def __init__(self, initial_greeks: np.ndarray | None = None, limits: np.ndarray | None = None):
        # state: [delta, gamma, vega]
        if initial_greeks is None:
            initial_greeks = np.zeros(3, dtype=np.float64)
        self._state = initial_greeks

        # limits: [max_delta, max_gamma, max_vega]
        if limits is None:
            limits = np.array([10000.0, 5000.0, 5000.0], dtype=np.float64)
        self._limits = limits

    @property
    def current_state(self) -> np.ndarray:
        return self._state

    def validate_and_update(
        self,
        price: float,
        quantity: int,
        side: int,
        d_delta: float,
        d_gamma: float,
        d_vega: float,
        max_qty: int = 1000,
        min_price: float = 0.01,
        max_price: float = 10000.0,
    ) -> bool:
        """
        Combined Multi-Point Risk Check.
        Executes in < 300ns using Rust core if available.
        """
        if CORE_AVAILABLE:
            try:
                ok, new_d, new_g, new_v = bsopt_core.full_risk_check(
                    float(price),
                    int(quantity),
                    int(side),
                    float(d_delta),
                    float(d_gamma),
                    float(d_vega),
                    float(self._state[0]),
                    float(self._state[1]),
                    float(self._state[2]),
                    int(max_qty),
                    float(min_price),
                    float(max_price),
                    float(self._limits[0]),
                    float(self._limits[1]),
                    float(self._limits[2]),
                )
                if ok:
                    self._state[0] = new_d
                    self._state[1] = new_g
                    self._state[2] = new_v
                return ok
            except Exception as e:
                logger.warning("rust_risk_vector_check_failed", error=str(e))

        return bool(
            _full_risk_check_v2_kernel(
                price,
                quantity,
                side,
                d_delta,
                d_gamma,
                d_vega,
                self._state,
                self._limits,
                max_qty,
                min_price,
                max_price,
            )
        )

    def reset(self, new_state: np.ndarray):
        """Periodic full-sync to prevent float-point drift."""
        self._state[:] = new_state


class IncrementalDeltaTracker:
    """
    Stateful tracker for portfolio-wide delta exposure.
    Maintains O(1) running total. Optimized to use Rust 'bsopt_core' if available.
    """

    def __init__(self, initial_delta: float = 0.0, max_net_delta: float = 10000.0):
        # Use an array to allow Numba to operate on it directly (zero-copy possible)
        self._state = np.array([initial_delta], dtype=np.float64)
        self.max_net_delta = max_net_delta

    @property
    def current_net_delta(self) -> float:
        return self._state[0]

    def validate_and_update(self, trade_delta: float) -> bool:
        """
        Sub-microsecond validation with state update.
        Uses Rust core or Numba fallback.
        """
        if CORE_AVAILABLE:
            try:
                # Using 0.0 for Gamma/Vega and Price/Qty checks for delta-only tracker
                ok, new_d, _, _ = bsopt_core.full_risk_check(
                    1.0, 1, 1, trade_delta, 0.0, 0.0, self._state[0], 0.0, 0.0, 100, 0.01, 100.0, self.max_net_delta, 1e18, 1e18
                )
                if ok:
                    self._state[0] = new_d
                return ok
            except Exception as e:
                logger.warning("rust_risk_check_failed", error=str(e))

        return bool(_validate_delta_kernel(self._state, trade_delta, self.max_net_delta))

    def full_risk_check(
        self,
        price: float,
        quantity: int,
        side: int,
        trade_delta: float,
        max_qty: int = 1000,
        min_price: float = 0.01,
        max_price: float = 10000.0,
    ) -> bool:
        """
        Combined risk check using Rust core if possible.
        """
        if CORE_AVAILABLE:
            try:
                ok, new_d, _, _ = bsopt_core.full_risk_check(
                    float(price),
                    int(quantity),
                    int(side),
                    float(trade_delta),
                    0.0,
                    0.0,
                    float(self._state[0]),
                    0.0,
                    0.0,
                    int(max_qty),
                    float(min_price),
                    float(max_price),
                    float(self.max_net_delta),
                    1e18,
                    1e18,
                )
                if ok:
                    self._state[0] = new_d
                return ok
            except Exception as e:
                logger.warning("rust_full_risk_failed", error=str(e))

        # Fallback to Numba
        ok = bool(
            _full_risk_check_kernel(
                price,
                quantity,
                side,
                trade_delta,
                self._state,
                max_qty,
                min_price,
                max_price,
                self.max_net_delta,
            )
        )
        return ok

    def reset(self, new_delta: float):
        """Periodic full-sync to prevent drift."""
        self._state[0] = new_delta


@njit(cache=True, fastmath=True)
def _full_risk_check_v2_kernel(
    price: float,
    quantity: int,
    side: int,
    d_delta: float,
    d_gamma: float,
    d_vega: float,
    state_arr: np.ndarray,  # 0:delta, 1:gamma, 2:vega
    limits_arr: np.ndarray,  # 0:max_delta, 1:max_gamma, 2:max_vega
    max_qty: int = 1000,
    min_price: float = 0.01,
    max_price: float = 10000.0,
) -> int:
    """
    Sub-300ns Multi-Dimensional Risk Kernel.
    Validates Delta, Gamma, and Vega in a single pass.
    """
    # 1. Base Silicon Checks
    if (
        price < min_price
        or price > max_price
        or quantity <= 0
        or quantity > max_qty
        or (side != 1 and side != -1)
    ):
        return 0

    # 2. Greeks Multi-Point Validation
    new_delta = state_arr[0] + d_delta
    new_gamma = state_arr[1] + d_gamma
    new_vega = state_arr[2] + d_vega

    if (
        abs(new_delta) > limits_arr[0]
        or abs(new_gamma) > limits_arr[1]
        or abs(new_vega) > limits_arr[2]
    ):
        return 0

    # 3. State Commit (Local Hot-State)
    state_arr[0] = new_delta
    state_arr[1] = new_gamma
    state_arr[2] = new_vega
    return 1
