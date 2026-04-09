## 2026-04-09 - Vectorized Array Math Overhead
**Learning:** Found a common performance anti-pattern in the ML/pricing loops: instantiating pandas Series (`pd.Series()`) inside hot loops simply to utilize built-in methods like `.pct_change()`. This introduces massive object creation and context-switching overhead, causing ~4-6x slowdowns compared to pure NumPy vectorization.
**Action:** Replace pandas array instantiation wrappers with pure NumPy equivalents (e.g., `np.diff(arr) / arr[:-1]`) across all high-frequency simulation and backtesting kernels.
