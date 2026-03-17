# Plan: Transformer-based RL Agent

## Steps
1.  **Refactor Silicon Kernel**:
    -   Update `_fused_state_kernel` in `kernels.py` to return 2D array.
    -   Maintain chronological order in the output window.

2.  **Update Environment**:
    -   Update `observation_space` in `TradingEnvironment`.
    -   Ensure `reset` and `step` return 2D tensors.

3.  **Refactor Transformer Policy**:
    -   Update `TransformerFeatureExtractor` to accept 3D input `(batch, seq, dim)`.
    -   Add positional embeddings.
    -   Use the latest token latent for the policy output.

## Validation
-   Run a test script to verify `TradingEnvironment.step()` returns `(window_size, 100)`.
-   Instantiate `TransformerTD3Policy` with the new environment and verify `forward()` pass.

