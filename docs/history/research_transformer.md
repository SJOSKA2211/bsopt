# Research: Transformer-based RL Agent

## Objectives
- Integrate Transformer architecture into RL.
- Update observation space for sequence data.

## Findings
- `TradingEnvironment` was flattening the history window into a 1D vector.
- `TransformerFeatureExtractor` was ignoring the sequence and treating it as a single point.
- `_fused_state_kernel` needed to return 2D matrices.

## Strategy
- Refactor `_fused_state_kernel` to return `(window_size, 100)` arrays.
- Update `TradingEnvironment` observation space to match.
- Refactor `TransformerFeatureExtractor` to process the full sequence with attention.

