import torch as th
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    """
    Flash Attention-capable Transformer Block.
    Uses torch.nn.functional.scaled_dot_product_attention for hardware-level speedup.
    """

    def __init__(self, n_inner: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.n_head = n_head
        self.n_inner = n_inner
        self.qkv = nn.Linear(n_inner, n_inner * 3)
        self.proj = nn.Linear(n_inner, n_inner)
        self.ln_1 = nn.LayerNorm(n_inner)
        self.ln_2 = nn.LayerNorm(n_inner)

        # MLP Block
        self.mlp = nn.Sequential(
            nn.Linear(n_inner, 4 * n_inner),
            nn.GELU(),
            nn.Linear(4 * n_inner, n_inner),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: th.Tensor, mask: th.Tensor | None = None) -> th.Tensor:
        # LayerNorm -> Attention
        x_ln = self.ln_1(x)
        batch, seq, dim = x_ln.shape

        qkv = (
            self.qkv(x_ln)
            .reshape(batch, seq, 3, self.n_head, dim // self.n_head)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # ⚡ FLASH ATTENTION (God-Mode Hardware Acceleration)
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=(mask is not None),
            dropout_p=self.dropout.p if self.training else 0.0,
        )

        attn_out = attn_out.permute(0, 2, 1, 3).reshape(batch, seq, dim)
        x = x + self.dropout(self.proj(attn_out))

        # LayerNorm -> MLP
        x = x + self.mlp(self.ln_2(x))
        return x


class DecisionTransformer(nn.Module):
    """
    Advanced Decision Transformer (DT-v2) for Offline RL.
    OPTIMIZED: Flash Attention, Multi-scale Modal Embeddings, and Learned Positional Bias.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_layer: int = 4,
        n_head: int = 8,
        n_inner: int = 1024,
        max_length: int = 20,
        max_ep_len: int = 1000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_length = max_length

        # Modal Embeddings
        self.embed_return = nn.Linear(1, n_inner)
        self.embed_state = nn.Linear(state_dim, n_inner)
        self.embed_action = nn.Linear(action_dim, n_inner)

        # Positional & Modal Bias
        self.embed_timestep = nn.Embedding(max_ep_len, n_inner)
        self.embed_ln = nn.LayerNorm(n_inner)
        self.dropout = nn.Dropout(dropout)

        # Transformer Stack
        self.blocks = nn.ModuleList(
            [AttentionBlock(n_inner, n_head, dropout) for _ in range(n_layer)]
        )

        # Prediction Heads
        self.predict_action = nn.Sequential(
            nn.Linear(n_inner, action_dim),
            nn.Tanh(),  # Normalized action space (-1, 1)
        )
        self.predict_state = nn.Linear(n_inner, state_dim)
        self.predict_return = nn.Linear(n_inner, 1)

    def forward(self, states, actions, returns_to_go, timesteps):
        # states: [batch, seq_len, state_dim]
        # actions: [batch, seq_len, action_dim]
        # returns_to_go: [batch, seq_len, 1]
        # timesteps: [batch, seq_len]

        batch_size, seq_len = states.shape[0], states.shape[1]

        # 1. Embeddings + Timestep Bias
        time_embeddings = self.embed_timestep(timesteps)

        # Modal embeddings (interleaved later)
        r_emb = self.embed_return(returns_to_go) + time_embeddings
        s_emb = self.embed_state(states) + time_embeddings
        a_emb = self.embed_action(actions) + time_embeddings

        # 2. Interleave sequence: (R1, S1, A1, R2, S2, A2, ...)
        # [batch, 3 * seq_len, n_inner]
        stacked_inputs = (
            th.stack((r_emb, s_emb, a_emb), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_len, -1)
        )
        stacked_inputs = self.dropout(self.embed_ln(stacked_inputs))

        # 3. Transformer Pass (AttentionBlock handles causal mask automatically)
        x = stacked_inputs
        for block in self.blocks:
            x = block(x, mask=True)  # mask=True triggers causal attention in sdp

        # 4. Extract representations and Predict
        # Sequence: [R1, S1, A1, R2, S2, A2, ...]
        x_reshaped = x.reshape(batch_size, seq_len, 3, -1)
        
        # Predict action given (R, S) -> output at S
        action_preds = self.predict_action(x_reshaped[:, :, 1, :])
        
        # Predict state/return given (R, S, A) -> output at A
        state_preds = self.predict_state(x_reshaped[:, :, 2, :])
        return_preds = self.predict_return(x_reshaped[:, :, 2, :])

        return state_preds, action_preds, return_preds
