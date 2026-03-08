import torch as th
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    """
    God-Mode: Rotary Positional Embeddings (RoPE).
    Provides relative positional information via rotation matrices in complex space.
    """
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (th.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        t = th.arange(self.max_seq_len_cached).type_as(self.inv_freq)
        freqs = th.einsum("i,j->ij", t, self.inv_freq)
        emb = th.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x, seq_len: int):
        return self.cos_cached[:, :, :seq_len, ...], self.sin_cached[:, :, :seq_len, ...]


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return th.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class GatedMLP(nn.Module):
    """
    God-Mode: Gated Linear Unit (SwiGLU variant).
    Commonly used in state-of-the-art LLMs for superior representation power.
    """
    def __init__(self, n_inner: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(n_inner, 4 * n_inner)
        self.w2 = nn.Linear(n_inner, 4 * n_inner)
        self.w3 = nn.Linear(4 * n_inner, n_inner)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w3(self.dropout(F.silu(self.w1(x)) * self.w2(x)))


class AttentionBlock(nn.Module):
    """
    Optimized Transformer Block with Flash Attention, RoPE, and Gated MLP.
    """

    def __init__(self, n_inner: int, n_head: int, dropout: float = 0.1, drop_path: float = 0.0):
        super().__init__()
        self.n_head = n_head
        self.n_inner = n_inner
        self.qkv = nn.Linear(n_inner, n_inner * 3)
        self.proj = nn.Linear(n_inner, n_inner)
        self.ln_1 = nn.LayerNorm(n_inner)
        self.ln_2 = nn.LayerNorm(n_inner)
        
        # ⚡ GATED MLP (SwiGLU)
        self.mlp = GatedMLP(n_inner, dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.drop_path = drop_path

    def _drop_path(self, x: th.Tensor, drop_prob: float = 0.0, training: bool = False) -> th.Tensor:
        if drop_prob == 0.0 or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + th.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(keep_prob) * random_tensor

    def forward(self, x: th.Tensor, mask: th.Tensor | None = None, rotary_emb: tuple | None = None) -> th.Tensor:
        # 1. Attention Path
        x_ln = self.ln_1(x)
        batch, seq, dim = x_ln.shape

        qkv = (
            self.qkv(x_ln)
            .reshape(batch, seq, 3, self.n_head, dim // self.n_head)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        if rotary_emb is not None:
            cos, sin = rotary_emb
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=(mask is not None),
            dropout_p=self.dropout.p if self.training else 0.0,
        )

        attn_out = attn_out.permute(0, 2, 1, 3).reshape(batch, seq, dim)
        
        # Stochastic Depth
        x = x + self._drop_path(self.dropout(self.proj(attn_out)), self.drop_path, self.training)
        
        # 2. MLP Path
        x = x + self._drop_path(self.mlp(self.ln_2(x)), self.drop_path, self.training)
        return x


class DecisionTransformer(nn.Module):
    """
    Advanced Decision Transformer (DT-v2) for Offline RL.
    OPTIMIZED: Flash Attention, RoPE, Gated MLP, Stochastic Depth.
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
        drop_path: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_length = max_length
        self.n_head = n_head
        self.head_dim = n_inner // n_head

        # Modal Embeddings with high-init scale for returns
        self.embed_return = nn.Linear(1, n_inner)
        self.embed_state = nn.Linear(state_dim, n_inner)
        self.embed_action = nn.Linear(action_dim, n_inner)

        # 🌀 RoPE Positional Embedding
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        
        self.embed_ln = nn.LayerNorm(n_inner)
        self.dropout = nn.Dropout(dropout)

        # Linear drop path rate schedule
        dpr = [x.item() for x in th.linspace(0, drop_path, n_layer)]
        self.blocks = nn.ModuleList(
            [AttentionBlock(n_inner, n_head, dropout, dpr[i]) for i in range(n_layer)]
        )

        self.predict_action = nn.Sequential(
            nn.Linear(n_inner, action_dim),
            nn.Tanh(),
        )
        self.predict_state = nn.Linear(n_inner, state_dim)
        self.predict_return = nn.Linear(n_inner, 1)

    def forward(self, states, actions, returns_to_go, timesteps, padding_mask=None):
        batch_size, seq_len = states.shape[0], states.shape[1]

        # Modal embeddings
        r_emb = self.embed_return(returns_to_go)
        s_emb = self.embed_state(states)
        a_emb = self.embed_action(actions)

        # 2. Interleave sequence: (R1, S1, A1, R2, S2, A2, ...)
        stacked_inputs = (
            th.stack((r_emb, s_emb, a_emb), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_len, -1)
        )
        stacked_inputs = self.dropout(self.embed_ln(stacked_inputs))

        # 3. Get RoPE coefficients
        cos, sin = self.rotary_emb(stacked_inputs, 3 * seq_len)

        # 4. Transformer Pass
        x = stacked_inputs
        for block in self.blocks:
            x = block(x, mask=True, rotary_emb=(cos, sin))

        x_reshaped = x.reshape(batch_size, seq_len, 3, -1)
        
        # Predict s_{t+1}, a_t, or r_t
        # action_preds are predicted from (R_t, S_t)
        action_preds = self.predict_action(x_reshaped[:, :, 1, :])
        # state_preds are predicted from (R_t, S_t, A_t)
        state_preds = self.predict_state(x_reshaped[:, :, 2, :])
        # return_preds are predicted from (R_t, S_t, A_t)
        return_preds = self.predict_return(x_reshaped[:, :, 2, :])

        return state_preds, action_preds, return_preds


class QNetwork(nn.Module):
    """
    Critic Network for IQL/CQL integration.
    """
    def __init__(self, state_dim: int, action_dim: int, n_inner: int = 256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, n_inner),
            nn.ReLU(),
            nn.Linear(n_inner, n_inner),
            nn.ReLU(),
            nn.Linear(n_inner, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, n_inner),
            nn.ReLU(),
            nn.Linear(n_inner, n_inner),
            nn.ReLU(),
            nn.Linear(n_inner, 1)
        )

    def forward(self, state, action):
        sa = th.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)


class ValueNetwork(nn.Module):
    """
    Expectile Value Network for IQL.
    """
    def __init__(self, state_dim: int, n_inner: int = 256):
        super().__init__()
        self.v = nn.Sequential(
            nn.Linear(state_dim, n_inner),
            nn.ReLU(),
            nn.Linear(n_inner, n_inner),
            nn.ReLU(),
            nn.Linear(n_inner, 1)
        )

    def forward(self, state):
        return self.v(state)
