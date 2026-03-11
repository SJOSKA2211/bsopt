import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.td3.policies import TD3Policy
from typing import Any, cast


class CausalSelfAttention(nn.Module): # type: ignore
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_positions: int,
        attn_pdrop: float,
        resid_pdrop: float,
    ) -> None:
        super().__init__()
        assert n_embd % n_head == 0

        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(n_positions, n_positions)).view(1, 1, n_positions, n_positions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # FLASH ATTENTION: Optimized kernel
        # causal mask handled by is_causal=True
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module): # type: ignore
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_positions: int,
        attn_pdrop: float,
        resid_pdrop: float,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, n_positions, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DecisionTransformer(nn.Module): # type: ignore
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden_size: int,
        max_length: int = 20,
        max_ep_len: int = 4096,
        action_tanh: bool = True,
        n_layer: int = 3,
        n_head: int = 1,
        n_inner: int = 4 * 128,
        activation_function: str = "relu",
        n_positions: int = 1024,
        resid_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length
        self.hidden_size = hidden_size

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_state = nn.Linear(state_dim, hidden_size)
        self.embed_action = nn.Linear(act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        layers: list[nn.Module] = [nn.Linear(hidden_size, act_dim)]
        if action_tanh:
            layers.append(nn.Tanh())
        self.predict_action = nn.Sequential(*layers)
        self.predict_return = nn.Linear(hidden_size, 1)
        self.predict_state = nn.Linear(hidden_size, state_dim)

        self.blocks = nn.Sequential(
            *[
                Block(hidden_size, n_head, n_positions, attn_pdrop, resid_pdrop)
                for _ in range(n_layer)
            ]
        )

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.long, device=states.device
            )

        # Embeddings
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns)
        time_embeddings = self.embed_timestep(timesteps)

        # Time embeddings are added similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # Stack embeddings: [R, s, a, R, s, a, ...]
        # Interleave embeddings: (batch, 3 * seq_len, hidden_size)
        stacked = torch.stack([returns_embeddings, state_embeddings, action_embeddings], dim=2)
        token_embeddings = stacked.reshape(batch_size, seq_length * 3, self.hidden_size)

        token_embeddings = self.embed_ln(token_embeddings)

        x = self.blocks(token_embeddings)

        # Outputs
        x_reshaped = x.reshape(batch_size, seq_length, 3, self.hidden_size)

        # Predict action given (R, s) -> using output at index 1 (state)
        action_preds = self.predict_action(x_reshaped[:, :, 1, :])

        # Predict return given (R, s, a) -> next R (index 2)
        return_preds = self.predict_return(x_reshaped[:, :, 2, :])

        # Predict state given (R, s, a) -> next s (index 2)
        state_preds = self.predict_state(x_reshaped[:, :, 2, :])

        return state_preds, action_preds, return_preds


class TransformerFeatureExtractor(BaseFeaturesExtractor): # type: ignore
    """Custom transformer feature extractor for RL handling 2D time-series input."""

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 512,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__(observation_space, features_dim)
        # observation_space.shape is (window_size, 100)
        self.window_size = observation_space.shape[0]
        self.input_dim = observation_space.shape[1]

        self.embed = nn.Linear(self.input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.window_size, d_model))

        self.blocks = nn.Sequential(
            *[
                Block(d_model, nhead, n_positions=1024, attn_pdrop=0.1, resid_pdrop=0.1)
                for _ in range(num_layers)
            ]
        )
        self.ln = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (batch, window_size, 100)
        x = self.embed(observations) + self.pos_embed  # (batch, window_size, d_model)
        x = self.blocks(x)
        x = self.ln(x)

        # Take the latent of the *latest* token for RL policy
        return cast(torch.Tensor, self.out(x[:, -1, :]))


class TransformerTD3Policy(TD3Policy): # type: ignore
    """TD3 Policy with Transformer extractor."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
