import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.td3.policies import TD3Policy


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, n_positions: int, attn_pdrop: float, resid_pdrop: float):
        super().__init__()
        assert n_embd % n_head == 0
        
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head
        self.register_buffer("mask", torch.tril(torch.ones(n_positions, n_positions)).view(1, 1, n_positions, n_positions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Causal Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, n_positions: int, attn_pdrop: float, resid_pdrop: float):
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

class DecisionTransformer(nn.Module):
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
        activation_function: str = 'relu',
        n_positions: int = 1024,
        resid_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
    ):
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
        
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = nn.Linear(hidden_size, 1)
        self.predict_state = nn.Linear(hidden_size, state_dim)

        self.blocks = nn.Sequential(
            *[Block(hidden_size, n_head, n_positions, attn_pdrop, resid_pdrop) for _ in range(n_layer)]
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
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=states.device)

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
        # For simplicity, let's use the standard DT ordering: (R_t, s_t, a_t)
        # But for inference we usually want a_t given (R_t, s_t).
        
        # Interleave embeddings: (batch, 3 * seq_len, hidden_size)
        stacked = torch.stack([returns_embeddings, state_embeddings, action_embeddings], dim=2)
        token_embeddings = stacked.reshape(batch_size, seq_length * 3, self.hidden_size)

        token_embeddings = self.embed_ln(token_embeddings)

        # Transformer forward
        # Adjust mask for 3 tokens per step
        # (batch, 3 * seq_len)
        all_mask = torch.zeros((batch_size, seq_length * 3), dtype=torch.long, device=states.device)
        all_mask[:, ::3] = attention_mask
        all_mask[:, 1::3] = attention_mask
        all_mask[:, 2::3] = attention_mask
        
        # We need to implement padding masking in the Attention block if we want variable length
        # For now, assuming fixed block processing or simple causal masking (handled by CausalSelfAttention)
        
        x = self.blocks(token_embeddings)

        # Outputs
        # Action prediction comes from state embedding (index 1)
        x_reshaped = x.reshape(batch_size, seq_length, 3, self.hidden_size)
        
        # Predict action given (R, s) -> using output at index 1 (state)
        action_preds = self.predict_action(x_reshaped[:, :, 1, :]) 
        
        # Predict return given (R, s, a) -> next R (index 2)
        return_preds = self.predict_return(x_reshaped[:, :, 2, :])
        
        # Predict state given (R, s, a) -> next s (index 2)
        state_preds = self.predict_state(x_reshaped[:, :, 2, :])

        return state_preds, action_preds, return_preds


class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """Custom transformer feature extractor for RL."""
    def __init__(self, observation_space, features_dim: int = 512, d_model: int = 256, nhead: int = 8, num_layers: int = 4):
        super().__init__(observation_space, features_dim)
        self.d_model = d_model
        input_dim = observation_space.shape[0]
        self.embed = nn.Linear(input_dim, d_model)
        self.blocks = nn.Sequential(
            *[Block(d_model, nhead, n_positions=1024, attn_pdrop=0.1, resid_pdrop=0.1) for _ in range(num_layers)]
        )
        self.out = nn.Linear(d_model, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # (batch, input_dim) -> (batch, 1, d_model)
        x = self.embed(observations).unsqueeze(1)
        x = self.blocks(x)
        return self.out(x.squeeze(1))

class TransformerTD3Policy(TD3Policy):
    """TD3 Policy with Transformer extractor."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
