import torch as th
import torch.nn as nn


class DecisionTransformer(nn.Module):
    """
    SOTA: Decision Transformer for Offline RL.
    Treats trading as a sequence modeling problem: predicting actions from (returns-to-go, state, action).
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
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_length = max_length

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=n_inner,
                nhead=n_head,
                dim_feedforward=n_inner * 4,
                batch_first=True,
                activation="gelu",
            ),
            num_layers=n_layer,
        )

        self.embed_return = nn.Linear(1, n_inner)
        self.embed_state = nn.Linear(state_dim, n_inner)
        self.embed_action = nn.Linear(action_dim, n_inner)
        self.embed_ln = nn.LayerNorm(n_inner)

        # Predicting action
        self.predict_action = nn.Sequential(
            nn.Linear(n_inner, action_dim), nn.Tanh()  # normalized action space
        )

    def forward(self, states, actions, returns_to_go, timesteps):
        # states: [batch, seq_len, state_dim]
        # actions: [batch, seq_len, action_dim]
        # returns_to_go: [batch, seq_len, 1]

        batch_size, seq_len = states.shape[0], states.shape[1]

        # Embeddings
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)

        # Interleave sequence: (R1, S1, A1, R2, S2, A2, ...)
        # [batch, 3 * seq_len, n_inner]
        stacked_inputs = (
            th.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_len, -1)
        )

        stacked_inputs = self.embed_ln(stacked_inputs)

        # Attention mask (causal)
        mask = th.triu(th.ones(3 * seq_len, 3 * seq_len) * float("-inf"), diagonal=1)

        # Transformer pass
        x = self.transformer(stacked_inputs, mask=mask)

        # Predict actions from state representations
        # We take the output at the state positions (index 1, 4, 7...)
        x = x.reshape(batch_size, seq_len, 3, -1)[:, :, 1, :]
        action_preds = self.predict_action(x)

        return action_preds
