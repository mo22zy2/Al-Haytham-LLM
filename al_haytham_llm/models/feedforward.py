
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Input:  (..., d_model)
        # Output: (..., d_ff)
        # Expands the hidden representation to a larger dimension.
        self.expansion_layer = nn.Linear(
            config.d_model,
            config.d_ff,
        )

        # Input:  (..., d_ff)
        # Output: (..., d_ff)
        # Applies GELU activation to introduce non-linearity.
        self.activation_function = nn.GELU(approximate="tanh")

        # Input:  (..., d_ff)
        # Output: (..., d_model)
        # Projects the expanded representation back to d_model.
        self.projection_layer = nn.Linear(
            config.d_ff,
            config.d_model,
        )

    def forward(self, hidden_states):
        # Input:  (batch_size, sequence_length, d_model)
        # Output: (batch_size, sequence_length, d_ff)
        hidden_states = self.expansion_layer(hidden_states)

        # Input / Output: (batch_size, sequence_length, d_ff)
        hidden_states = self.activation_function(hidden_states)

        # Input:  (batch_size, sequence_length, d_ff)
        # Output: (batch_size, sequence_length, d_model)
        return self.projection_layer(hidden_states)
