
import torch
import torch.nn as nn

from al_haytham_llm.models.attention import MultiHeadAttention
from al_haytham_llm.models.feedforward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Input:  (batch_size, sequence_length, d_model)
        # Output: (batch_size, sequence_length, d_model)
        # Applies multi-head self-attention to the normalized input.
        self.attention = MultiHeadAttention(config)

        # Input:  (batch_size, sequence_length, d_model)
        # Output: (batch_size, sequence_length, d_model)
        # Processes each token independently through an MLP.
        self.feedforward = FeedForward(config)

        # Normalizes the last dimension: d_model.
        # Input / Output: (batch_size, sequence_length, d_model)
        self.layer_norm1 = nn.LayerNorm(config.d_model)
        self.layer_norm2 = nn.LayerNorm(config.d_model)

        # Randomly zeros some values during training.
        # Input / Output: same shape as the input.
        self.dropout_layer = nn.Dropout(config.dropout)

    def forward(self, hidden_states, kv_cache=None, use_cache=False):
        # hidden_states:
        # Input: (batch_size, sequence_length, d_model)

        # Normalize before attention (Pre-LayerNorm).
        # attention_output has the same shape as hidden_states.
        # updated_kv_cache stores the updated keys and values if caching is enabled.
        attention_output, updated_kv_cache = self.attention(
            self.layer_norm1(hidden_states),
            kv_cache,
            use_cache,
        )

        # Residual connection: add attention output to the original input.
        # Output: (batch_size, sequence_length, d_model)
        hidden_states += self.dropout_layer(attention_output)

        # Normalize before the FeedForward network.
        # Input / Output: (batch_size, sequence_length, d_model)
        normalized_hidden_states = self.layer_norm2(hidden_states)

        # FeedForward temporarily expands d_model to d_ff,
        # then projects it back to d_model.
        # Output: (batch_size, sequence_length, d_model)
        feedforward_output = self.feedforward(normalized_hidden_states)

        # Second residual connection.
        # Output: (batch_size, sequence_length, d_model)
        hidden_states += self.dropout_layer(feedforward_output)

        # Return the transformed hidden states and updated KV cache.
        return hidden_states, updated_kv_cache
