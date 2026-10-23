import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Store the model configuration
        self.seq_len = config.seq_len
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.qkv_bias = config.qkv_bias

        self.d_k = config.d_k

        # Linear layers that project the input into Query, Key, and Value
        # Each layer transforms the last dimension from d_model to d_model
        self.query_projection = nn.Linear(
            self.d_model, self.d_model, bias=self.qkv_bias
        )
        self.key_projection = nn.Linear(
            self.d_model, self.d_model, bias=self.qkv_bias
        )
        self.value_projection = nn.Linear(
            self.d_model, self.d_model, bias=self.qkv_bias
        )

        # Final linear layer applied after combining all attention heads
        self.output_projection = nn.Linear(self.d_model, self.d_model)

        # Create a lower-triangular causal mask
        # True means the token is allowed to attend to that position
        # False means the token must not attend to that position
        causal_mask = torch.tril(
            torch.ones(
                config.seq_len,
                config.seq_len,
                dtype=torch.bool
            )
        )

        # Register the mask as a buffer:
        # It moves with the model between devices but is not a trainable parameter
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, hidden_states, kv_cache=None, use_cache=False):
        # hidden_states shape: (batch_size, sequence_length, d_model)
        batch_size, sequence_length, d_model = hidden_states.shape

        # Project input into Query
        # Reshape: (B, S, d_model) -> (B, S, n_heads, d_k)
        # Transpose: (B, S, n_heads, d_k) -> (B, n_heads, S, d_k)
        query_states = self.query_projection(hidden_states).view(
            batch_size, sequence_length, self.n_heads, self.d_k
        ).transpose(1, 2)

        # Project input into Key and arrange dimensions for multi-head attention
        key_states = self.key_projection(hidden_states).view(
            batch_size, sequence_length, self.n_heads, self.d_k
        ).transpose(1, 2)

        # Project input into Value and arrange dimensions for multi-head attention
        value_states = self.value_projection(hidden_states).view(
            batch_size, sequence_length, self.n_heads, self.d_k
        ).transpose(1, 2)

        # Initially, there are no previously cached tokens
        cached_sequence_length = 0
        if kv_cache is not None:

            # Retrieve Key and Value tensors from the previous cache
            cached_key_states, cached_value_states = kv_cache

            # Get the number of previously cached tokens
            # Shape example: (batch, heads, tokens, d_k) = (1, 2, 5, 4)
            cached_sequence_length = cached_key_states.size(2)

            # Append the new Keys and Values to the cached ones
            # Concatenate along dimension 2, which represents the sequence length
            key_states = torch.cat([cached_key_states, key_states], dim=2)
            value_states = torch.cat([cached_value_states, value_states], dim=2)

        # Return the combined Key/Value tensors as the updated cache if requested
        updated_kv_cache = (key_states, value_states) if use_cache else None

        # Total number of tokens available as Keys and Values
        total_key_length = cached_sequence_length + sequence_length

        # Calculate attention scores between Queries and Keys
        # key_states.transpose(-2, -1) changes (B, H, total_key_length, d_k)
        # into (B, H, d_k, total_key_length)
        attention_scores = query_states @ key_states.transpose(-2, -1)

        # Scale scores by sqrt(d_k) to help prevent excessively large values
        attention_scores = attention_scores / (self.d_k ** 0.5)

        # Select the causal-mask rows corresponding to the new query tokens
        # cached_sequence_length skips rows for old tokens;
        # total_key_length includes all available tokens
        # The resulting mask has shape (sequence_length, total_key_length)
        causal_mask = self.causal_mask[
            cached_sequence_length:total_key_length, :total_key_length
        ]

        # Replace forbidden attention scores with negative infinity
        # After softmax, these positions receive zero attention weight
        attention_scores = attention_scores.masked_fill(
            ~causal_mask, float("-inf")
        )

        # Convert attention scores into probabilities along the Key sequence dimension
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Multiply attention probabilities by Values to get the attention output
        # Shape: (B, H, S, total_key_length) @ (B, H, total_key_length, d_k)
        # Result: (B, H, S, d_k)
        attention_output = attention_weights @ value_states

        # Move the heads dimension back next to the sequence dimension
        # Make the tensor contiguous, then merge heads into the d_model dimension
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, d_model
        )

        # Apply the final output projection
        # Return the attention output and the updated KV cache (or None)
        return self.output_projection(attention_output), updated_kv_cache
