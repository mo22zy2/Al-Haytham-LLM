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
        self.W_q = nn.Linear(
            self.d_model, self.d_model, bias=self.qkv_bias
        )
        self.W_k = nn.Linear(
            self.d_model, self.d_model, bias=self.qkv_bias
        )
        self.W_v = nn.Linear(
            self.d_model, self.d_model, bias=self.qkv_bias
        )

        # Final linear layer applied after combining all attention heads
        self.W_o = nn.Linear(self.d_model, self.d_model)

        # Create a lower-triangular causal mask
        # True means the token is allowed to attend to that position
        # False means the token must not attend to that position
        mask = torch.tril(
            torch.ones(
                config.seq_len,
                config.seq_len,
                dtype=torch.bool
            )
        )

        # Register the mask as a buffer:
        # It moves with the model between devices but is not a trainable parameter
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x, kv_cache=None, use_cache=False):
        # x shape: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape

        # Project input into Query
        # Reshape: (B, S, d_model) -> (B, S, n_heads, d_k)
        # Transpose: (B, S, n_heads, d_k) -> (B, n_heads, S, d_k)
        q = self.W_q(x).view(
            batch_size, seq_len, self.n_heads, self.d_k
        ).transpose(1, 2)

        # Project input into Key and arrange dimensions for multi-head attention
        k = self.W_k(x).view(
            batch_size, seq_len, self.n_heads, self.d_k
        ).transpose(1, 2)

        # Project input into Value and arrange dimensions for multi-head attention
        v = self.W_v(x).view(
            batch_size, seq_len, self.n_heads, self.d_k
        ).transpose(1, 2)

        # Initially, there are no previously cached tokens
        past_len = 0
        if kv_cache is not None:

            # Retrieve Key and Value tensors from the previous cache
            past_k , past_v = kv_cache

            # Get the number of previously cached tokens
            # Shape example: (batch, heads, tokens, d_k) = (1, 2, 5, 4)
            past_len = past_k.size(2) # (batch , 2 , token , d_model) => (1,2,5,4)

            # Append the new Keys and Values to the cached ones
            # Concatenate along dimension 2, which represents the sequence length
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # Return the combined Key/Value tensors as the updated cache if requested
        updated_cache = (k, v) if use_cache else None

        # Total number of tokens available as Keys and Values
        total_len = past_len + seq_len

        # Calculate attention scores between Queries and Keys
        # k.transpose(-2, -1) changes (B, H, total_len, d_k)
        # into (B, H, d_k, total_len)
        scores = q @ k.transpose(-2, -1)

        # Scale scores by sqrt(d_k) to help prevent excessively large values
        scores = scores / (self.d_k ** 0.5)

        # Select the causal-mask rows corresponding to the new query tokens
        # past_len skips rows for old tokens; total_len includes all available tokens
        # The resulting mask has shape (seq_len, total_len)
        causal = self.mask[past_len:total_len, :total_len]

        # Replace forbidden attention scores with negative infinity
        # After softmax, these positions receive zero attention weight
        scores = scores.masked_fill(~causal, float("-inf"))

        # Convert attention scores into probabilities along the Key sequence dimension
        weights = F.softmax(scores, dim=-1)

        # Multiply attention probabilities by Values to get the attention output
        # Shape: (B, H, seq_len, total_len) @ (B, H, total_len, d_k)
        # Result: (B, H, seq_len, d_k)
        out = weights @ v

        # Move the heads dimension back next to the sequence dimension
        # Make the tensor contiguous, then merge heads into the d_model dimension
        out = out.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )

        # Apply the final output projection
        # Return the attention output and the updated KV cache (or None)
        return self.W_o(out), updated_cache