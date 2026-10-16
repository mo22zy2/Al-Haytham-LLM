import torch 
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self,config):
        super().__init__()  
        self.d_k=config.d_k
        self.n_heads=config.n_heads
        
        
        self.w_q=nn.Linear(
            config.d_model,
            config.d_model,
            config.qkv_bais
        )
        self.w_k=nn.Linear(
            config.d_model,
            config.d_model,
            config.qkv_bais
        )
        self.w_v=nn.Linear(
            config.d_model,
            config.d_model,
            config.qkv_bais
        )
        
        self.w_o=nn.Linear(
            config.d_model,
            config.d_model
        )
        
        #Masked Attention Casual Model
        mask = torch.tril(
            torch.ones(config.max_seq_len,
                       config.max_seq_len,
                       dtype=torch.bool)
                            )
        
        self.register_buffer("mask",mask,persistent=False)

    def forward():
        
        
        
        
        
        
        pass