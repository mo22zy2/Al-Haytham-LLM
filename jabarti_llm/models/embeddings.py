import torch
import torch.nn as nn

class InputEmbedding(nn.Module):
    def __init__(self,config):
        super().__init__()
        
        self.seq_len=config.seq_len
        self.dropout= nn.Dropout(config.dropout) #Droput Layer
        self.token_embedding= nn.Embedding(
            config.vocab_size,
            config.d_model
        )
        self.postional_embedding = nn.Embedding(
            config.seq_len,
            config.d_model
        )
        
        
        
    def forward(self, input_ids,postion_offset:int=0):
        
        _ , seq_len = input_ids.shape
        
        if seq_len > self.seq_len:
            raise ValueError(
                f"sequance length {seq_len} exceeds"
                f"seq_len = {self.seq_len}"
            )
            
        postions = torch.arange(
            postion_offset,
            postion_offset+seq_len,
            device=input_ids.device
        )
        
        tokens=self.token_embedding(input_ids)
        
        pos = self.postional_embedding(postions)
        
        return self.dropout(tokens+pos) # (B,seq_len,d_model)
