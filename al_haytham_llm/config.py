from dataclasses import dataclass

@dataclass
class ModelConfig:
    
    vocab_size :int = 32_000
    seq_len : int= 1024
    
    d_model : int = 768
    d_ff: int = 4 * 768  # Default for GPT2 = 4 * d_model
    n_heads:int = 12
    n_layers: int = 12
    qkv_bias:bool = False
    dropout:float =0.1
    pad_id:int = 0
    tie_weights:bool=True

    @property
    def d_k(self):
        return self.d_model//self.n_heads # d_k
    
    
    
    @classmethod
    def tiny(cls):
        return cls(seq_len=512,d_model=64,n_heads=8,n_layers=8)
    