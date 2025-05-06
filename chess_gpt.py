import torch
import torch.nn as nn
import torch.nn.functional as F
from chess_tokenizer import load_tokenizer

class ChessGPTConfig:
    def __init__(
        self,
        vocab_size,
        n_embd=384,        # Embedding dimension
        n_head=6,          # Number of attention heads
        n_layer=6,         # Number of transformer layers
        block_size=128,    # Maximum sequence length
        dropout=0.1        # Dropout rate
    ):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.dropout = dropout

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # Key, Query, Value projections
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        
        # Output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        
        # Regularization
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()
        
        # Split heads
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Compute attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(k.size(-1))))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        
        # Recombine heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class ChessGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input embedding
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        
        # Final layer normalization
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        # Output head
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.config.block_size, "Sequence length exceeds block size"

        # Get token embeddings
        tok_emb = self.tok_emb(idx)
        
        # Add positional embeddings
        pos_emb = self.pos_emb[:, :t, :]
        x = self.drop(tok_emb + pos_emb)
        
        # Apply transformer blocks
        x = self.blocks(x)
        x = self.ln_f(x)
        
        # Get logits
        logits = self.head(x)

        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        """
        Generate chess moves autoregressively
        """

        token_to_idx, _, _ = load_tokenizer()
        
        for _ in range(max_new_tokens):
            # Get predictions
            logits, _ = self(idx[:, -self.config.block_size:])
            
            # Focus on last time step
            logits = logits[:, -1, :] / temperature
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample next token
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled index to sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Stop if we predict a win/loss/draw token
            if idx_next.item() in [token_to_idx.get('[WWIN]', -1), 
                                 token_to_idx.get('[BWIN]', -1), 
                                 token_to_idx.get('[DRAW]', -1)]:
                break
        
        return idx