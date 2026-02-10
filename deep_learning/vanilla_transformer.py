"""
PyTorch Transformer Implementation
==================================

A clean, well-documented implementation of the Transformer architecture
from "Attention is All You Need" (Vaswani et al., 2017).

Structure:
    1. Utility Functions
    2. Core Components (Attention, Feed-Forward, LayerNorm, Positional Encoding)
    3. Layer Components (Encoder/Decoder Layers)
    4. Full Architecture (Encoder, Decoder, Transformer)
    5. Helper Functions and Examples
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math


# =============================================================================
# 1. UTILITY FUNCTIONS
# =============================================================================

def clones(module, n_layers):
    """Create n identical copies of a module."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n_layers)])


def subsequent_mask(size):
    """
    Create a mask to hide future tokens in decoder self-attention.
    
    Args:
        size: Sequence length
        
    Returns:
        Boolean mask of shape (1, size, size) where True = attend, False = mask
    """
    attn_shape = (1, size, size)
    mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return mask == 0


# =============================================================================
# 2. CORE COMPONENTS
# =============================================================================

class LayerNorm(nn.Module):
    """
    Layer normalization module.
    """
    
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))  # Learnable scale
        self.beta = nn.Parameter(torch.zeros(features))  # Learnable shift
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    
    Computes attention using multiple heads in parallel:
    MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    """
    
    def __init__(self, n_heads, d_model, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_k = d_model // n_heads  # Dimension per head
        self.n_heads = n_heads
        
        # Linear projections for Q, K, V, and output
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None  # Store for visualization
        
    def scaled_dot_product_attention(self, query, key, value, mask=None):
        """
        Compute scaled dot-product attention.
        
        Args:
            query: (batch, heads, seq_len_q, d_k)
            key: (batch, heads, seq_len_k, d_k)  
            value: (batch, heads, seq_len_k, d_k)
            mask: Optional attention mask
            
        Returns:
            output: (batch, heads, seq_len_q, d_k)
            attention_weights: (batch, heads, seq_len_q, seq_len_k)
        """
        d_k = query.size(-1)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        """
        Forward pass for multi-head attention.
        
        Args:
            query: (batch, seq_len_q, d_model)
            key: (batch, seq_len_k, d_model)
            value: (batch, seq_len_k, d_model)
            mask: Optional attention mask
            
        Returns:
            output: (batch, seq_len_q, d_model)
        """
        batch_size = query.size(0)
        
        # Expand mask for multiple heads
        if mask is not None:
            mask = mask.unsqueeze(1)
        
        # 1) Linear projections and reshape for multi-head attention
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 2) Apply attention
        x, self.attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 3) Concatenate heads and apply final linear layer
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        
        return self.w_o(x)


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    
    A two-layer feed-forward network with ReLU activation.
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class PositionalEncoding(nn.Module):
    """
    Positional encoding using sinusoidal functions.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(0)  # Add batch dimension
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add positional encoding to input embeddings
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return self.dropout(x)


class Embeddings(nn.Module):
    """Token embeddings scaled by sqrt(d_model)."""
    
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


# =============================================================================
# 3. LAYER COMPONENTS
# =============================================================================

class SublayerConnection(nn.Module):
    """
    Residual connection followed by layer norm.
    
    output = LayerNorm(x + Sublayer(x))
        x ──┐
    │
    ├─→ LayerNorm → Sublayer (Att / FFN) → Dropout ─┐
    │                                               │
    └──────────────────  Residual ──────────────────┼─→ output
                                                    │
                                                   ADD
    Note: This applies layer norm before the sublayer (Pre-LN),
    which is often more stable than Post-LN.
    """
    
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """
    Single encoder layer with self-attention and feed-forward.
    
    Structure:
    Input → Self-Attention (with residual) → Feed-Forward (with residual) → Output
    """
    
    def __init__(self, size, self_attention, feed_forward, dropout):
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
        
    def forward(self, x, mask):
        # Self-attention sublayer
        x = self.sublayer[0](x, lambda x: self.self_attention(x, x, x, mask))
        # Feed-forward sublayer
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    """
    Single decoder layer with masked self-attention, cross-attention, and feed-forward.
    
    Structure:
        Decoder Input → Masked Self-Attention → src-Attention → Feed-Forward → Output
     ↑               ↑                      ↑                ↑
     └─residual─────┘                      │                │
                                           │                │
                    Encoder Output ────────┘                │
                                                            │
                                           └───residual─────┘
    """
    
    def __init__(self, size, self_attention, cross_attention, feed_forward, dropout):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
        self.size = size
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Masked self-attention sublayer
        x = self.sublayer[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
        # Cross-attention sublayer (attend to encoder output)
        x = self.sublayer[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        # Feed-forward sublayer
        return self.sublayer[2](x, self.feed_forward)


# =============================================================================
# 4. FULL ARCHITECTURE
# =============================================================================

class Encoder(nn.Module):
    """Stack of N encoder layers."""
    
    def __init__(self, layer, n_layers):
        super().__init__()
        self.layers = clones(layer, n_layers)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        """Pass input through each layer in sequence."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N decoder layers."""
    
    def __init__(self, layer, n_layers):
        super().__init__()
        self.layers = clones(layer, n_layers)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """Pass input through each layer in sequence."""
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class Generator(nn.Module):
    """Final linear layer + log-softmax for generating output probabilities."""
    
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        return F.log_softmax(self.projection(x), dim=-1)


class EncoderDecoder(nn.Module):
    """
    Complete Encoder-Decoder architecture.
    
    This is the main model that combines encoder, decoder, embeddings, and generator.
    """
    
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed  # Source embeddings + positional encoding
        self.tgt_embed = tgt_embed  # Target embeddings + positional encoding
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        """Standard forward pass: encode source, then decode target."""
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(encoder_output, src_mask, tgt, tgt_mask)
        return decoder_output
    
    def encode(self, src, src_mask):
        """Encode source sequence."""
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        """Decode target sequence given encoder output."""
        return self.decoder(self.tgt_embed(tgt), encoder_output, src_mask, tgt_mask)


class Transformer(nn.Module):
    """
    Complete Transformer model.
    
    This is the main class that assembles all components into the full
    Transformer architecture from "Attention is All You Need".
    """
    
    def __init__(self, src_vocab_size, tgt_vocab_size, n_layers=6, d_model=512, 
                 d_ff=2048, n_heads=8, dropout=0.1, max_len=5000):
        """
        Initialize Transformer model.
        
        Args:
            src_vocab_size: Size of source vocabulary
            tgt_vocab_size: Size of target vocabulary
            n_layers: Number of encoder/decoder layers
            d_model: Model dimension
            d_ff: Feed-forward dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
            max_len: Maximum sequence length for positional encoding
        """
        super().__init__()
        
        # Create shared components
        attention = MultiHeadAttention(n_heads, d_model, dropout)
        feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout, max_len)
        
        # Build the complete model
        self.model = EncoderDecoder(
            encoder=Encoder(
                EncoderLayer(d_model, copy.deepcopy(attention), 
                           copy.deepcopy(feed_forward), dropout), 
                n_layers
            ),
            decoder=Decoder(
                DecoderLayer(d_model, copy.deepcopy(attention), 
                           copy.deepcopy(attention), copy.deepcopy(feed_forward), 
                           dropout), 
                n_layers
            ),
            src_embed=nn.Sequential(
                Embeddings(d_model, src_vocab_size), 
                copy.deepcopy(position)
            ),
            tgt_embed=nn.Sequential(
                Embeddings(d_model, tgt_vocab_size), 
                copy.deepcopy(position)
            ),
            generator=Generator(d_model, tgt_vocab_size)
        )
        
        # Initialize parameters with Xavier uniform
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize model parameters."""
        for param in self.model.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        """Forward pass through the transformer."""
        return self.model(src, tgt, src_mask, tgt_mask)
    
    def encode(self, src, src_mask):
        """Encode source sequence."""
        return self.model.encode(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        """Decode target sequence."""
        return self.model.decode(encoder_output, src_mask, tgt, tgt_mask)
    
    def generate_step(self, encoder_output, src_mask, tgt):
        """Generate next token probabilities."""
        tgt_mask = subsequent_mask(tgt.size(-1)).to(tgt.device)
        decoder_output = self.decode(encoder_output, src_mask, tgt, tgt_mask)
        return self.model.generator(decoder_output[:, -1])


# =============================================================================
# 5. HELPER FUNCTIONS AND EXAMPLES
# =============================================================================

def make_model(src_vocab_size, tgt_vocab_size, n_layers=6, d_model=512, 
               d_ff=2048, n_heads=8, dropout=0.1):
    """
    Create a Transformer model with specified hyperparameters.
    
    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        n_layers: Number of layers (default: 6)
        d_model: Model dimension (default: 512)
        d_ff: Feed-forward dimension (default: 2048)
        n_heads: Number of attention heads (default: 8)
        dropout: Dropout rate (default: 0.1)
        
    Returns:
        Transformer model
    """
    return Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        n_layers=n_layers,
        d_model=d_model,
        d_ff=d_ff,
        n_heads=n_heads,
        dropout=dropout
    )


def create_padding_mask(seq, pad_idx=0):
    """Create padding mask for sequences."""
    return (seq != pad_idx).unsqueeze(-2)


def create_look_ahead_mask(seq):
    """Create look-ahead mask for decoder self-attention."""
    seq_len = seq.size(-1)
    look_ahead_mask = subsequent_mask(seq_len).to(seq.device)
    padding_mask = create_padding_mask(seq)
    return padding_mask & look_ahead_mask


def count_parameters(model):
    """Count total and trainable parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


# =============================================================================
# 6. EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("🤖 Creating Transformer Model...")
    print("=" * 50)
    
    # Model hyperparameters
    config = {
        'src_vocab_size': 10000,
        'tgt_vocab_size': 10000,
        'n_layers': 6,
        'd_model': 512,
        'd_ff': 2048,
        'n_heads': 8,
        'dropout': 0.1
    }
    
    # Create model
    model = make_model(**config)
    
    # Print model info
    total_params, trainable_params = count_parameters(model)
    print(f"Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print()
    
    # Example forward pass
    print("🔄 Running example forward pass...")
    
    # Create sample data
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 9
    
    src = torch.randint(1, 1000, (batch_size, src_seq_len))
    tgt = torch.randint(1, 1000, (batch_size, tgt_seq_len))
    
    # Create masks
    src_mask = create_padding_mask(src)
    tgt_mask = create_look_ahead_mask(tgt)
    
    # Forward pass
    with torch.no_grad():  # No gradients needed for example
        output = model(src, tgt, src_mask, tgt_mask)
    
    print(f"Results:")
    print(f"   Source shape: {src.shape}")
    print(f"   Target shape: {tgt.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Expected shape: ({batch_size}, {tgt_seq_len}, {config['tgt_vocab_size']})")
    
    print("\nModel created and tested successfully!")
