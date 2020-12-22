from flax import linen as nn
from jax import numpy as jnp


class FeedForward(nn.Module):
    """Position-wise FeedForward with Gates Linear Units and GELU activation
    as in https://arxiv.org/pdf/2002.05202.pdf
    """
    multiplicative: int = 4  # d_ff / d_model
    dropout_rate: float = 0.

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        d_model = x.shape[-1]
        # keep similar number of params wrt non-GLU
        mult = int(self.multiplicative/3 * 2)
        d_ff = d_model * mult

        gate = nn.Dense(d_ff, use_bias=False, name="wi_0")(x)
        x = nn.Dense(d_ff, use_bias=False, name="wi_1")(x)
        x = nn.gelu(gate, approximate=True) * x
        x = nn.Dropout(rate=self.dropout_rate, name="dropout")(x)
        x = nn.Dense(d_model, use_bias=False, name="wo")(x)
        return x


class SelfAttention(nn.Module):
    num_heads: int
    causal: bool = True
    dropout_rate: float = 0.0
    
    @nn.compact
    def __call__(self, x, attn_mask, deterministic: bool = True):
        """
        Args:
        - x of shape (b, n, d)
        - attn_mask of shape (b, n)
        """
        b, n, d, h = *x.shape, self.num_heads
        head_size = d / h

        # (b, n, d*3)
        x = nn.Dense(d * 3, use_bias=False, name="qkv_projection")(x)
        # (b, n, d) -> (b, n, h, hsize) -> (b, h, n, hsize)
        q, k, v = [_x.reshape(b, n, h, head_size).transpose((0, 2, 1, 3)) for _x in x.split(3, axis=-1)] 
        # attention : (b, h, n, n)
        attention = q @ k.transpose((0,1,3,2)) * (d ** -0.5)
        
        if attn_mask is None and self.causal:
            attn_mask = jnp.triu( jnp.ones( (n, n) ), k=1 )

        # Fill -inf into mask
        if attn_mask is not None:
            # (b, 1, n, 1) * (b, 1, 1, n)
            attn_mask = attn_mask[:, None, :, None] * attn_mask[:, None, None, :]
            attention = attention * attn_mask
            attention = jnp.where(attention == 0., -jnp.inf, attention)
        
        attention_weights = nn.softmax(attention, axis=-1)
        attention_weights = nn.Dropout(self.dropout_rate)(attention_weights)
        # context : (b, h, n, hsize) = (b, h, n, n) * (b, h, n, hsize) 
        context = attention_weights @ v
        context = context.transpose((0, 2, 1, 3)).reshape((b,n,d))
        context = nn.Dense(d, name="out_projection")(context)
        return context


class CrossAttention(nn.Module):
    num_heads: int
    dropout_rate: float = 0.

    @nn.compact
    def __call__(self, x, context, doc_similarities, mask = None, context_mask = None):
        b, n, d, h = *x.shape, self.num_heads
        head_size = d / h
        beta = self.param("beta", lambda rng, shape: jnp.ones(shape=()), ())

        # (b, n, d)
        q = nn.Dense(d, use_bias=False, name="q_projection")(x)
        # (b, h, n, hsize)
        q = q.reshape((b, n, h, head_size)).transpose((0, 2, 1, 3))

        _, m, context_len, _ = context.shape # m = num evidences
        # (b, m, cn, d) -> (b, m*cn, d)
        context = context.reshape((b, m*context_len, d))
        if context_mask is not None:
            context_mask = context_mask.reshape((b, m*context_len))

        # (b, m) -> (b, m, cn) -> (b, m*cn)
        doc_similarities = doc_similarities[:, :, None].repeat(context_len, axis=-1)
        doc_similarities = doc_similarities.reshape((b, m*context_len))
        # (b, 1, 1, m*cn)
        doc_similarities = doc_similarities[:, None, None, :] * beta

        # (b, m*cn, d) -> (b, m*cn, 2*d)
        kv = nn.Dense(d * 2, use_bias=False, name="kv_projection")(context)
        # (b, m*cn, d)
        k, v = kv.split(2, axis=-1)
        # (b, m*cn, h, hsize) -> (b, h, hsize, m*cn)
        k = k.reshape((b, m*context_len, h, head_size)).transpose((0, 2, 3, 1))
        # (b, h, m*cn, hsize)
        v = v.reshape((b, m*context_len, h, head_size)).transpose((0, 2, 1, 3))
        # (b, h, n, m*cn) = (b, h, n, hsize) * (b, h, hsize, m*cn)
        attention_pre_doc_sim = q @ k * (d ** -0.5)
        # (b, h, n, m*cn) + (b, 1, 1, m*cn)
        attention = attention_pre_doc_sim + doc_similarities

        if mask is not None or context_mask is not None:
            if mask is None:
                mask = jnp.full((b, n), True)

            if context_mask is None:
                context_mask = jnp.full(context.shape[:2], True)

            cross_mask = mask[:, None, :, None] * context_mask[:, None, None, :]
            attention = attention * cross_mask
            attention = jnp.where(attention == 0., -jnp.inf, attention)
            
        attention_weights = nn.softmax(attention, axis=-1)
        attention_weights = nn.Dropout(rate=self.dropout_rate)(attention_weights)
        # (b, h, n, hsize) = (b, h, n, m*cn) * (b, h, m*cn, hsize)
        context = attention_weights @ v
        # (b, h, n, hsize) -> (b, n, h, hsize) -> (b, n, d)
        context = context.transpose((0, 2, 1, 3)).reshape((b, n, d))
        context = nn.Dense(d, name="out_projection")(context)
        
        return context, attention_pre_doc_sim


class SubLayer(nn.Module):
    num_heads: int
    ff_multiplicative: int = 4
    causal: bool = False
    attention_dropout: float = 0.
    ff_dropout: float = 0.

    def setup(self):
        self.ln0 = nn.LayerNorm()
        self.self_attn = SelfAttention(num_heads=self.num_heads, causal=self.causal, dropout_rate=self.attention_dropout)
        self.ln1 = nn.LayerNorm()
        self.ffn = FeedForward(multiplicative=self.ff_multiplicative, dropout_rate=self.ff_dropout)

    def __call__(self, x, attention_mask = None):
        x = self.self_attn(self.ln0(x), attention_mask) + x
        x = self.ffn(self.ln1(x)) + x
        return x
    

class Encoder(nn.Module):
    N: int
    num_heads: int
    N_retrieval: int = 4  # first 4 layers of the encoder (paper)
    ff_multiplicative: int = 4
    attention_dropout: float = 0.
    ff_dropout: float = 0.

    def setup(self):
        self.retrieval = [
            SubLayer(num_heads=self.num_heads, ff_multiplicative=self.ff_multiplicative, 
                     attention_dropout=self.attention_dropout, 
                     ff_dropout=self.ff_dropout)
            for i in range(self.N_retrieval)
        ]
        self.encoder_tail = [
            SubLayer(num_heads=self.num_heads, ff_multiplicative=self.ff_multiplicative, 
                     attention_dropout=self.attention_dropout, 
                     ff_dropout=self.ff_dropout)
            for i in range(self.N - self.N_retrieval)          
        ]

    def __call__(self, x, src_mask = None, only_retrieval = False):        
        for layer in self.retrieval:
            x = layer(x, attention_mask = src_mask)

        if only_retrieval:
            return x

        for layer in self.encoder_tail:
            x = layer(x, attention_mask = src_mask)

        return x
    

class DecoderSubLayer(nn.Module):
    "same as SubLayer with an addition of CrossAttention"
    num_heads: int
    causal : bool = True
    ff_multiplicative: int = 4
    attention_dropout: float = 0.
    ff_dropout: float = 0.

    def setup(self):
        self.ln0 = nn.LayerNorm()
        self.self_attn = SelfAttention(num_heads=self.num_heads, causal=self.causal, dropout_rate=self.attention_dropout)
        self.ln1 = nn.LayerNorm()
        self.ffn0 = FeedForward(multiplicative=4, dropout_rate=self.ff_dropout)
        self.cross_attn = CrossAttention(num_heads=self.num_heads, dropout_rate=self.attention_dropout)
        # larger ffn post-cross attn
        self.ffn1 = FeedForward(multiplicative=self.ff_multiplicative, dropout_rate=self.ff_dropout)

    def __call__(self, x, context, similarities, src_mask = None, context_mask = None):
        x = self.self_attn(self.ln0(x), src_mask) + x
        x = self.ffn(self.ln1(x)) + x
        x_out, attn = self.cross_attn(x, context, similarities, src_mask = src_mask, context_mask = context_mask)
        x = x_out + x
        x = self.ffn1(x) + x
        return x, attn


class Decoder(nn.Module):
    N: int
    N_head: int   # d 4 additional Transformer layers to the base of the decoder with only self-attention and
    # feedforward layers of size 4096
    num_heads: int
    ff_multiplicative: int
    attention_dropout: float = 0.
    ff_dropout: float = 0.

    def setup(self):
        self.decoder_head = [
            SubLayer(num_heads=self.num_heads, ff_multiplicative=self.ff_multiplicative, 
                     causal=True,
                     attention_dropout=self.attention_dropout, 
                     ff_dropout=self.ff_dropout)
            for i in range(self.N_head)   
        ]
        
        self.decoder_tail = [
            SubLayer(num_heads=self.num_heads, ff_multiplicative=self.ff_multiplicative, 
                     attention_dropout=self.attention_dropout, 
                     ff_dropout=self.ff_dropout)
            for i in range(self.N - self.N_head)   
        ]
    
    def __call__(self, x, context, similarities, src_mask = None, context_mask = None):
        for layer in self.decoder_head:
            x = layer(x)
            
        cross_pre_attns = []
        for layer in self.decoder_tail:
            x, cross_attn = layer(x)
            cross_pre_attns.append(cross_attn)
        
        return x, cross_pre_attns
        