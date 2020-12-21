from flax import linen as nn
from jax import numpy as jnp


class FFN(nn.Module):
    """FFN with Gates Linear Units and GELU activation
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
        x = nn.gelu(x, approximate=True)
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
