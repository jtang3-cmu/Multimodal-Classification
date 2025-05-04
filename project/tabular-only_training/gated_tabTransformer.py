import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from einops import rearrange, repeat

from hyper_connections import HyperConnections

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# PreNorm wrapper
class PreNorm(Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# Gating mechanism
class Gating(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (batch, seq, dim)
        coeff = torch.sigmoid(self.gate(x))
        return x * coeff

# GEGLU & FeedForward
class GEGLU(Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )
        self.gating = Gating(dim)

    def forward(self, x, **kwargs):
        out = self.net(x)
        return self.gating(out)

# Attention with gating
class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 16,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.gating = Gating(dim)

    def forward(self, x):
        b, n, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))
        attn_scores = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = attn_scores.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = self.heads)
        out = self.to_out(out)
        return self.gating(out), attn

# Transformer with gated blocks
class Transformer(Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout,
        num_residual_streams = 4
    ):
        super().__init__()
        self.layers = ModuleList([])
        init_hc, self.expand, self.reduce = HyperConnections.get_init_and_expand_reduce_stream_functions(
            num_residual_streams, disable = num_residual_streams == 1
        )
        for _ in range(depth):
            self.layers.append(ModuleList([
                init_hc(dim=dim, branch=PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout))),
                init_hc(dim=dim, branch=PreNorm(dim, FeedForward(dim, dropout=ff_dropout)))
            ]))

    def forward(self, x, return_attn=False):
        attns = []
        x = self.expand(x)
        for attn_layer, ff_layer in self.layers:
            x, attn = attn_layer(x)
            attns.append(attn)
            x = ff_layer(x)
        x = self.reduce(x)
        if return_attn:
            return x, torch.stack(attns)
        return x

# MLP
class MLP(Module):
    def __init__(self, dims, act=None):
        super().__init__()
        layers = []
        for i in range(len(dims)-1):
            in_dim, out_dim = dims[i], dims[i+1]
            layers.append(nn.Linear(in_dim, out_dim))
            if i < len(dims)-2:
                layers.append(default(act, nn.ReLU()))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

# TabTransformer with gating
class TabTransformer(Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head = 16,
        dim_out = 1,
        mlp_hidden_mults = (4,2),
        mlp_act = None,
        num_special_tokens = 2,
        continuous_mean_std = None,
        attn_dropout = 0.,
        ff_dropout = 0.,
        use_shared_categ_embed = True,
        shared_categ_dim_divisor = 8.,
        num_residual_streams = 4
    ):
        super().__init__()
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)
        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens
        shared_dim = int(dim // shared_categ_dim_divisor) if use_shared_categ_embed else 0
        self.category_embed = nn.Embedding(total_tokens, dim - shared_dim)
        self.use_shared = use_shared_categ_embed
        if use_shared_categ_embed:
            self.shared_embed = nn.Parameter(torch.randn(self.num_categories, shared_dim)*0.02)
        if self.num_unique_categories>0:
            offs = F.pad(torch.tensor(categories), (1,0), value=num_special_tokens)
            offs = offs.cumsum(dim=-1)[:-1]
            self.register_buffer('offsets', offs)
        self.num_continuous = num_continuous
        if num_continuous>0:
            if exists(continuous_mean_std):
                assert continuous_mean_std.shape == (num_continuous,2)
            self.register_buffer('cont_mstd', continuous_mean_std)
            self.norm_cont = nn.LayerNorm(num_continuous)
        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            num_residual_streams=num_residual_streams
        )
        input_size = dim*self.num_categories + num_continuous
        mlp_dims = [input_size, *[input_size*m for m in mlp_hidden_mults], dim_out]
        self.mlp = MLP(mlp_dims, act=mlp_act)

    def forward(self, x_categ, x_cont, return_attn=False):
        parts = []
        if self.num_categories>0:
            x_cat = x_categ + self.offsets
            emb = self.category_embed(x_cat)
            if self.use_shared:
                b = emb.shape[0]
                shared = repeat(self.shared_embed, 'n d -> b n d', b=b)
                emb = torch.cat((emb, shared), dim=-1)
            out, attns = self.transformer(emb, return_attn=True)
            parts.append(rearrange(out, 'b n d -> b (n d)'))
        if self.num_continuous>0:
            cont = x_cont
            if exists(self.cont_mstd):
                m, s = self.cont_mstd.unbind(-1)
                cont = (cont - m) / s
            parts.append(self.norm_cont(cont))
        x = torch.cat(parts, dim=-1)
        logits = self.mlp(x)
        return (logits, attns) if return_attn else logits
