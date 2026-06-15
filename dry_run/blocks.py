import torch

import math

from torch import nn

from torch.nn import functional as F



# ---------------------------------------------------------------------------

# Code from blocks.py (condensed into one cell for demo purposes)

# ---------------------------------------------------------------------------



def validate_attention_heads(channels: int, heads: int) -> int:

    if heads < 1:

        raise ValueError("Attention head count must be positive")

    if heads > channels or channels % heads:

        raise ValueError(

            f"Attention channels ({channels}) must be divisible by heads ({heads})"

        )

    return heads





class LayerNorm2d(nn.Module):

    def __init__(self, channels: int) -> None:

        super().__init__()

        self.norm = nn.LayerNorm(channels)



    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)





class ResidualBlock(nn.Module):

    def __init__(self, channels: int, groups: int = 8) -> None:

        super().__init__()

        groups = min(groups, channels)

        while channels % groups:

            groups -= 1

        self.net = nn.Sequential(

            nn.GroupNorm(groups, channels),

            nn.SiLU(),

            nn.Conv2d(channels, channels, 3, padding=1),

            nn.GroupNorm(groups, channels),

            nn.SiLU(),

            nn.Conv2d(channels, channels, 3, padding=1),

        )



    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x + self.net(x)





class FiLMResidualBlock(nn.Module):

    def __init__(self, channels: int, style_dim: int) -> None:

        super().__init__()

        self.norm1 = LayerNorm2d(channels)

        self.norm2 = LayerNorm2d(channels)

        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)

        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

        self.style = nn.Linear(style_dim, channels * 4)



    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:

        gamma1, beta1, gamma2, beta2 = self.style(style).chunk(4, dim=1)

        gamma1 = gamma1[:, :, None, None]

        beta1 = beta1[:, :, None, None]

        gamma2 = gamma2[:, :, None, None]

        beta2 = beta2[:, :, None, None]

        h = self.norm1(x) * (1 + gamma1) + beta1

        h = self.conv1(F.silu(h))

        h = self.norm2(h) * (1 + gamma2) + beta2

        return x + self.conv2(F.silu(h))





class SinusoidalEmbedding(nn.Module):

    def __init__(self, dim: int) -> None:

        super().__init__()

        self.dim = dim



    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:

        half = self.dim // 2

        scale = math.log(10000) / max(half - 1, 1)

        frequencies = torch.exp(

            -scale * torch.arange(half, device=timesteps.device, dtype=torch.float32)

        )

        args = timesteps.float()[:, None] * frequencies[None]

        embedding = torch.cat((args.sin(), args.cos()), dim=1)

        if self.dim % 2:

            embedding = F.pad(embedding, (0, 1))

        return embedding





class ConditionedResBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, condition_dim: int) -> None:

        super().__init__()

        groups_in = math.gcd(in_channels, 8)

        groups_out = math.gcd(out_channels, 8)

        self.norm1 = nn.GroupNorm(groups_in, in_channels)

        self.norm2 = nn.GroupNorm(groups_out, out_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.condition = nn.Linear(condition_dim, out_channels * 2)

        self.skip = (

            nn.Conv2d(in_channels, out_channels, 1)

            if in_channels != out_channels

            else nn.Identity()

        )



    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:

        h = self.conv1(F.silu(self.norm1(x)))

        scale, shift = self.condition(condition).chunk(2, dim=1)

        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

        h = self.conv2(F.silu(h))

        return self.skip(x) + h





class CrossAttention2d(nn.Module):

    def __init__(self, channels: int, context_dim: int, heads: int = 8) -> None:

        super().__init__()

        heads = validate_attention_heads(channels, heads)

        self.norm = LayerNorm2d(channels)

        self.context_proj = nn.Linear(context_dim, channels)

        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)



    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:

        batch, channels, height, width = x.shape

        query = self.norm(x).flatten(2).transpose(1, 2)  # (B, H*W, C)

        key_value = self.context_proj(context)           # (B, T, C)

        attended, _ = self.attention(query, key_value, key_value, need_weights=False)

        return x + attended.transpose(1, 2).reshape(batch, channels, height, width)





def high_pass(x: torch.Tensor) -> torch.Tensor:

    padded = F.pad(x, (2, 2, 2, 2), mode="reflect")

    return x - F.avg_pool2d(padded, kernel_size=5, stride=1)





def haar_wavelet(x: torch.Tensor):

    if x.shape[-2] % 2 or x.shape[-1] % 2:

        x = F.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2), mode="reflect")

    x00 = x[:, :, 0::2, 0::2]

    x01 = x[:, :, 0::2, 1::2]

    x10 = x[:, :, 1::2, 0::2]

    x11 = x[:, :, 1::2, 1::2]

    ll = (x00 + x01 + x10 + x11) * 0.5

    lh = (-x00 - x01 + x10 + x11) * 0.5

    hl = (-x00 + x01 - x10 + x11) * 0.5

    hh = (x00 - x01 - x10 + x11) * 0.5

    return ll, lh, hl, hh



# ---------------------------------------------------------------------------

# Dry‑run demonstration

# ---------------------------------------------------------------------------



torch.manual_seed(0)  # deterministic demo



print("\n=== validate_attention_heads ===")

print("32 channels, 8 heads ->", validate_attention_heads(32, 8))

try:

    validate_attention_heads(30, 8)

except ValueError as e:

    print("30 channels, 8 heads raises:", e)



# Sample feature map

x = torch.randn(2, 32, 8, 8)

print("input shape:", x.shape)
print("input mean:", x.mean().item(), "input std:", x.std().item())
print("input min:", x.min().item(), "input max:", x.max().item())
print("input 5 random values:", x.view(-1)[torch.randperm(x.numel())[:5]].tolist())
print("input mean per channel:", x.mean(dim=[0, 2, 3]))
print("input std per channel:", x.std(dim=[0, 2, 3]))

print("\n=== LayerNorm2d ===")

ln = LayerNorm2d(32)

y = ln(x)

print("output shape:", y.shape, "mean≈", y.mean().item(), "std≈", y.std().item())
print("mean per channel:", y.mean(dim=[0, 2, 3]))
print("std per channel:", y.std(dim=[0, 2, 3]))

print("\n=== ResidualBlock ===")

rb = ResidualBlock(32)

y = rb(x)

print("output shape:", y.shape, "residual diff (L1 mean):", (y - x).abs().mean().item())



print("\n=== FiLMResidualBlock ===")

film = FiLMResidualBlock(32, style_dim=64)

style = torch.randn(2, 64)

y = film(x, style)

print("output shape:", y.shape)



print("\n=== SinusoidalEmbedding ===")

se = SinusoidalEmbedding(10)

timesteps = torch.tensor([0, 1, 10, 999])

emb = se(timesteps)

print("embedding shape:", emb.shape)

print("t=0 vector:", emb[0])



print("\n=== ConditionedResBlock ===")

crb = ConditionedResBlock(32, 64, condition_dim=16)

condition = torch.randn(2, 16)

y = crb(x, condition)

print("output shape:", y.shape)



print("\n=== CrossAttention2d ===")

ca = CrossAttention2d(channels=32, context_dim=64, heads=4)

context = torch.randn(2, 5, 64)  # 5 text tokens

y = ca(x, context)

print("output shape:", y.shape)



print("\n=== high_pass filter ===")

hp = high_pass(x)

print("output shape:", hp.shape, "abs mean:", hp.abs().mean().item())



print("\n=== Haar wavelet decomposition ===")

ll, lh, hl, hh = haar_wavelet(x)

print("LL shape:", ll.shape, "LH shape:", lh.shape, "HL shape:", hl.shape, "HH shape:", hh.shape)