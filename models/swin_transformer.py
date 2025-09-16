# models/swin_transformer.py
import torch
import torch.nn as nn


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, input_dim, window_size, num_heads):
        super().__init__()
        self.dim = input_dim
        self.window_size = (window_size, window_size)
        self.num_heads = num_heads
        head_dim = input_dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(input_dim, input_dim * 3)
        self.proj = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, input_dim, num_heads, window_size=7, shift_size=0):
        super().__init__()
        self.dim = input_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(input_dim)
        self.attn = WindowAttention(input_dim=input_dim, window_size=self.window_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(input_dim)
        mlp_hidden_dim = int(input_dim * 4.0)
        self.mlp = nn.Sequential(nn.Linear(input_dim, mlp_hidden_dim), nn.GELU(), nn.Linear(mlp_hidden_dim, input_dim))

    def forward(self, x):
        B, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x