"""Numeric RealMLP-TD-S, paper §3 / Table A.1 / Appendix A.2.

Independent teaching implementation. Compare with the pinned authors' standalone
implementation in sources/l053. Categorical preprocessing and full TD are excluded.
"""
import math
import numpy as np
import torch
from torch import nn


def robust_parameters(x):
    """Fit training-column medians and scales, including zero-IQR fallbacks."""
    median = np.median(x, axis=0)
    iqr = np.quantile(x, .75, axis=0) - np.quantile(x, .25, axis=0)
    span = np.ptp(x, axis=0)
    denominator = np.where(iqr != 0, iqr, span / 2)
    scale = np.divide(1., denominator, out=np.zeros_like(denominator), where=denominator != 0)
    return median, scale


def smooth_clip(z):
    """§3: monotone smooth saturation in (-3,3), without hard clipping."""
    return z / np.sqrt(1 + (z / 3) ** 2)


class RobustSmooth:
    def fit(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
            raise ValueError('Fit requires a nonempty finite numeric matrix')
        self.median, self.scale = robust_parameters(x)
        return self

    def transform(self, x):
        return smooth_clip((np.asarray(x) - self.median) * self.scale).astype('float32')


def coslog4(t):
    """§3: four cycles over normalized optimizer-step progress 0 <= t <= 1."""
    return .5 * (1 - math.cos(2 * math.pi * math.log2(1 + 15 * t)))


def ntp_linear(x, weight, bias):
    """W is [input,output]; scale the matrix product, not the bias."""
    return (x @ weight) / math.sqrt(x.shape[-1]) + bias


def last_best(history):
    """Minimize validation error; choose the latest epoch in an exact tie."""
    if not len(history) or not np.isfinite(history).all():
        raise ValueError('Expected nonempty finite validation errors')
    return len(history) - 1 - int(np.argmin(history[::-1]))


class NTPLinear(nn.Module):
    def __init__(self, din, dout, zero=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(din, dout) * (0 if zero else 1))
        self.bias = nn.Parameter(torch.randn(1, dout) * (0 if zero else 1))

    def forward(self, x):
        return ntp_linear(x, self.weight, self.bias)


class RealMLPS(nn.Module):
    """Three hidden layers, learned input scales; zero head; no dropout/decay."""
    def __init__(self, din, width=256, regression=False):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(din))
        dims = [din, width, width, width, 1 if regression else 2]
        self.layers = nn.ModuleList([NTPLinear(a,b,zero=i==3)
                                    for i,(a,b) in enumerate(zip(dims[:-1],dims[1:]))])
        self.activation = nn.Mish() if regression else nn.SELU()

    def forward(self, x):
        h = x * self.scale
        for layer in self.layers[:-1]:
            h = self.activation(layer(h))
        return self.layers[-1](h)

    def parameter_groups(self):
        return [dict(params=[self.scale], factor=6.),
                dict(params=[l.weight for l in self.layers], factor=1.),
                dict(params=[l.bias for l in self.layers], factor=.1)]
