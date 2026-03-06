import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

class Snake(nn.Module):
    """
    Snake activation function: x + (sin(alpha * x)^2) / alpha.
    Designed to better model periodic signals (like audio waveforms)
    compared to standard ReLU or GELU.
    """
    def __init__(self, in_features, alpha=1.0):
        super().__init__()
        # Alpha is a learnable parameter per channel
        self.alpha = nn.Parameter(torch.ones(1, in_features, 1) * alpha)

    def forward(self, x):
        # Prevent division by zero or too small values
        alpha_safe = self.alpha.clamp(min=1e-2)
        return x + (torch.sin(alpha_safe * x) ** 2) / alpha_safe

class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation layer.
    Used in Version 3 to dynamically scale and shift the audio features
    based on the external parameter 'p' (e.g., Gain/Drive knob).
    """
    def __init__(self, num_features, cond_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, num_features * 2),
            nn.GELU()
        )

        # Initialize to zero so the layer starts as an identity transform (1.0 scale, 0.0 shift)
        nn.init.zeros_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)

    def forward(self, x, p):
        # Generate scale (gamma) and shift (beta) from parameter p
        cond = self.mlp(p).unsqueeze(-1)
        gamma, beta = torch.chunk(cond, 2, dim=1)
        return x * (1.0 + gamma) + beta

class TCNBlock(nn.Module):
    """
    Residual Block of the TCN.
    Uses Causal Convolution (padding only on the left) to ensure the model
    doesn't "look into the future", making it real-time capable.
    """
    def __init__(self, in_ch, out_ch, dilation, cond_dim, kernel_size=5, use_film=False):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.use_film = use_film

        # Weight-normalized convolution for training stability
        self.conv = weight_norm(nn.Conv1d(
            in_ch, out_ch, kernel_size,
            dilation=dilation, padding=0 # Padding is handled manually in forward
        ))

        if self.use_film:
            self.film = FiLM(out_ch, cond_dim)

        self.act = Snake(out_ch)
        # Residual skip connection (1x1 conv if channel count changes)
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, p=None):
        res = self.res(x)

        # MANUAL CAUSAL PADDING:
        # We only pad the left side to ensure the output sample at time 't'
        # depends only on samples from time 't' and earlier.
        padding_l = (self.kernel_size - 1) * self.dilation
        x_pad = F.pad(x, (padding_l, 0))

        out = self.conv(x_pad)

        # Optional conditioning via FiLM (V3)
        if self.use_film and p is not None:
            out = self.film(out, p)

        # Standard residual block flow: Activation(Signal + Residual)
        return self.act(out + res), out

class ParametricTCN(nn.Module):
    """
    Main model architecture supporting 3 paradigms:
    V1: Basic TCN, parameter concatenated to input.
    V2: TCN with a deeper output stage (multi-layer MLP-like).
    V3: Modern TCN with FiLM conditioning and global skip connections.
    """
    def __init__(self, version='v1', num_layers=10, hidden_ch=64, cond_dim=1):
        super().__init__()
        self.version = version
        self.layers = nn.ModuleList()

        # V1 & V2: Input channel is 2 (Audio + Parameter Channel)
        # V3: Input channel is 1 (Pure Audio), parameter enters via FiLM
        in_ch = 2 if version in ['v1', 'v2'] else 1
        use_film = (version == 'v3')

        # Build TCN stack with exponential dilations (1, 2, 4, 8, 16...)
        # This allows the model to have a very large receptive field.
        self.layers.append(TCNBlock(in_ch, hidden_ch, dilation=1, cond_dim=cond_dim, use_film=use_film))

        for i in range(1, num_layers):
            self.layers.append(TCNBlock(hidden_ch, hidden_ch, dilation=2**i, cond_dim=cond_dim, use_film=use_film))

        # Output stage: Final projection back to 1-channel audio
        if version == 'v1':
            self.out_stage = nn.Conv1d(hidden_ch, 1, kernel_size=1)
        else:
            # Deeper output stage for V2 and V3 for better non-linear reconstruction
            self.out_stage = nn.Sequential(
                nn.Conv1d(hidden_ch, hidden_ch, kernel_size=1),
                Snake(hidden_ch),
                nn.Conv1d(hidden_ch, 1, kernel_size=1)
            )

    def _init_weights(self):
        """Kaiming initialization for better convergence in deep audio networks."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, p):
        # 1. Input Shape Standardization
        if x.dim() == 4: x = x.squeeze(1)

        # Standardize parameter p to [Batch, 1] for conditioning
        if p.dim() == 4:
            p = p.squeeze(-1).squeeze(-1)
        elif p.dim() == 3:
            p = p.squeeze(-1)

        # 2. Parameter Integration
        # For V1/V2, we create a constant 'audio channel' filled with the parameter value
        p_expanded = p.unsqueeze(-1).expand(-1, -1, x.size(-1))

        if self.version in ['v1', 'v2']:
            out = torch.cat([x, p_expanded], dim=1) # Shape: [B, 2, L]
        else:
            out = x # Shape: [B, 1, L]

        # 3. Processing Layers
        skips = []
        for layer in self.layers:
            # V3 passes 'p' into each block for FiLM modulation
            out, skip = layer(out, p if self.version == 'v3' else None)
            skips.append(skip)

        # 4. Global Skip Connections (V3 only)
        # Aggregates features from all layers for a richer representation
        if self.version == 'v3':
            out = torch.stack(skips).sum(dim=0) / (len(skips) ** 0.5)

        return self.out_stage(out)
