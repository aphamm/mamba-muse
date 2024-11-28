import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from torch.nn.utils.parametrize import remove_parametrizations

from utils import adaptive_instance_normalization, conv_weight_norm, get_padding

LRELU_SLOPE = 0.1


class ChannelLayerNorm(nn.Module):
    # LayerNorm along the channel dimension
    # Input is (B, C, L), so we transpose to (B, L, C) before applying LayerNorm
    def __init__(self, shape):
        super().__init__()
        self.ln = nn.LayerNorm(shape)

    def forward(self, x):
        _, C, L = x.size()
        assert C <= 256, "Supports up to 256 channels"
        assert L >= 2100, "Only supports at least 2100 timesteps"
        x = self.ln(x.transpose(1, 2))
        return x.transpose(1, 2)


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()

        self.dilation1 = nn.Sequential(
            nn.LeakyReLU(LRELU_SLOPE),
            conv_weight_norm(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1,
                padding=get_padding(kernel_size, dilation[0]),
                dilation=dilation[0],
            ),
            nn.LeakyReLU(LRELU_SLOPE),
            conv_weight_norm(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1,
                padding=get_padding(kernel_size, 1),
                dilation=1,
            ),
        )

        self.dilation2 = nn.Sequential(
            nn.LeakyReLU(LRELU_SLOPE),
            conv_weight_norm(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1,
                padding=get_padding(kernel_size, dilation[1]),
                dilation=dilation[1],
            ),
            nn.LeakyReLU(LRELU_SLOPE),
            conv_weight_norm(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1,
                padding=get_padding(kernel_size, 1),
                dilation=1,
            ),
        )

    def forward(self, x):
        x = x + self.dilation1(x)
        x = x + self.dilation2(x)
        return x

    def remove_weight_norm(self):
        for i in (1, 3):
            remove_parametrizations(self.dilation1[i], "weight")
            remove_parametrizations(self.dilation2[i], "weight")


class MambaBlock(nn.Module):
    def __init__(self, out_channels, n_blocks, cfg):
        super(MambaBlock, self).__init__()

        self.n_blocks = n_blocks
        self.lns = nn.ModuleList()
        self.mambas = nn.ModuleList()

        for _ in range(self.n_blocks):
            self.lns.append(ChannelLayerNorm(out_channels))
            self.mambas.append(
                Mamba(
                    d_model=out_channels,
                    d_state=cfg.mamba["d_state"],
                    d_conv=cfg.mamba["d_conv"],
                    expand=cfg.mamba["expand"],
                )
            )

    def forward(self, x):
        # (B, C, L) -> mamba(ln(B, L, C)) -> (B, C, L)
        for ln, mamba in zip(self.lns, self.mambas):
            x = x + mamba(ln(x).transpose(1, 2)).transpose(1, 2)

        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        num_mambas,
        bottleneck=False,
        cfg=None,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.bottleneck = bottleneck

        self.conv1 = nn.Sequential(
            conv_weight_norm(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=0,
            ),
            ChannelLayerNorm(out_channels),
            nn.LeakyReLU(LRELU_SLOPE),
        )

        self.conv2 = nn.Sequential(
            conv_weight_norm(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=0,
            ),
            ChannelLayerNorm(out_channels),
            nn.LeakyReLU(LRELU_SLOPE),
        )

        if not self.bottleneck:
            self.Avgpool = nn.AvgPool1d(kernel_size=2, stride=2, padding=0)

        self.mamba = MambaBlock(out_channels, num_mambas, cfg)

    def forward(self, x):
        x = self.conv1(self.pad(x))
        x = x + self.conv2(self.pad(x))

        if not self.bottleneck:
            x = self.Avgpool(x)  # exactly halving the sequence

        return self.mamba(x)

    def pad(self, x):
        # desired_len = x.size(-1) // 2 if not self.bottleneck
        right_pad = 2 if self.bottleneck else self.kernel_size - 1
        x = F.pad(x, (0, right_pad))
        return x

    def remove_weight_norm(self):
        remove_parametrizations(self.conv1[0], "weight")
        remove_parametrizations(self.conv2[0], "weight")


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()

        in_dims = (1, 32, 64, 128, 256)
        out_dims = (32, 64, 128, 256, 256)
        kernel_sizes = (4, 4, 4, 4, 3)
        num_mambas = (2, 2, 2, 2, 3)
        # last ConvBlock is the bottleneck
        bottlenecks = (False, False, False, False, True)

        for i in range(len(in_dims)):
            self.convs.append(
                ConvBlock(
                    in_channels=in_dims[i],
                    out_channels=out_dims[i],
                    kernel_size=kernel_sizes[i],
                    num_mambas=num_mambas[i],
                    bottleneck=bottlenecks[i],
                    cfg=cfg,
                )
            )
            self.lns.append(ChannelLayerNorm(out_dims[i]))

    def forward(self, x):
        residuals = []  # no residual for original signal
        for i, (conv, ln) in enumerate(zip(self.convs, self.lns)):
            x = conv(x)
            residuals.append(x)
            x = ln(x)
            if i < len(self.convs) - 1:
                x = F.leaky_relu(x, LRELU_SLOPE)

        # no residual for bottleneck (reverse order)
        return x, residuals[:-1][::-1]

    def remove_weight_norm(self):
        for layer in self.convs:
            layer.remove_weight_norm()


class Generator(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.encoder = Encoder(cfg)

        self.num_kernels = 1
        self.num_upsamples = 4
        self.ups = nn.ModuleList()
        self.mambas = nn.ModuleList()

        # https://github.com/jik876/hifi-gan/blob/master/models.py
        for i, (u, k) in enumerate(zip([2, 2, 2, 2], [4, 4, 8, 8])):
            self.ups.append(
                conv_weight_norm(
                    in_channels=256 // (2**i),
                    out_channels=256 // (2 ** (i + 1)),
                    kernel_size=k,
                    stride=u,
                    padding=(k - u) // 2,
                    transpose=True,
                )
            )
            self.mambas.append(MambaBlock(256 // (2 ** (i + 1)), 2, cfg))

        self.resblocks = nn.ModuleList()
        for i in range(self.num_upsamples):
            ch = 256 // (2 ** (i + 1))
            for k, d in zip([3], [[1, 3]]):
                self.resblocks.append(ResidualBlock(ch, k, d))

        self.final_upsample = nn.Sequential(
            nn.LeakyReLU(LRELU_SLOPE),
            conv_weight_norm(
                in_channels=ch, out_channels=1, kernel_size=7, stride=1, padding=3
            ),
            nn.Tanh(),
        )

        self.adain = adaptive_instance_normalization

    def forward(self, content, style=None, alphas=(1.0, 1.0, 1.0, 1.0, 0.0)):
        # (B, C=1, L=33600) --> (B, 256, 33600/(2**4)=2100)
        x, residuals = self.encoder(content)

        assert (
            len(residuals) == self.num_upsamples
        ), "Number of residuals should equal number of upsamples/downsamples"

        # apply AdaIN style transfer to bottleneck AND all residuals
        if style is not None:
            style_encoded, style_residuals = self.encoder(style)
            x = self.adain(x, style_encoded, alphas[0])
            for i in range(self.num_upsamples):
                residuals[i] = self.adain(
                    residuals[i], style_residuals[i], alphas[i + 1]
                )

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE) + residuals[i]
            x = self.mambas[i](self.ups[i](x))
            x = self.resblocks[i](x)

        return self.final_upsample(x)

    def remove_weight_norm(self):
        for layer in self.ups:
            remove_parametrizations(layer, "weight")
        for layer in self.resblocks:
            layer.remove_weight_norm()
        remove_parametrizations(self.final_upsample[1], "weight")
        self.encoder.remove_weight_norm()
