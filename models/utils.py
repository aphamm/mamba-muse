import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

LRELU_SLOPE = 0.1


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


# inputs are of shape (batch_size, channel, length)
# calculate mean and std along the channel dimension
def calculate_mean_std(feat, eps=1e-5):
    feat_std = (feat.var(dim=1, keepdim=True) + eps).sqrt()
    feat_mean = feat.mean(dim=1, keepdim=True)
    return feat_mean, feat_std


def adaptive_instance_normalization(content, style, alpha=1.0):
    assert content.size() == style.size()
    size = content.size()
    style_mean, style_std = calculate_mean_std(style)
    content_mean, content_std = calculate_mean_std(content)
    normalized_feat = (content - content_mean.expand(size)) / content_std.expand(size)
    stylized_feat = normalized_feat * style_std.expand(size) + style_mean.expand(size)
    return alpha * stylized_feat + (1 - alpha) * content


# https://arxiv.org/abs/1602.07868
def conv_weight_norm(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
    transpose=False,
):
    conv_class = nn.ConvTranspose1d if transpose else nn.Conv1d
    layer = weight_norm(
        conv_class(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
    )
    layer.weight.data.normal_(mean=0.0, std=0.01)
    return layer
