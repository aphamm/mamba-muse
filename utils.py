import auraloss
import librosa
import torch
import torch.nn as nn
import wandb
from torch.nn.utils.parametrizations import weight_norm

from remote import CHECKPOINT_DIR


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim,
        warmup_steps: int,
        base_lr: float,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linearly increase the learning rate
            return [
                self.base_lr * (self.last_epoch + 1) / self.warmup_steps
                for _ in self.optimizer.param_groups
            ]
        else:
            # Use the base learning rate after warmup
            return [self.base_lr for _ in self.optimizer.param_groups]


def mel_spectrogram(x, cfg):
    # input: (batch_size, sequence_length)
    try:
        audio_np = x.cpu().detach().numpy().astype(float)
    except:
        audio_np = x

    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio_np,
        sr=cfg.audio["sr"],
        n_fft=cfg.audio["n_fft"],
        n_mels=cfg.audio["n_mels"],
        fmin=cfg.audio["fmin"],
        fmax=cfg.audio["fmax"],
        center=False,
    )
    return mel_spectrogram


# https://github.com/csteinmetz1/auraloss
def mr_stft_loss_fn(sr):
    return auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[1024, 2048, 8192],
        hop_sizes=[256, 512, 2048],
        win_lengths=[1024, 2048, 8192],
        scale="mel",
        n_bins=128,
        sample_rate=sr,
        perceptual_weighting=True,
    )


def save_epoch(gen, mpd, msd, optim_g, optim_d, epoch, checkpoint_dir):
    epoch_name = f"epoch_{epoch:02d}"
    checkpoint_path = (
        CHECKPOINT_DIR / epoch_name
    )  # os.path.join(checkpoint_dir, epoch_name)
    torch.save(
        {
            "gen": gen.state_dict(),
            "mpd": mpd.state_dict(),
            "msd": msd.state_dict(),
            "optim_g": optim_g.state_dict(),
            "optim_d": optim_d.state_dict(),
            "epoch": epoch,
        },
        checkpoint_path,
    )
    print(f"checkpoint saved for {epoch_name}")


def wandb_mel(mel, sr, caption):
    return wandb.Image(
        librosa.display.specshow(
            mel,
            sr=sr,
            x_axis="time",
            y_axis="mel",
            fmax=24000,
        ),
        caption=caption,
    )


def calculate_mean_std(feat, eps=1e-5):
    feat_std = (feat.var(dim=1, keepdim=True) + eps).sqrt()
    feat_mean = feat.mean(dim=1, keepdim=True)
    return feat_mean, feat_std


# calculate statistic along the channel dimension
# content and style shapes are (batch_size, channel, length)
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


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)
