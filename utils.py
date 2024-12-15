import auraloss
import librosa
import torch
import wandb

from run_train import CHECKPOINT_DIR


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
def stft_loss_fn(sr):
    return auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[1024, 2048, 8192],
        hop_sizes=[256, 512, 2048],
        win_lengths=[1024, 2048, 8192],
        scale="mel",
        n_bins=128,
        sample_rate=sr,
        perceptual_weighting=True,
    )


def save_epoch(gen, mpd, msd, optim_g, optim_d, epoch):
    epoch_name = f"epoch_{epoch:02d}"
    checkpoint_path = CHECKPOINT_DIR / epoch_name
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
