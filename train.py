import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import itertools
import json
import time

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from dataset import NSynthDataset
from gan import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    discriminator_loss,
    generator_loss,
)
from model import Generator
from utils import AttrDict, WarmupScheduler, loss_fn, mel_spectrogram


def train(rank, h):
    torch.cuda.manual_seed(h.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("using device:", device)

    ##############
    # LOAD MODEL #
    ##############

    generator = Generator(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        msd = DistributedDataParallel(msd, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(
        generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2]
    )

    optim_d = torch.optim.AdamW(
        itertools.chain(msd.parameters(), mpd.parameters()),
        h.learning_rate,
        betas=[h.adam_b1, h.adam_b2],
    )

    scheduler_g_warmup = WarmupScheduler(
        optim_g, warmup_steps=5, base_lr=h.learning_rate
    )

    scheduler_d_warmup = WarmupScheduler(
        optim_g, warmup_steps=5, base_lr=h.learning_rate
    )

    scheduler_g_decay = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=h.lr_decay
    )

    scheduler_d_decay = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=h.lr_decay
    )

    ################
    # LOAD DATASET #
    ################

    trainset = NSynthDataset(dataset="test", shuffle=False if h.num_gpus > 1 else True)
    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None
    train_loader = DataLoader(
        trainset,  # single example of shape (64, 1, 32000)
        num_workers=h.num_workers,
        shuffle=False,
        sampler=train_sampler,
        batch_size=h.batch_size,
        pin_memory=True,
        drop_last=True,
    )

    ###############
    # TRAIN MODEL #
    ###############

    generator.train()
    mpd.train()
    msd.train()

    losses_g = []
    losses_d = []

    # https://github.com/jik876/hifi-gan/blob/master/train.py
    for epoch in range(h.num_epochs):
        if rank == 0:
            start = time.time()

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, x in tqdm(
            enumerate(train_loader), unit="batches", total=len(train_loader)
        ):
            x = torch.autograd.Variable(x.to(device, non_blocking=True))
            y_hat = generator(x)

            y_mel = mel_spectrogram(
                x.squeeze(1),
                sr=h.sr,
                n_fft=h.n_fft,
                n_mels=h.n_mels,
                fmin=h.fmin,
                fmax=h.fmax,
            )

            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))

            try:
                y_hat_mel = mel_spectrogram(
                    y_hat.squeeze(1),
                    sr=h.sr,
                    n_fft=h.n_fft,
                    n_mels=h.n_mels,
                    fmin=h.fmin,
                    fmax=h.fmax,
                )
            except:
                print("y_hat_mel error")

            y_hat_mel = torch.autograd.Variable(y_hat_mel.to(device, non_blocking=True))

            ##################
            # GENERATOR LOSS #
            ##################

            optim_g.zero_grad()

            # L1 mel-spectrogram loss
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * h.lambda_mel

            # multi-resolution STFT loss
            loss_stft = loss_fn(x, y_hat) * h.lambda_stft

            y_df_hat_r, y_df_hat_g, _, _ = mpd(x, y_hat)
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(x, y_hat)

            loss_gen_f, _ = generator_loss(y_df_hat_g)
            loss_gen_s, _ = generator_loss(y_ds_hat_g)

            loss_gen = loss_mel + loss_stft + loss_gen_f + loss_gen_s
            losses_g.append(loss_gen.item())
            loss_gen.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=3.0)
            optim_g.step()

            ######################
            # DISCRIMINATOR LOSS #
            ######################

            optim_d.zero_grad()

            y_df_hat_r, y_df_hat_g, _, _ = mpd(x, y_hat.detach())

            loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)

            y_ds_hat_r, y_ds_hat_g, _, _ = msd(x, y_hat.detach())
            loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc = loss_disc_f + loss_disc_s
            losses_d.append(loss_disc.item())
            loss_disc.backward()

            torch.nn.utils.clip_grad_norm_(mpd.parameters(), max_norm=2.0)
            torch.nn.utils.clip_grad_norm_(msd.parameters(), max_norm=2.0)
            optim_d.step()

            if i % 25 == 0 and rank == 0:
                loss_g_avg = sum(losses_g[-25:]) / 25
                loss_d_avg = sum(losses_d[-25:]) / 25
                print(
                    f"epoch: {epoch + 1}, batch: {i + 1}, gen_loss: {loss_gen.item():.4f}, disc_loss: {loss_disc.item():.4f}"
                )

        if epoch < h.warmup_epoch:
            scheduler_g_warmup.step()
            scheduler_d_warmup.step()
        else:
            scheduler_g_decay.step()
            scheduler_d_decay.step()

        total_mins = (time.time() - start) / 60
        loss_g_avg = sum(losses_g) / len(losses_g)
        loss_d_avg = sum(losses_d) / len(losses_d)
        if rank == 0:
            print(
                f"epoch: {epoch + 1}, gen_loss: {loss_g_avg:.4f}, disc_loss: {loss_d_avg:.4f}, time: {total_mins} mins"
            )


def run_train():
    with open("config.json") as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print("Batch size per GPU :", h.batch_size)
    else:
        pass

    train(0, h)


if __name__ == "__main__":
    main()
