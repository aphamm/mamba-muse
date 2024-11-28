import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import glob
import itertools
import json
import os
import time

import torch
import torch.nn.functional as F
import wandb
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import NSynthDataset
from gan import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    discriminator_loss,
    generator_loss,
)
from model import Generator
from utils import (
    AttrDict,
    WarmupScheduler,
    mel_spectrogram,
    mr_stft_loss_fn,
    save_epoch,
)
from validation import validation


def train(cfg):
    torch.cuda.manual_seed(cfg.train["seed"])
    device = torch.device("cuda")
    wandb.login(key=cfg.wandb["key"])
    wandb.init(project=cfg.wandb["project"])
    wandb.run.name = cfg.wandb["run_name"]
    wandb.config.update(cfg)
    previous_val_err = torch.inf
    nums_did_not_improve = 0
    stop_early = False

    ################
    # LOAD DATASET #
    ################

    trainset = NSynthDataset(split="train", shuffle=True, cfg=cfg)
    train_loader = DataLoader(
        trainset,  # single example of shape (64, 1, 32000)
        num_workers=cfg.train["num_workers"],
        shuffle=False,
        sampler=None,
        batch_size=cfg.train["batch_size"],
        pin_memory=True,
        drop_last=True,
    )
    validset = NSynthDataset(split="validation", shuffle=True, cfg=cfg)
    valid_loader = DataLoader(
        validset,  # single example of shape (1, 1, 32000)
        num_workers=1,
        shuffle=True,
        sampler=None,
        batch_size=1,
        pin_memory=True,
        drop_last=True,
    )

    ##############
    # LOAD MODEL #
    ##############

    gen = Generator(cfg).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    lr = cfg.train["learning_rate"]
    b1 = cfg.train["adam_b1"]
    b2 = cfg.train["adam_b2"]

    optim_g = torch.optim.AdamW(gen.parameters(), lr, betas=[b1, b2])
    optim_d = torch.optim.AdamW(
        itertools.chain(msd.parameters(), mpd.parameters()), lr, betas=[b1, b2]
    )

    ###############
    # CHECKPOINTS #
    ###############

    os.makedirs(cfg.path["checkpoint_dir"], exist_ok=True)
    results = glob.glob(os.path.join(cfg.path["checkpoint_dir"], "*"))

    if len(results) > 0:
        checkpoint = sorted(results)[-1]
        print(f"loading checkpoint from {checkpoint}")

        state_dict = torch.load(checkpoint, map_location=device)

        gen.load_state_dict(state_dict["gen"])
        mpd.load_state_dict(state_dict["mpd"])
        msd.load_state_dict(state_dict["msd"])
        last_epoch = state_dict["epoch"]
        optim_g.load_state_dict(state_dict["optim_g"])
        optim_d.load_state_dict(state_dict["optim_d"])
    else:
        state_dict, last_epoch = None, -1

    warmup = cfg.train["warmup_epoch"]
    gamma = cfg.train["lr_decay"]

    scheduler_g_warmup = WarmupScheduler(optim_g, warmup_steps=warmup, base_lr=lr)
    scheduler_d_warmup = WarmupScheduler(optim_g, warmup_steps=warmup, base_lr=lr)

    scheduler_g_decay = ExponentialLR(optim_g, gamma=gamma, last_epoch=last_epoch)
    scheduler_d_decay = ExponentialLR(optim_d, gamma=gamma, last_epoch=last_epoch)

    ###############
    # TRAIN MODEL #
    ###############

    mpd.train()
    msd.train()

    losses_g = []
    losses_d = []

    # https://github.com/jik876/hifi-gan/blob/master/train.py
    for epoch in range(max(0, last_epoch), cfg.train["num_epochs"]):
        start = time.time()
        gen.train()

        for i, x in tqdm(
            enumerate(train_loader), unit="batches", total=len(train_loader)
        ):
            x = x.to(device, non_blocking=True)
            y_hat = gen(x)

            y_mel = torch.from_numpy(mel_spectrogram(x.squeeze(1), cfg)).to(
                device, non_blocking=True
            )

            nan_mask = torch.isnan(y_hat)
            if torch.any(nan_mask):
                nan_indices = torch.nonzero(nan_mask)
                print(f"nan_indices: {nan_indices}")
                y_hat = torch.where(nan_mask, torch.zeros_like(y_hat), y_hat)

            inf_mask = torch.isinf(y_hat)
            if torch.any(inf_mask):
                inf_indices = torch.nonzero(inf_mask)
                print(f"inf_indices: {inf_indices}")
                y_hat = torch.where(inf_mask, torch.zeros_like(y_hat), y_hat)

            y_hat_mel = torch.from_numpy(mel_spectrogram(y_hat.squeeze(1), cfg)).to(
                device, non_blocking=True
            )

            ##################
            # GENERATOR LOSS #
            ##################

            optim_g.zero_grad()

            # L1 mel-spectrogram loss
            loss_mel = F.l1_loss(y_mel, y_hat_mel)

            # multi-resolution STFT loss
            loss_stft = mr_stft_loss_fn(cfg.audio["sr"])(x, y_hat)

            y_df_hat_r, y_df_hat_g, _, _ = mpd(x, y_hat)
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(x, y_hat)

            loss_gen_f, _ = generator_loss(y_df_hat_g)
            loss_gen_s, _ = generator_loss(y_ds_hat_g)

            loss_gen = (
                loss_mel * cfg.hps["lambda_mel"]
                + loss_stft * cfg.hps["lambda_stft"]
                + loss_gen_f
                + loss_gen_s
            )
            losses_g.append(loss_gen.item())
            loss_gen.backward()

            clip_grad_norm_(gen.parameters(), max_norm=cfg.hps["gen_max_norm"])
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

            torch.nn.utils.clip_grad_norm_(
                mpd.parameters(), max_norm=cfg.hps["dis_max_norm"]
            )
            torch.nn.utils.clip_grad_norm_(
                msd.parameters(), max_norm=cfg.hps["dis_max_norm"]
            )
            optim_d.step()

            #################
            # WANDB LOGGING #
            #################

            wandb.log(
                {
                    "step": i + epoch * len(train_loader),
                    "gen_loss": loss_gen.item(),
                    "mel_loss": loss_mel.item(),
                    "stft_loss": loss_stft.item(),
                    "gen_f_loss": loss_gen_f.item(),
                    "gen_s_loss": loss_gen_s.item(),
                    "disc_loss": loss_disc.item(),
                    "disc_f_loss": loss_disc_f.item(),
                    "disc_s_loss": loss_disc_s.item(),
                }
            )

        ##############
        # VALIDATION #
        ##############

        val_err, original_audio, generated_audio, original_mel, generated_mel = (
            validation(gen, valid_loader, device, cfg)
        )

        wandb.log(
            {
                "epoch": epoch,
                "val_mel_loss": val_err,
                "original_audio": original_audio,
                "generated_audio": generated_audio,
                "original_mel": original_mel,
                "generated_mel": generated_mel,
            }
        )

        ##################
        # CHECK POINTING #
        ##################

        if val_err < previous_val_err:
            previous_val_err = val_err
            nums_did_not_improve = 0
            save_epoch(
                gen, mpd, msd, optim_g, optim_d, epoch, cfg.path["checkpoint_dir"]
            )
        else:
            nums_did_not_improve += 1

        ###############
        # EPOCH STATS #
        ###############

        mins = (time.time() - start) / 60
        loss_g_avg = sum(losses_g) / len(losses_g)
        loss_d_avg = sum(losses_d) / len(losses_d)
        lr = optim_g.param_groups[0]["lr"]

        print(
            f"epoch: {epoch}, gen_loss: {loss_g_avg:.6f}, disc_loss: {loss_d_avg:.6f}, lr: {lr:.6f}, time: {mins:.2f} mins"
        )

        ##################
        # EARLY STOPPING #
        ##################

        stop_early = True if nums_did_not_improve > cfg.hps["patience"] else False

        if stop_early:
            print("stopping early to prevent overfitting")
            break
        elif epoch < cfg.train["warmup_epoch"]:
            scheduler_g_warmup.step()
            scheduler_d_warmup.step()
        else:
            scheduler_g_decay.step()
            scheduler_d_decay.step()


def init_train():
    with open("config.json") as f:
        data = f.read()

    json_config = json.loads(data)
    cfg = AttrDict(json_config)

    torch.manual_seed(cfg.train["seed"])

    if not torch.cuda.is_available():
        "CUDA is not available. Exiting..."
        pass

    train(cfg)
