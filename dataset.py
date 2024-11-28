import os

import aiohttp
import librosa
import torch
from datasets import load_dataset
from torch.utils.data import Dataset


class NSynthDataset(Dataset):
    def __init__(self, split: str = "train", shuffle: bool = True, cfg=None):
        nsynth = load_dataset(
            cfg.path["data"],
            trust_remote_code=True,
            cache_dir=os.getcwd() + "/" + cfg.path["data_dir"],
            storage_options={
                "client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}
            },  # https://github.com/huggingface/datasets/issues/7164
        )

        if shuffle:
            nsynth[split] = nsynth[split].shuffle(seed=42)

        self.dataset = nsynth[split]

    def __getitem__(self, idx):
        audio = self.dataset[idx]["audio"]["array"]

        # normalize audio
        audio = librosa.util.normalize(audio) * 0.9
        audio = torch.tensor(audio, dtype=torch.float32)

        # create channel dimension
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)

        return audio

    def __len__(self):
        return len(self.dataset)
