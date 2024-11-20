import os

import aiohttp
import librosa
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import Dataset


class NSynthDataset(Dataset):
    def __init__(self, dataset: str = "test", shuffle: bool = True):
        nsynth = load_dataset(
            "jg583/NSynth",
            trust_remote_code=True,
            cache_dir=os.getcwd(),
            storage_options={
                "client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}
            }, # https://github.com/huggingface/datasets/issues/7164
        )

        # remove all synthetic instruments
        nsynth[dataset] = nsynth[dataset].filter(
            lambda x: x["instrument_source_str"] != "synthetic", num_proc=5
        )

        if shuffle:
            nsynth[dataset] = nsynth[dataset].shuffle(seed=42)

        self.dataset = nsynth[dataset]

    def __getitem__(self, idx):
        audio = self.dataset[idx]["audio"]["array"]

        # normalize audio
        audio = librosa.util.normalize(audio) * 0.9
        audio = torch.tensor(audio, dtype=torch.float32)

        # randomly select a 2 second segement of audio
        if audio.shape[0] > 32000:
            start = torch.randint(0, audio.shape[0] - 32000, (1,)).item()
            audio = audio[start : start + 32000]
        else:
            audio = F.pad(audio, (0, 32000 - audio.shape[0]), "constant")

        # create channel dimension
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)

        return audio

    def __len__(self):
        return len(self.dataset)
