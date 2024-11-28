## MambaMuse 🐍

Real-time, one-shot timbre transfer using a U-Net-based architecture that integrates Adaptive Instance Normalization (AdaIN) and Mamba, a Selective State Space Model (SSM). We train directly in the time domain to preserve fine-grained audio details for higher fidelity.

### Install Dependencies

```bash
$ virtualenv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

### Training Configurations

Wave-U-Mamba
- 2 RTX A6000 GPUs
- batch size 64
- 8 hours to train
- 44255 (40703 train / 3552 val) wav files * 0.7 secs * 48kHz

MambaMuse
- 0.7 * 48_000 / 16_000 = 2.1 secs
- 44255 wav files * 2.1 secs * 16kHz = 33600 seq length
- 41600 train / 3200 val / 128 test

### Run on [Modal](https://modal.com/docs)

``` bash
$ modal run remote.py::entry
$ modal volume ls mamba-volume
$ modal volume get mamba-volume tensor.pth .
```