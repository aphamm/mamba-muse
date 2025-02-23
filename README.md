## MambaMuse 🐍

Real-time, one-shot timbre transfer using a U-Net-based architecture that integrates Adaptive Instance Normalization (AdaIN) and Mamba, a Selective State Space Model (SSM). We train directly in the time domain to preserve fine-grained audio details for higher fidelity.

### Install Dependencies

Sign up using a free modal account and get `$30` worth of compute credit. Optionally, change the `wandb.key` in `config.json`.


```bash
$ virtualenv .venv
$ source .venv/bin/activate
$ pip install modal
$ modal setup
```

### Run on [Modal](https://modal.com/docs)

``` bash
# start training
$ modal run --detach run_train.py
# view distributed file system
$ modal volume ls mamba-volume
# download model weights
$ modal volume get mamba-volume epoch_00 .
```

### Inference

See the `inference.ipynb` for details on how to use the model.