import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


from pathlib import Path

import modal

image = (
    modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel")
    .workdir("/mamba-muse")
    .copy_local_file("config.json", ".")
    .copy_local_file("dataset.py", ".")
    .copy_local_file("requirements.txt", ".")
    .copy_local_file("train.py", ".")
    .copy_local_file("utils.py", ".")
    .copy_local_file("remote.py", ".")
    .copy_local_dir("models/", "models/")
    .run_commands("pip install --upgrade pip")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("torchinfo")
    .pip_install("fvcore")
)

app = modal.App("mamba-muse")
volume = modal.Volume.from_name("mamba-volume", create_if_missing=True)
CHECKPOINT_DIR = Path("/checkpoints")


@app.function(
    gpu=modal.gpu.A100(size="80GB"),
    image=image,
    volumes={CHECKPOINT_DIR: volume},
    timeout=43_200,  # 12-hour timeout
)
def entry():
    import json

    import torch
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    from torchinfo import summary

    from models import Generator
    from utils import AttrDict

    with open("config.json") as f:
        data = f.read()

    json_config = json.loads(data)
    cfg = AttrDict(json_config)

    torch.manual_seed(cfg.train["seed"])

    device = torch.device("cuda")

    model = Generator(cfg).to(device)

    input_shape = (2, 1, 25600)

    x = torch.rand(input_shape).to(device)

    print(FlopCountAnalysis(model, x))

    print(parameter_count_table(model))

    print(summary(model, input_size=input_shape))
