import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


from pathlib import Path

import modal

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.10")
    .workdir("/mamba-muse")
    .copy_local_file("requirements.txt", ".")
    .run_commands("pip install --upgrade pip")
    .pip_install("uv")
    .run_commands("uv pip install --system --compile-bytecode -r requirements.txt")
    .run_commands(
        "uv pip install --system --compile-bytecode https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    )
    .run_commands(
        "uv pip install --system --compile-bytecode --no-build-isolation mamba-ssm==2.2.2"
    )
    .copy_local_file("dataset.py", ".")
    .copy_local_file("utils.py", ".")
    .copy_local_dir("models/", "models/")
    .copy_local_file("train.py", ".")
    .copy_local_file("config.json", ".")
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
    from train import init_train

    init_train()

    volume.commit()
