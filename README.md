## MambaMuse 🐍

This paper presents a novel U-Net-based architecture that integrates Adaptive Instance Normalization (AdaIN) and Mamba, a Selective State Space Model (SSM), for efficient musical timbre transfer. The proposed method addresses limitations in existing approaches, such as high computational demand, instability, and slow inference speed, by enabling real-time, one-shot timbre transfer with expressive and natural outputs. The integration of AdaIN allows user-driven customization of content-style blending, while Mamba efficiently captures long-term dependencies to maintain coherence over extended musical phrases. By operating directly in the time domain, as opposed to using spectrograms or encoded features, our approach aims to preserve fine-grained audio details for higher fidelity. We hope to demonstrate the model’s performance through a comparative evaluation against state-of-the-art methods, highlighting its efficiency and versatility.

### Install Dependencies

```bash
$ virtualenv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

