## MambaMuse 🐍

Real-time, one-shot timbre transfer using a U-Net-based architecture that integrates Adaptive Instance Normalization (AdaIN) and Mamba, a Selective State Space Model (SSM). We train directly in the time domain to preserve fine-grained audio details for higher fidelity.

### Install Dependencies

```bash
$ virtualenv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
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

### Evaluation

Total params: 4,205,186
Trainable params: 4,205,186
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 0.02

Input size (MB): 0.20
Forward/backward pass size (MB): 275.25
Params size (MB): 0.04
Estimated Total Size (MB): 275.49




- 1.1. Log-Spectral Distance (LSD)
What It Measures: The Log-Spectral Distance measures the similarity between the log power spectra of the generated and target waveforms.
How to Use: Calculate LSD between the generated audio and the target timbre reference audio. This gives an indication of how closely the harmonic content, spectral envelope, and other frequency characteristics align with the target timbre.
1.2. Mel-Cepstral Distortion (MCD)
What It Measures: The Mel-Cepstral Distortion measures differences between the mel-frequency cepstral coefficients (MFCCs) of the original style and the generated output.
How to Use: MCD is particularly useful for speech and musical timbre, as MFCCs are designed to capture the perceptually relevant spectral features. A lower MCD indicates a closer match between the generated and style audio timbres.
1.3. Signal-to-Noise Ratio (SNR) or Signal-to-Distortion Ratio (SDR)
What It Measures: SNR measures the amount of noise introduced into the generated signal compared to the reference, while SDR measures the overall distortion including artifacts.
How to Use: A higher SNR or SDR suggests that less unwanted noise or distortion has been introduced, implying that the model has retained key details of the audio without adding too many artifacts.
1.4. Perceptual Evaluation of Speech Quality (PESQ) or ViSQOL
What It Measures: PESQ and ViSQOL (Virtual Speech Quality Objective Listener) provide a perceptual score that estimates audio quality in a way that correlates well with human perception.
How to Use: Use PESQp for evaluating speech timbre transfer or ViSQOL for music. These metrics simulate how humans perceive differences in quality between two audio signals and can be a good proxy for the subjective listening experience.


2. Subjective Evaluation for Timbre Transfer
Objective metrics alone might not capture all aspects of timbre transfer quality, especially since the goal is often a perceptual one. Here’s how you can involve subjective methods:

2.1. Listening Tests
A/B Testing: Have listeners conduct A/B tests, where they compare the generated audio to the style reference and rate how well the timbre matches. This allows you to gauge the perceived similarity.
Content and Style Trade-Off: Ask listeners to evaluate whether the musical content has been retained while also having the timbre of the style. This helps you assess if the content is preserved while the timbre is transferred correctly.
MOS (Mean Opinion Score): Use a 1-5 scale where listeners rate the audio for aspects like:
Naturalness: Does the audio sound natural or does it contain artifacts?
Timbre Similarity: How well does the timbre of the output match the intended style?
Quality: Rate the overall quality considering any distortions or noise.