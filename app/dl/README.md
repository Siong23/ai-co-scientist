# 5G Deep Learning IDS Experiments

PyTorch implementation of four sequential deep-learning models for the
5G-NIDD intrusion-detection experiments:

- LSTM
- LSTM + temporal attention
- BiLSTM
- BiTAD (stacked BiLSTM + single-head temporal self-attention)

The architecture is based on the paper:

Lau et al. (2025), "BiTAD: An Interpretable Temporal Anomaly Detector
for 5G Networks with TwinLens Explainability", Future Internet 17(11), 482.

## Dataset

Put the Kaggle 5G-NIDD CSV at:

    data/5g_nidd/5g_nidd.csv

The pipeline expects a `Label` column.

## Run

From the project root:

    python -m app.experiments.experiment_runner --binary --epochs 10

For a quick smoke test:

    python -m app.experiments.experiment_runner --binary --epochs 1

Change the sequence length:

    python -m app.experiments.experiment_runner --binary --sequence-length 5

## Important

This is a PyTorch reimplementation for your experiments. The reference
paper reports TensorFlow 2.12, 10 epochs, Adam, learning rate 0.001,
batch size 32, dropout 0.3, sequence window T=5, 70:30 train/test,
and three repeated runs. Exact numerical reproduction is not guaranteed
because framework, preprocessing details, randomization, and implementation
choices can differ.

For a rigorous comparison, add repeated seeds (42, 43, 44) and report
mean ± standard deviation.

The current categorical preprocessing uses integer-coded categories as
numeric inputs. For a final study, we should consider one-hot encoding or
learned embeddings and verify the feature preprocessing against the paper
and the eventual ZTE dataset.
