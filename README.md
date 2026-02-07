# DKECAlexCapsNet-MSClassification
This repository provides the official PyTorch implementation of DKECAlexCapsNet, a dynamic-kernel attention–enhanced CNN-CapsNet framework for single- and multi-channel microseismic signal classification, as described in our corresponding journal manuscript.

# Overview
The proposed framework integrates a CNN-CapsNet backbone with a Dynamic-Kernel Efficient Channel Attention (DKECA) module to improve feature discrimination under low-SNR conditions. The pipeline includes signal preprocessing, optional Gaussian-noise augmentation, model training, validation-based model selection, and final evaluation. The code is designed to be modular and reproducible, supporting both single-channel and multi-channel microseismic waveform inputs.

# Repository Contents
1.	data_preprocessing/: waveform normalization and segmentation scripts, datasetprocessing.py
2.	model configurations/: CNN-CapsNet and DKECA module definitions, DKECAlexCapsNet.py
3.	model training.py: training and validation pipeline, modeltraining.py
4.	evaluation scripts/: metric computation and logging utilities, test.py

# Environment
1.	Python 3.7
2.	PyTorch ≥ 1.8
3.	NumPy, SciPy, scikit-learn
4.	tqdm, tensorboard

# Usage
1.	Configure training parameters in configs/*.yaml
2.	Run training:
python train.py --config configs/single_channel.yaml
3.	Evaluate the trained model:
python test.py --checkpoint path_to_checkpoint

Due to data confidentiality constraints, raw microseismic waveforms are not included. The code supports direct integration of user-provided datasets following the documented input format.
