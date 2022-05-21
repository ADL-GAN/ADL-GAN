# ADL-GAN

This repository contains codes used to conduct experiments in ADL-GAN: Data Augmentation to Improve In-the-wild ADL Recognition using GANs paper.
We proposed GAN-based models to augment smartphone sensor data for Activity of Daily Living (ADL) recognition task. Since our implementation heavily depends on [blind] dataset, some codes are currently left out for double-blind review process. The current version of code is only working for Extrasensory dataset.

## ADL-transfer
python main.py ADL-transfer

## Subject-transfer
python main.py subject-transfer

## Dependency
* Librosa=0.7.2
* pytorch=1.3.1
* scikit-learn

The training configuration is in pre_processing.py
