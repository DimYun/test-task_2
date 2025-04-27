# Experiment's history

* ClearML logs closed by default except one best model for each model type (check [README](README.md))
* For comparability I use constant architecture of NN based on my experience: UNet + MobileNetV3 to choose best approach
* Each approach were trained for 100 epoch


## 27.04.2025, exp1 Dmitrii

- Use spec_1 dataset based on [windowed FFT spectrogram](notebooks/create_datasets.ipynb)

### Pros
- new MobileNetv4 model

### Cons
- 512 px siutable, but long training, switch to 256 px - loss the same so far
- first mask is pretty random, lack of spectra info
- new Mobile Netv4 model

### Идеи на будущее
- попробовать уменьшить маски (erode)
- попробовать лосс для сохранения границ
- уменьшить маски с помощью circle erode
- попробовать другие архитектуры


---
