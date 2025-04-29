# Experiment's history

* ClearML logs closed by default except one best model for each model type (check [README](README.md))
* For comparability I use constant architecture of NN based on my experience: UNet + MobileNetV3 to choose best approach
* Each approach were trained for 100 epoch

## 27.04.2025, exp1 Dmitrii
- Use spec_1 dataset based on [windowed FFT spectrogram](notebooks/Create_Datasets.ipynb)
- Implement combine loss with [Boundary Do-U Loss](https://arxiv.org/abs/2308.00220) 
- Use exp1 for model parameter tune (image size, batch size, workers, etc.)
- Setup simple augmentation to preserve spectrogram properties

### Pros
- new MobileNetv4 model
- settled model parameters
- high IoU (>0.95) and good visual mask prediction from earl epochs

### Cons
- 512 px suitable, but long training, switch to 256 px - loss the same so far
- new MobileNetv4 model

### Ideas
- try other spectra to compare
- use larger images for better results
- use combination of spectra
- change model and backbone architectures


---


## 27.04.2025, exp2 Dmitrii

- Use spec_2 dataset based on [windowed FFT spectrogram](notebooks/Create_Datasets.ipynb))
- Implement combine loss with [Boundary Do-U Loss](https://arxiv.org/abs/2308.00220) 
- Tune simple augmentation to preserve spectrogram properties in ulta-low res (64x64)

### Pros
- new MobileNetv4 model
- settled model parameters
- small images => fast training

### Cons
- low resolution of images, limited augmentations
- new MobileNetv4 model

### Ideas
- try other spectra to compare
- use combination of spectra
- change model and backbone architectures


---


## 27.04.2025, exp3 Dmitrii

- Use spec_3 dataset based on [simple stacking](notebooks/Create_Datasets.ipynb))
- Implement combine loss with [Boundary Do-U Loss](https://arxiv.org/abs/2308.00220)

### Pros
- new MobileNetv4 model
- settled model parameters
- fast tuning from early epochs

### Cons
- 512 px suitable, but long training, switch to 256 px - loss the same so far
- bigger information loose from reduce image resolution
- ultra-thin pattern on the images, may be too hard for the UNet+MobileNetV4
- new MobileNetv4 model

### Ideas
- try other spectra to compare
- use larger images for better results
- use combination of spectra
- change model and backbone architectures

---


## 27.04.2025, exp4 Dmitrii

- Use spec_4 dataset based on [Gaussian approximation with preserve init trace structure](notebooks/Create_Datasets.ipynb))
- Implement combine loss with [Boundary Do-U Loss](https://arxiv.org/abs/2308.00220)

### Pros
- interesting idea - Gaussian better fit the noise, but for signals give wider distributions
- one of the best visual representation of the traces
- new MobileNetv4 model
- settled model parameters
- fast tuning from early epochs

### Cons
- 512 px suitable, but long training, switch to 256 px 
- bigger information loose from reduce image resolution
- ultra-thin pattern on the images, may be too hard for the UNet+MobileNetV4
- new MobileNetv4 model

### Ideas
- try other spectra to compare, maybe use 1st approach not for pure trace, but for that ones?
- use larger images for better results
- use combination of spectra as one of the channel
- change model and backbone architectures
- initial masks preserve some spectra structures, interesting

---


## 27.04.2025, exp5 Dmitrii

- Use spec_5 dataset based on [classic 2D spectrogram](notebooks/Create_Datasets.ipynb))
- Implement combine loss with [Boundary Do-U Loss](https://arxiv.org/abs/2308.00220)

### Pros
- classic representation
- new MobileNetv4 model
- settled model parameters

### Cons
- 512 px suitable, but long training, switch to 256 px 
- bigger information loose from reduce image resolution
- ultra-thin pattern on the images, may be too hard for the UNet+MobileNetV4
- new MobileNetv4 model
- complicated masks, may be too much for UNet+MobileNetV4

### Ideas
- use larger images for better results
- change model and backbone architectures

---


## 28.04.2025, exp6 Dmitrii

- Use spec_4 dataset based on [Gaussian approximation with preserve init trace structure](notebooks/Create_Datasets.ipynb))
- Implement combine loss with [Boundary Do-U Loss](https://arxiv.org/abs/2308.00220)
- Try bigger resolution

### Pros
- interesting idea - Gaussian better fit the noise, but for signals give wider distributions
- one of the best visual representation of the traces
- new MobileNetv4 model
- settled model parameters
- fast tuning from early epochs

### Cons
- 512 px suitable, but long training, switch to 256 px 
- bigger information loose from reduce image resolution
- ultra-thin pattern on the images, may be too hard for the UNet+MobileNetV4
- new MobileNetv4 model

### Ideas
- try other spectra to compare, maybe use 1st approach not for pure trace, but for that ones?
- use larger images for better results
- use combination of spectra as one of the channel
- change model and backbone architectures
- initial masks preserve some spectra structures, interesting

---