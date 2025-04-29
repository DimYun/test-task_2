# First Break Identification Algorithm


## Overview

This repository contains the example of R&D development with code for an algorithm designed to automatically identify "first breaks" in seismic data. 
First breaks are the initial specific waves detected on a seismogram. The algorithm utilizes computer vision techniques to 
analyze seismic traces and 2D spectrograms.

The project was developed by Dr. Dmitrii Iunovidov, who has over 10 years of experience in R&D, machine learning, and 
innovative technology implementation in chemical industry.

* [Dmitrii's Website](https://dimyun.space/)
* [Dmitrii's LinkedIn](https://www.linkedin.com/in/dmitrii-iunovidov)


## Key Concepts

* **First Break:** The first arrival of seismic energy at a receiver.
* **Trace:** A 1D representation of seismic waves.
* **2D Spectrogram:** A visual representation of the frequency content of seismic data over time.


## Methodology

The project explores and implements several approaches for first break detection:

* Five types of graphical representations were studied.
* Four of these are based on single trace images.
* One approach uses a 2D spectrogram from each receiver.
* A windowed Fourier approximation (spectra set No. 1) was identified as a strong solution.
* A self-developed approach using Gaussian approximation for noise separation was also investigated.
* A neural network pipeline with a self-written loss function was implemented.
* A combined loss function with border attention was used for precise mask prediction.


## Results

* The developed algorithm achieved an average relative error of 2.44% (3.57% standard deviation) for first break detection.
* The windowed Fourier approximation approach can operate in real-time on CPU devices (39 fps).
* All approaches generally showed good performance (IoU > 0.95, error < 15%), except for the 2D spectrogram method, which had a higher error (33%) due to limited training.
* The solution can be used on CPU devices with real-time calculations (39 fps) and achieves 188 fps on a GPU.


## Limitations

The project faced the following limitations due to time constraints and data availability:

* Limited neural network architecture exploration (UNet with MobileNetV4 backbone).
* Limited training time for models (less than 13 epochs on average).
* Small image resolution (256x256 pixels, or 64x64 for some spectra).
* Use of a predefined combined loss function.
* Use of a subset of the available data for training (100,000 out of 9,000,000+ traces).
* Limited information about the physical nature of the data.
* Use of JPG images for spectra representation.


## Future Work

The following improvements are suggested for future work:

* Train the models for more than 100 epochs.
* Experiment with Fourier loss and other distance-based loss functions.
* Use all available data to improve generalization.
* Check visual data and masks for outliers.
* Explore Weivlet functions for signal approximation.
* Incorporate receiver group information into 2D spectrogram construction.
* Study alternative methods for filling missing first break values (e.g., median, distribution-based).
* Use PNG or other lossless image formats for spectra representation.


## References

* [Presentation: "The \"First Breaks\" identification algorithm" - Dr. Dmitrii Iunovidov]([Dmitrii-Iunovidov_first-break.pdf](Dmitrii-Iunovidov_first-break.pdf))
* Exploratory Data Analyses in [EDA.ipynb](notebooks/EDA.ipynb)
* Dataset creation in [Create_Datasets.ipynb](notebooks/Create_Datasets.ipynb)
* Solutions evaluation in [Models_Evaluation_and_Inference.ipynb](notebooks/Models_Evaluation_and_Inference.ipynb)


## Code Structure

* Min code repository [https://github.com/DimYun/test-task\_2](https://github.com/DimYun/test-task_2)
* Reports in Jupyter notebooks format: [notebooks](notebooks)
* Code of model: [src](src)
* Train initial module: [train.py](train.py)


## Makefile

* To install python environment - set the `SYSTEM_PYTHON` and `VENV` variables
* To apply linters - use `make lint`
* To train your model - setup [config.yaml](configs/config.yaml) and `make train`