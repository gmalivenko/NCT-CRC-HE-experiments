# NCT-CRC-HE-experiments

#### 1. Overview [[Paper (in progress)]]()

Numerous deep learning-based solutions have been proposed for histopathological image analysis over the past years. While they usually demonstrate exceptionally high accuracy, one key question is whether their precision might be affected by low-level image properties not related to histopathology but caused by microscopy image handling and pre-processing. In this paper, we analyze a popular NCT-CRC-HE-100K colorectal cancer dataset used in numerous prior works and show that both this dataset and the obtained results may be affected by data-specific biases. The most prominent revealed dataset issues are inappropriate color normalization, severe JPEG artifacts inconsistent between different classes, and completely corrupted tissue samples resulting from incorrect image dynamic range handling. We show that even the simplest model using only 3 features per image (red, green and blue color intensities) can demonstrate over 50% accuracy on this 9-class dataset, while using color histogram not explicitly capturing cell morphology features yields over 82% accuracy. Moreover, we show that a basic EfficientNet-B0 ImageNet pretrained model can achieve over 97.7% accuracy on this dataset, outperforming all previously proposed solutions developed for this task, including dedicated foundation histopathological models and large cell morphology-aware neural networks. The NCT-CRC-HE dataset is publicly available and can be freely used to replicate the presented results. 

This repository provides the implementation of further improvement of the EfficientNet-based solution originally presented in [this paper (in progress)]() and the dataset analysis tools. 


#### 2. Prerequisites

- Python: torch, timm, scipy, numpy, opencv packages
- NVidia GPU


### 3. Data

1. Download the [CRC-VAL-HE-7K](https://zenodo.org/records/1214456) dataset and extract it to the ``data`` directory.
2. Download the [NCT-CRC-HE-100K](https://zenodo.org/records/1214456) (the same page) dataset and extract it to the ``data`` directory.


### 4. Training

Use `scripts/train_model.py` to train EfficientNet-B0 (avg) model.

Pretrained weights will be available at `releases` page.


### 5. Results

| Model                                              | Accuracy | 
|----------------------------------------------------|----------|
| [DeepCMorph](https://github.com/aiff22/DeepCMorph) | 96.99%   |
| EfficientNet-B0 model (avg, this repo)             | 97.73%   |
| Ensemble of 2×EfficientNet-B0 models (this repo)   | 98.33%   |


#### 6. License

Copyright (C) 2024 Andrey Ignatov, Grigory Malivenko. All rights reserved.

Licensed under the [CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

The code is released for academic research use only.


#### 7. Citation

```
...
```


#### 8. Any further questions?

```
Please contact Andrey Ignatov (andrey@vision.ee.ethz.ch) for more information
```