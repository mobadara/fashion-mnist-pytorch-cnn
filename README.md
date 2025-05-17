# Fashion MNIST Classification with PyTorch CNN

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Fashion MNIST Example](assets/fashion-mnist-example.png)

## Overview

This project demonstrates my proficiency in Deep Learning using PyTorch by building and training a Convolutional Neural Network (CNN) to classify images from the Fashion MNIST dataset. The Fashion MNIST dataset is a popular benchmark in machine learning, consisting of grayscale images of 10 different fashion items. This repository contains the complete code for data loading, model definition, training, and evaluation, designed to run seamlessly on Google Colab.

This project was undertaken to showcase my skills in:

* **PyTorch:** Utilizing PyTorch for tensor manipulation, dataset handling, model building, and training loops.
* **Convolutional Neural Networks (CNNs):** Designing and implementing CNN architectures suitable for image classification tasks.
* **Deep Learning Fundamentals:** Understanding and applying concepts like forward and backward propagation, loss functions, and optimizers.
* **Data Handling:** Loading and preprocessing image data using `torchvision`.

## Author

**Muyiwa Obadara** ([mobadara](https://github.com/mobadara))

## Table of Contents

* [Overview](#overview)
* [Author](#author)
* [Table of Contents](#table-of-contents)
* [Dataset](#dataset)
* [Project Structure](#project-structure)
* [Getting Started](#getting-started)
    * [Prerequisites](#prerequisites)
    * [Setup in Google Colab](#setup-in-google-colab)
* [Usage](#usage)
* [Model Architecture](#model-architecture)
* [Results](#results) (Will be updated after training)
* [Future Work](#future-work)
* [License](#license)
* [Acknowledgments](#acknowledgments)

## Dataset

The [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset is used in this project. It contains 70,000 grayscale images of fashion products from 10 categories, with 7,000 images per category. The dataset is split into a training set of 60,000 images and a test set of 10,000 images. The 10 classes are:

1.  T-shirt/top
2.  Trouser
3.  Pullover
4.  Dress
5.  Coat
6.  Sandal
7.  Shirt
8.  Sneaker
9.  Bag
10. Ankle boot

## Project Structure

For Google Colab, the project structure will be simpler:

```
├── assets/
│   └── fashion-mnist-example.png  # Example image
├── fashion-mnist-cnn.ipynb         # Main Jupyter Notebook containing the code
├── model.py                      # (Optional) Defines the CNN model architecture
├── train.py                      # (Optional) Script for training the model
├── evaluate.py                   # (Optional) Script for evaluating the trained model
├── README.md                     # This file
└── requirements.txt
```

You can either have everything in a single notebook (`fashion_mnist_cnn.ipynb`) or split the model, training, and evaluation logic into separate `.py` files that can be imported into the notebook.

## Getting Started

### Prerequisites

* A Google account (to use Google Colab).
* Basic familiarity with Python.

### Setup in Google Colab

1.  Open a new Google Colab notebook.
2.  You can either:
    * Upload all the project files (including the notebook, Python scripts if you create them, the `assets` folder, and `requirements.txt`) to the Colab environment.
    * Alternatively, you can clone this GitHub repository directly into your Colab environment using:
        ```python
        !git clone [https://github.com/mobadara/fashion-mnist-pytorch-cnn.git](https://github.com/mobadara/fashion-mnist-pytorch-cnn.git)
        cd fashion-mnist-pytorch-cnn
        ```
3.  If you have a `requirements.txt` file, you can install the necessary libraries by running:
    ```python
    !pip install -r requirements.txt
    ```

## Usage

If you're using a single notebook (`fashion-mnist-cnn.ipynb`), simply open it in Google Colab and run the cells sequentially.

If you've separated the code into `.py` files, you can import and use them within your notebook. For example:

```python
# In your notebook
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# If model definition is in model.py
# from model import Net

# If training logic is in train.py
# from train import train_model

# If evaluation logic is in evaluate.py
# from evaluate import evaluate_model

# Your data loading and preprocessing code here (as you generated earlier)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
class_names = ['T-Shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

# Define your CNN model directly in the notebook or in model.py
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example training loop (you might put this in train.py)
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            # ... (your training code) ...
            pass
    print("Training finished.")

# Example evaluation function (you might put this in evaluate.py)
def evaluate_model(model, test_loader):
    # ... (your evaluation code) ...
    pass

train_model(model, train_loader, criterion, optimizer, epochs=5)
evaluate_model(model, test_loader)
```

## Model Architecture
(A detailed description of the CNN architecture you implemented will go here. For example, based on the Net class above:)

The CNN architecture consists of:

A 2D convolutional layer with 1 input channel (grayscale), 32 output channels, a kernel size of 3, and padding of 1, followed by ReLU activation.
A max-pooling layer with a kernel size of 2 and a stride of 2.
Another 2D convolutional layer with 32 input channels, 64 output channels, a kernel size of 3, and padding of 1, followed by ReLU activation.
Another max-pooling layer with a kernel size of 2 and a stride of 2.
A fully connected (linear) layer mapping the flattened features (64 * 7 * 7) to 128 output units, followed by ReLU activation.
A final fully connected (linear) layer mapping 128 input units to 10 output units (one for each class).

## Results
(This section will be updated with the performance metrics of your trained model on the test set, such as accuracy, precision, recall, and F1-score. You might also include a few examples of correctly and incorrectly classified images.)

Test Accuracy: ...

## Future Work
Experiment with different CNN architectures (e.g., adding more layers, using different activation functions).
Implement techniques to improve performance, such as data augmentation and dropout.
Visualise the learned filters of the convolutional layers.

## License
This project is licensed under the MIT License.

## Acknowledgments
The Fashion MNIST dataset provided by Zalando Research.
The PyTorch library for providing the necessary tools for building and training neural networks.
