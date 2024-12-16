# **DCGAN: Deep Convolutional Generative Adversarial Network**

## Overview

This project implements a **Deep Convolutional Generative Adversarial Network (DCGAN)** for generating realistic images based on the **Fashion MNIST** dataset. The model learns to produce synthetic images that resemble the real Fashion MNIST images, which include items such as clothing, shoes, and accessories. The implementation demonstrates how GANs can be used for image synthesis, creative generation, and data augmentation.

## Features

- **Dataset**: Fashion MNIST with 60,000 training images and 10,000 test images, each 28x28 pixels in grayscale, representing 10 clothing categories.
- **Model Type**: Deep Convolutional Generative Adversarial Network (DCGAN).
- **Generator**: Takes 100-dimensional random noise as input and generates synthetic images of size 64x64 or 128x128 pixels using dense and transposed convolution layers. Activations used: **ReLU** (hidden) and **Tanh** (output).
- **Discriminator**: Classifies images as real (from Fashion MNIST) or fake (generated). Uses convolutional layers and Leaky ReLU activation. The final output layer uses **Sigmoid** for binary classification (real vs. fake).
- **Training Process**: 
  - **Epochs**: 50-200.
  - **Optimizer**: Adam with a learning rate of ~0.0002.
  - **Mini-batch size**: 64.
  - **Loss function**: Binary Cross-Entropy for both the generator and discriminator.

## Sprint Features

### Sprint 1: Dataset Preparation
- Preprocess the **Fashion MNIST** dataset by normalizing the images and converting them into a suitable format for training.
- **Deliverable**: Clean and preprocessed dataset ready for training.

### Sprint 2: Model Architecture Design
- Design the generator and discriminator networks using convolutional layers, ReLU, and Tanh activation functions.
- **Deliverable**: Initial model architecture for both the generator and discriminator.

### Sprint 3: Model Training
- Train the DCGAN using the preprocessed dataset. The generator and discriminator are trained in an adversarial manner.
- **Deliverable**: Trained model that can generate realistic synthetic images.

### Sprint 4: Image Generation and Evaluation
- Generate synthetic images and evaluate the quality using visual inspection and performance metrics.
- **Deliverable**: High-quality synthetic images that resemble the Fashion MNIST dataset.

## Conclusion

The DCGAN model successfully generates realistic images of clothing that resemble those in the Fashion MNIST dataset. This project demonstrates the power of GANs in unsupervised learning for image generation. The DCGAN architecture can be further enhanced by training on higher-resolution datasets or experimenting with more complex models for better image generation.
