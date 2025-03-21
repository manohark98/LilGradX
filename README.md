# LilGradX 

LilGradX is a lightweight neural network library built entirely from scratch in Python . This project is designed to help beginners understand the core principles of neural networks by implementing the fundamental building blocks like tensors with automatic differentiation, neurons, layers, loss functions, and optimizers.

In this version, training and testing have been separated into distinct scripts. The model parameters are saved in a JSON file after training, and then reloaded during testing for inference.

---
## Upcoming updates:
- Weights and bias initialization
- Batch Normalization 

## Table of Contents

- [Features](#features)
- [Folder Structure](#folder-structure)
- [Detailed File Descriptions](#detailed-file-descriptions)
  - [Tensor Module (`tensor.py`)](#tensor-module-tensorpy)
  - [Loss Functions (`losses/losses.py`)](#loss-functions-losseslossespy)
  - [Neural Network Components (`ll/`)](#neural-network-components-ll)
  - [Dataset Module (`dataset.py`)](#dataset-module-datasetpy)
  - [Training Script (`train.py`)](#training-script-trainpy)
  - [Testing Script (`test.py`)](#testing-script-testpy)


---

## Features

- **Custom Autograd Engine:**  
  Implemented a `Value` class that supports automatic differentiation (backpropagation) with overloaded arithmetic operators.
  
- **Neural Network Fundamentals:**  
  Build your network using neurons, layers, and a multi-layer perceptron (MLP) structure.
  
- **Loss Functions:**  
  Supports both Negative Log Likelihood (NLL) and Mean Squared Error (MSE) loss functions.
  
- **Optimizer:**  
  Uses the Adam optimizer for adaptive learning rate adjustments during training.
  
- **Dataset Handling:**  
  Provides a simple interface for loading, cleaning, normalizing, and splitting CSV data.
  
- **Separation of Concerns:**  
  Training and testing scripts are separated for clarity and modularity.
  
- **JSON Model Saving:**  
  Saves only the model configuration and numerical parameters  into a JSON file, which can later be loaded for testing or inference.

---

## Folder Structure
LilGradX/ \
│── lilgradx/ \
│   ├── losses/ \
│   │   ├── losses.py \
│   ├── ll/ \
│   │   ├── __init__.py \
│   │   ├── activations.py \
│   │   ├── layer.py \
│   │   ├── loss.py \
│   │   ├── mlp.py \
│   │   ├── neuron.py \
│   │   ├── optimizer.py \
│   ├── __init__.py \
│   ├── dataset.py \
│   ├── tensor.py \
│── train.py \
│── test.py \

---

## Detailed File Descriptions

### Tensor Module (`tensor.py`)

The `Value` class is the heart of the LilGradX . It encapsulates a scalar value along with its gradient and the backward propagation function. the following key functionalities included:
- **Arithmetic Operations:** Overloaded `+`, `*`, `-`, `/`, and power operations.
- **Activation Functions:** Implemented functions like `tanh()`, `exp()`, `log()`, and `leaky_relu()`.
- **Backpropagation:** The `backward()` method performs backpropagation in the computational graph and propagates gradients through it.

### Loss Functions (`losses/losses.py`)

This module implemented two loss functions:
- **`nll_loss(probs, target_index)`**:  
  Calculates the Negative Log Likelihood loss by taking the log of the softmax probability corresponding to the target class.
- **`mse_loss(outputs, targets)`**:  
  Computes the Mean Squared Error loss by averaging the squared differences between predictions and targets.

### Neural Network Components (`ll/`)

This directory contains the core building blocks of your neural network:

- **`activations.py`**:  
  Defines the `SoftmaxLayer` to normalize the logit outputs of nueral layer  into a probability distribution. It subtracts the maximum logit to bring the  numerical stability.
  
- **`layer.py`**:  
  Implements the `Layer` class, which is essentially a collection of `Neuron` objects. The layer processes an input vector and returns the outputs.
  
- **`neuron.py`**:  
  The `Neuron` class represents a single neuron with randomly initialized weights and bias. It computes the weighted sum of inputs and applies a leaky ReLU activation.
  
- **`mlp.py`**:  
  The `MLP` (Multi-Layer Perceptron) class stacks multiple layers to form a full network. After passing the input through all layers, it applies the softmax activation to produce class probabilities.
  
- **`optimizer.py`**:  
  Implements the Adam optimizer. It updates each parameter using adaptive learning rates based on first and second moment estimates of gradients.


### Dataset Module (`dataset.py`)

The `Dataset` class handles:
- **Data Loading:** Reads CSV files using Pandas.
- **Data Cleaning:** Drops unwanted columns and missing values.
- **Data Preprocessing:** Converts categorical targets into numerical values, normalizes features, and splits data into training and testing sets using scikit-learn.

### Training Script (`train.py`)

The training script performs the following tasks:
1. **Data Preparation:**  
   Loads and preprocesses the data from a CSV file.
2. **Model Initialization:**  
   Constructs an MLP with the given number of input features and output features.
3. **Training Loop:**  
   Iterates over the training data for a set number of epochs, performing forward passes, loss computation (either NLL or MSE), backpropagation, and parameter updates using the Adam optimizer.
4. **Model Saving:**  
   Instead of pickling (which has issues with lambda functions), the script extracts and saves only the numerical values (from the `.data` attribute) along with the network configuration (input size and layer sizes) into a JSON file (`model_state.json`).

### Testing Script (`test.py`)

The testing script is responsible for:
1. **Model Loading:**  
   Trained model parameters will be loaded from json file .
2. **Inference:**  
   Reconstructs the MLP using the saved configuration, assigns the saved parameter values, and uses the model to perform predictions on the test set.
3. **Evaluation:**  
   Computes test accuracy and displays performance metrics, including a confusion matrix and a detailed classification report using scikit-learn.

