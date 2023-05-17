# DeepCPP: Neural Network Framework

DeepCPP is a comprehensive Neural Network Framework implemented entirely from scratch in C++. It contains several modularized classes for creating and training neural networks, using custom-built layers, optimizers, cost functions, learning rate schedulers, and basic algebraic operations. 

## Table of Contents

1. [About the Project](#about-the-project)
2. [Features](#features)
3. [Prerequisites & Installation](#prerequisites)
4. [Example Projects](#example-projects)
5. [Contributing & License](#contributing--license)
6. [Contact](#contact)

## About The Project

DeepCPP was built with the aim of providing a deeper understanding of the inner workings of neural networks. By coding all the elements from scratch, I have gained a detailed comprehension of the many intricacies of these powerful tools.

## Features

This framework includes:

- **Modularized Classes**: Users can easily define and manipulate layers, optimizers, and cost functions, amongst others.
- **Layers**: Fully connected (dense) layers, dropout layers for regularization, activation layers including ReLU, LeakyReLU, GELU, TanH, Sigmoid, Softmax, and Normal Sampling.
- **Optimizers**: Adam and SGD.
- **Cost Functions**: Mean Squared Error, Cross-Entropy, Binary Cross-Entropy.
- **Learning Rate Schedulers**: Allows changing the learning rate during training.
- **Gradient Clipping**: To prevent exploding gradients.
- **Algebraic Operations**: Basic operations such as addition, multiplication, matrix multiplication, etc, are implemented for comprehensive control over the model.
- **Multi-threading Support**: The framework uses OpenMP to speed up operations by using multi-threading.
- **Fully Implemented in C++**: Allowing for robust performance and deep customization.

## Prerequisites

To run this project, you will need:

- A C++ compiler that supports C++17 (gcc, clang, etc.)
- OpenMP library for multi-threading.
- Basic knowledge of C++ and neural networks

Clone the repository:
```bash
git clone https://github.com/kripxera1/DeepCPP.git
```

## Example Projects

To run the provided examples, you need to first unzip the training data located in the `/data` folder (unzip mnist.zip). You can compile and run the examples using `make` from the root directory.

To execute the examples, do so from the root directory with the following commands:

- `./bin/classifier`
- `./bin/vae`
- `./bin/denoising-vae`

The generated data will be stored in the `/images` folder.

The `/examples` folder contains several projects that demonstrate the usage of DeepCPP. These include:

- **MNIST Classifier**: A fully connected network trained to classify handwritten digits from the MNIST dataset.
- **MNIST VAE**: A Variational Autoencoder trained to generate images resembling the MNIST dataset.
- **MNIST Denoising VAE**: A Variational Autoencoder trained to denoise images from the MNIST dataset.

![MNIST Image + Normal Noise](https://github.com/kripxera1/DeepCPP/blob/main/noisy.jpg)
![Denoised MNIST Image using VAE](https://github.com/kripxera1/DeepCPP/blob/main/denoised.jpg)

## Contributing & License

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**. 

Distributed under the MIT License. See `LICENSE.txt` for more information.

## Contact

Antonio Manuel Escudero Vargas - antoniomanuelescuderovargas@gmail.com

Project Link: [https://github.com/kripxera1/DeepCPP](https://github.com/kripxera1/DeepCPP)
