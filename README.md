# a-simple-neural-network-for-classification

## Introduction
This repository contains an R script for setting up and training a simple neural network. The code is designed for classification tasks and uses stochastic gradient descent as the optimization method. 

## Overview

### Script Structure
- Initialization Function (`netup`)

  + Initializes a neural network with random weights and biases.
  + Parameters include a vector specifying the number of nodes in each layer.

- Forward Propagation Function (`forward`)

  + Performs a forward pass through the neural network.
  + Computes the values of the nodes given the input.
- Backward Propagation Function (`backward`)

  + Executes a backward pass through the neural network.

- Calculates gradients for loss optimization.
Training Function (`train`)

  + Implements mini-batch gradient descent for network training.
  + Parameters include the neural network, input data, target labels, learning rate, mini-batch size, and number of training iterations.

- Prediction Function (`predict.nn`)

  + Predicts class labels for input samples using the trained neural network.

- Test Function (`test_func`)

  + Demonstrates the creation, training, and evaluation of the network using the Iris dataset.
  + Calculates and prints the misclassification rate on the test set.


### Usage Instructions
1. Setup
   - Ensure that R is installed on your system.
Clone this repository to your local machine.
1. Running the Script
   - Open the script in an R environment (like RStudio).
   - Load the script into your R environment.
   - Call `test_func()` to run the test procedure. This will create a neural network, train it using the Iris dataset, and output the misclassification rate.
2. Customization
   - Modify the `layer_sizes` in the `test_func` to experiment with different network structures.
   - Change the dataset or modify the neural network configuration as needed for different classification tasks.

### Important Notes
- This script is designed for educational purposes and to demonstrate basic concepts in neural network implementation.
- The Iris dataset is used as a sample dataset for testing.
- To compute the misclassification rate on the test set, ensure the last line in `test_func()` is uncommented and run.

### Dependencies
- R environment for running the script.
- No additional libraries are required as the script uses basic R functions.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
