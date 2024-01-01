## ----------------- Important Notice -----------------
# This script solely loads necessary functions into the environment. It does not
# directly execute functions. In the 'test_func()', the Iris dataset is used to
# test our neural network. To compute the misclassification rate on the
# test set, uncomment the last line and run it.

## ------------------- Overview ------------------------
# Introduction:
# This code includes functions setting up a simple neural network for
# classification. It uses stochastic gradient descent for training, a popular
# optimization method in machine learning.

# Principle of Training the Network:
# The network has some layers, First layer nodes are set to input data values.
# Output layer nodes predict the class of the input data. The network parameters
# control the links and transformations between layers. These are adjusted to
# minimize the loss function, optimizing input data class prediction.
# The code divides the Iris dataset into training and test data.
# Post-training with the training dataset, the model performs predictions
# on the test set and calculates the misclassification rate.


netup <- function(d) {
  # Initializes a neural network with random weights and biases.
  #
  # This function initializes the values of each nodes ,and weights and biases
  # for each layer in the neural network.

  # Parameters:
  # - d: A numeric vector specifying the number of nodes in each layer.
  #
  # Returns:
  # network: A list representing the neural network and it contains:
  # - h: Node values, initialized to zeros.
  # - W: Weights, randomly initialized.
  # - b: Biases, randomly initialized.
  # - dh, dW, db: Gradients for network parameters, initialized to zeros.
  # - sizes: The structure of the network, the same as 'd'.

  # Determine the number of layers in the network.
  num_layers <- length(d)

  # Initialize nodes to zeros.
  h <- lapply(d, function(dim) numeric(dim))

  # Initialize weights with U(0, 0.2) random deviates.
  W <- lapply(2:num_layers, function(idx) {
    matrix(runif(d[idx] * d[idx - 1], 0, 0.2), d[idx], d[idx - 1])
  })

  # Initialize biases(offset) randomly.
  b <- lapply(d[2:num_layers], function(dim) runif(dim, 0, 0.2))

  # Initialize gradients for node values (dh), weights (dW), and biases (db).
  dh <- lapply(d, function(dim) numeric(dim))
  dW <- lapply(2:num_layers, function(idx) matrix(0, d[idx], d[idx - 1]))
  db <- lapply(d[2:num_layers], function(dim) numeric(dim))

  # Return the constructed network as a list.
  return(list(h = h, W = W, b = b, dh = dh, dW = dW, db = db, sizes = d))
}


forward <- function(nn, inp) {
  # Performs a forward pass through a neural network.
  #
  # Given input values and initialized parameters of the neural network, this
  # function computes the values of the remaining nodes and updates the neural
  # network list.

  # Parameters:
  # - nn: A list representing the neural network.
  # - inp: A numeric vector representing the input to the neural network.
  #
  # Returns:
  # - nn: A list representing the neural network with updated
  #             node values ('h') after applying the forward pass.

  # Get the number of layers in the neural network.
  num_layers <- length(nn$sizes)

  # Initialize the first layer's nodes with the provided input.
  nn$h[[1]] <- inp

  # Compute node values for next layer.
  for (idx in 2:num_layers) {
    # Apply weighted sum and ReLU transformation for the current layer.
    nn$h[[idx]] <- nn$W[[idx - 1]] %*% nn$h[[idx - 1]] + nn$b[[idx - 1]]
    nn$h[[idx]][nn$h[[idx]] < 0] <- 0
  }

  # Return the updated network with the new node values.
  return(nn)
}

backward <- function(nn, k) {
  # Executes a backward pass through the neural network.
  #
  # This function compute the derivatives of the loss corresponding to output
  # class k for network. And add the derivatives w.r.t. the nodes, weights and
  # offsets to the network list.

  # Parameters:
  # - nn: A list representing the neural network.
  # - k: The true class label in the output layer's probability distribution.
  #
  # Returns:
  # - nn: Updated neural network (list) with computed gradients.

  # Determine the number of layers in the neural network.
  num_layers <- length(nn$sizes)

  # Ensure numerical stability with log-sum-exp for softmax calculation.
  max_h <- max(nn$h[[num_layers]])
  log_sum_exp <- max_h + log(sum(exp(nn$h[[num_layers]] - max_h)))

  # Initialize the gradient of the output layer using softmax derivative.
  nn$dh[[num_layers]] <- exp(nn$h[[num_layers]] - log_sum_exp)
  nn$dh[[num_layers]][k] <- nn$dh[[num_layers]][k] - 1

  # backward pass through the network.
  for (idx in (num_layers - 1):1) {
    # For current layer:
    # Calculate gradients. Apply ReLU derivative conditionally.
    d <- nn$dh[[idx + 1]] * (nn$h[[idx + 1]] > 0)
    # Compute gradients for nodes(dh), weights(dW), and biases(db).
    nn$dh[[idx]] <- crossprod(nn$W[[idx]], d)
    nn$db[[idx]] <- d
    nn$dW[[idx]] <- tcrossprod(d, nn$h[[idx]])
  }

  return(nn)
}

train <- function(nn, inp, k, eta = .01, mb = 10, nstep = 10000) {
  # Function to train a neural network using mini-batch gradient descent.
  #
  # This function implements mini-batch gradient descent to train the neural
  # network, updating its weights and biases iteratively over a specified number
  # of training steps.

  # Parameters:
  # - nn: A list representing the neural network.
  # - inp: Input data matrix, each row is an input sample.
  # - k: Vector of target class labels for the input data.
  # - eta: Learning rate, controls the update magnitude in gradient descent.
  # - mb: Mini-batch size, number of samples used per training iteration.
  # - nstep: Number of training iterations.
  #
  # Returns:
  # - nn: Trained neural network (list) with updated weights and biases.

  # Determine the number of layers in the neural network.
  num_layers <- length(nn$sizes)


  # Initialize cumulative gradients for weights and biases.
  dW <- lapply(2:num_layers, function(idx) {
    matrix(0, nn$sizes[idx], nn$sizes[idx - 1])
  })
  db <- lapply(nn$sizes[2:num_layers], function(dim) numeric(dim))

  for (epoch in 1:nstep) {
    # Random sampling to create a mini-batch for training.
    sub_idx <- sample(nrow(inp), mb)

    for (idx in 1:mb) {
      # For each sample in mini-batch:
      # Conduct forward and backward passes on the network.
      # Forward pass calculates node values, backward pass calculates gradients.
      backward_result <- backward(
        forward(nn, inp[sub_idx[idx], ]),
        k[sub_idx[idx]]
      )

      # Accumulate gradients from each sample in the mini-batch.
      if (idx == 1) {
        for (i in 1:length(db)) {
          db[[i]] <- backward_result$db[[i]]
          dW[[i]] <- backward_result$dW[[i]]
        }
      } else {
        for (i in 1:length(db)) {
          db[[i]] <- backward_result$db[[i]] + db[[i]]
          dW[[i]] <- backward_result$dW[[i]] + dW[[i]]
        }
      }
      # end of for-loop
    }

    # Average the gradients over the mini-batch.
    db <- lapply(db, function(gd) gd / mb)
    dW <- lapply(dW, function(gd) gd / mb)

    # Update the neural network's parameters using the averaged gradients.
    nn$b <- mapply(function(b, gd) b - eta * gd, nn$b, db, SIMPLIFY = FALSE)
    nn$W <- mapply(function(w, gd) w - eta * gd, nn$W, dW, SIMPLIFY = FALSE)
  }
  return(nn)
}

predict.nn <- function(nn, inp) {
  # Predicts class labels for input samples using a trained neural network.
  #
  # This function predicts the class labels for each input sample using the
  # forward pass of the neural network. What's more it includes input validation
  # to ensure that the provided input is a matrix.

  # Parameters:
  # - nn: A list representing the neural network.
  # - inp: Input data matrix. Each row is an input sample to be classified.
  #
  # Returns:
  # Vector of predicted class labels for each input sample.

  # Validate input.
  if (!is.matrix(inp)) {
    stop("Input must be a matrix.")
  }

  # Number of layers in the neural network.
  num_layers <- length(nn$sizes)

  # Predict the class for each sample.
  apply(inp, 1, function(input) {
    # Get the output of the last layer for the given input.
    output <- forward(nn, input)$h[[num_layers]]
    # Determine the class with the highest probability.
    which.max(output)
  })
}

test_func <- function(random_state = 123) {
  # Function to create and train a neural network using iris dataset,
  # and calculate misclassification rate on test data.

  set.seed(random_state)

  # 'iris' is a built-in dataset in R.
  # Represent the species of iris as numerical form.
  iris$Species <- as.integer(as.factor(iris$Species))

  # Divide the iris dataset into training set and test est.
  test_data <- iris[seq(5, nrow(iris), by = 5), ]
  train_data <- iris[-seq(5, nrow(iris), by = 5), ]

  # Extract the input data and target result labels for the training set.
  train_input <- as.matrix(train_data[, 1:4])
  train_target <- train_data$Species

  # Extract the test data and target result labels for the test set.
  test_input <- as.matrix(test_data[, 1:4])
  test_target <- test_data$Species

  # Set the number of layers.
  layer_sizes <- c(4, 8, 7, 3)

  # Create a neural network with specified layer sizes.
  neural_network <- netup(layer_sizes)

  # Train the neural network using the training set and target labels.
  neural_network <- train(neural_network, train_input, train_target)

  # Make predictions on test set.
  test_results <- predict.nn(neural_network, test_input)

  # Calculate misclassification rate.
  test_err_rate <- sum(test_results != test_target) / length(test_target)

  # Print the misclassification rate
  cat("Test set misclassification rate:", test_err_rate, "\n")
}

# system.time(test_func())
