function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% useful vector size information 
% (m = samples, n = features, s2 = hidden units, K = labels)
%
% Theta1 : [s2 x (n+1)]
% Theta2 : [K x (s2+1)]
% X      : [m x n] -> [m x (n+1)] with bias
% y      : [m x 1] -> [m x K] with refactor
% lambda : [1 x 1]
%
% J      : [1 x 1]

% ---------------------------
% PART 1: Forward Propagation

% append bias unit to input layer [m x (n+1)]
X = [ones(m, 1) X];
% compute the hidden layer z and activation [m x s2]
z_hidden = X * Theta1';
activation_hidden = sigmoid(z_hidden);
% append bias unit to hidden layer [m x (s2+1)]
activation_hidden = [ones(m, 1) activation_hidden];

% compute hypothesis [m x K]
hypothesis = sigmoid(activation_hidden * Theta2');

% refactor y to be vector of vectors (matrix) [m x K]
y_matrix = zeros(m, num_labels);
for index = 1:m
    y_matrix(index, y(index)) = 1;
end

% compute cost function [1 x 1]
J = (1 / m) * sum(sum(-y_matrix .* log(hypothesis) - ...
    (1 - y_matrix) .* log(1 - hypothesis)));

% generate regularized theta matrices with  theta_0 = 0 (bias) column
Theta1_reg = Theta1;
Theta2_reg = Theta2;
Theta1_reg(:, 1) = 0;
Theta2_reg(:, 1) = 0;
% compute regularized cost function [1 x 1]
J = J + lambda / (2 * m) * (sum(sum(Theta1_reg .^ 2)) + ...
    sum(sum(Theta2_reg .^ 2)));

% ------------------------
% PART 2: Back Propagation

% delta_h      : [m x K]
% delta_hidden : [m x s2] 

% compute error term for hypothesis [m x K]
delta_h = hypothesis - y_matrix;
% compute error term for hidden layer [m x s2]
delta_hidden = (delta_h * Theta2(:, 2:end)) .* sigmoidGradient(z_hidden);

% compute example gradient for each layer [s2 x (n+1)], [K x (s2 + 1)]
Delta1 = delta_hidden' * X;
Delta2 = delta_h' * activation_hidden;

% compute unregularized gradient for neural network
Theta1_grad = Delta1 / m;
Theta2_grad = Delta2 / m;

% ----------------------------
% PART 3: Regularized Gradient

Theta1_grad = Theta1_grad + lambda / m * Theta1_reg;
Theta2_grad = Theta2_grad + lambda / m * Theta2_reg;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
