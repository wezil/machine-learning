function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% useful vector size information 
% (m = examples, n = features, K = classes, s2 = hidden units):
% Theta1: [s2 x (n+1)]
% Theta2: [K x (s2+1)]
% X: [m x (n+1)]
%
% p: [m x 1]

% append a^(1)_0 = 1 for all examples [m x (n+1)]
X = [ones(m, 1) X];

% compute hidden layer a^(2) matrix values [m x s2]
a_2 = sigmoid(X * Theta1');
% calculate number of hidden units s^(2)
s_2 = size(a_2, 1);
% append a^(2)_0 = 1 in the hidden layer [m x (s2+1)]
a_2 = [ones(s_2, 1) a_2];

% now calculate output layer hypothesis [m x K]
hypothesis = sigmoid(a_2 * Theta2');
% use max function to determine which digit (class) each example is
[y, p] = max(hypothesis, [], 2);


% =========================================================================


end
