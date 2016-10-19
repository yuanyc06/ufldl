function [cost,grad,features] = sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, ...
                                                            lambda, sparsityParam, beta, data)
% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Copy sparseAutoencoderCost in sparseAutoencoderCost.m from your
%   earlier exercise onto this file, renaming the function to
%   sparseAutoencoderLinearCost, and changing the autoencoder to use a
%   linear decoder.

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

M = size(data, 2);
z2 = W1 * data + b1 * ones(1, M);
a2 = sigmoid(z2);
z3 = W2 * a2 + b2 * ones(1, M);
a3 = z3; % Linear decoder
p = sparsityParam;
q = sum(a2, 2) / M;

cost = (1/M) * 0.5 * sum(sum((a3 - data).^2)) + ...
    (lambda/2) * (sum(sum(W1.^2)) + sum(sum(W2.^2))) + ...
    beta * sum(KL(p, q));
d3 = a3 - data; % Linear decoder
d2 = (W2' * d3 + beta * (-(p./q) + (1-p)./(1-q)) * ones(1, M)) .* a2 .* (1 - a2);

W2grad = d3 * a2' / M + lambda * W2;
W1grad = d2 * data' / M + lambda * W1;
b2grad = sum(d3, 2) / M;
b1grad = sum(d2, 2) / M;

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

function kl = KL(p, q)
    kl = p * log(p./q) + (1-p) * log((1-p)./(1-q));
end