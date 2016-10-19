function numgrad = computeNumericalGradient_4D(J, theta)
% numgrad = computeNumericalGradient(J, theta)
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta. 
  
% Initialize numgrad with zeros
numgrad = zeros(size(theta));

%% ---------- YOUR CODE HERE --------------------------------------
% Instructions: 
% Implement numerical gradient checking, and return the result in numgrad.  
% (See Section 2.3 of the lecture notes.)
% You should write code so that numgrad(i) is (the numerical approximation to) the 
% partial derivative of J with respect to the i-th input argument, evaluated at theta.  
% I.e., numgrad(i) should be the (approximately) the partial derivative of J with 
% respect to theta(i).
%                
% Hint: You will probably want to compute the elements of numgrad one at a time. 

EPSILON = 1e-4;
% L = length(theta);
% E = eye(L) * EPSILON;
% for i = 1:L
%     numgrad(i) = (J(theta + E(:, i)) - J(theta - E(:, i))) / (2 * EPSILON);
% end

[h, w, ch, b] = size(theta);
for i = 1:b
    for j = 1:ch
        for m = 1:h
            for n = 1:w
                theta1 = theta;
                theta2 = theta;
                theta1(m,n,j,i) = theta1(m,n,j,i) + EPSILON;
                theta2(m,n,j,i) = theta2(m,n,j,i) - EPSILON;
                numgrad(m,n,j,i) = (J(theta1) - J(theta2)) / (2 * EPSILON);
            end
        end
    end
end


%% ---------------------------------------------------------------
end
