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
%         Theta2_grad, respectively. After implementing Part 2, you can check

X=[ones(m,1),X];
n=size(Theta2(:,1));
hypothesis=zeros(m,n(1));

for c=1:m
    
    v=X(c,:);
    
    z1=Theta1*v';
    
    a1=sigmoid(z1);
    
    a1=[1;a1];
    
    z2=Theta2*a1;
    
    a2=sigmoid(z2);
    
    hypothesis(c,:)=a2;
    
end

yp=zeros(size(hypothesis));

for d = 1 : m
    
    index=y(d);
    
    yp(d,index)=1;
    
end

for i=1:m
    
tempvec1=yp(i,:);
tempvec2=hypothesis(i,:);

temp1 = -tempvec1.*log(tempvec2);
temp2 = -(1-tempvec1).*log(1-tempvec2);

preCost(i)= sum(temp1+temp2);
    
end

%J=(1/m)*sum(preCost);

[rows1,cols1]=size(Theta1);
[rows2,cols2]=size(Theta2);

theta1=Theta1(:,2:cols1);
theta2=Theta2(:,2:cols2);

regterm1= sum((theta1.^2),'all');
regterm2= sum((theta2.^2),'all');


regularizationTerm= (lambda/(2*m))*(regterm1+regterm2);
J=(1/m)*sum(preCost) + regularizationTerm;

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
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
 
GradDelta_1 = Theta1_grad;
GradDelta_2 = Theta2_grad;

for t=1:m
    A1=X(t,:);
    
    Z2=Theta1*A1';
    
    A2=sigmoid(Z2);
    
    A2=[1;A2];
    
    Z3=Theta2*A2;
    
    A3=sigmoid(Z3);
    
    delta_3 = A3 - (yp(t,:)');
    
    delta_2 = ((Theta2')*delta_3).*sigmoidGradient([1;Z2]);
    
    delta_2 = delta_2(2:end);
    
    GradDelta_1 = GradDelta_1 + delta_2*(A1);
    
    GradDelta_2 = GradDelta_2 + delta_3*(A2');
    
    
end

regTheta1 = Theta1;
regTheta1(:,1) = 0;
regTheta1 = (lambda/m)*regTheta1;

regTheta2 = Theta2;
regTheta2(:,1) = 0;
regTheta2 = (lambda/m)*regTheta2;

Theta1_grad = (1/m)*GradDelta_1 + regTheta1;

Theta2_grad = (1/m)*GradDelta_2 + regTheta2;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
