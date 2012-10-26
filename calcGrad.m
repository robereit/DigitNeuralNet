function [dEdW,dEdV,MSE] = calcGrad(inputs_train, target_train,W,V)
% [dEdW,dEdV,MSE] = calcGrad(trainPats,W,V)
% BACKPROP CALCULATION OF GRADIENT OF SQUARED ERROR
%   n is the number of input units
%   h is the number of hidden units
%   m is the number of output units
%   N is the number of training cases
%   inputs_train is an n x N matrix of inputs
%   target_train is an m x N matrix of outputs
%   V is h x (n+1), giving the weights to the hidden units and their biases.
%   W is m x (h+1), giving the weights to the output units and their biases.
%
%   dEdW and dEdV are the gradients of the cross entropy function
%          with respect to the weights W and V
%
%   MSE is the normalized value of the error, calculated using cross
%   entropy
%
%   NB: THIS ASSUMES A ONE HIDDEN LAYER NET WITH SIGMOID UNITS

N  = size(inputs_train, 1); 
n = size(V,2);
h = size(V,1); 
m = size(W,1); 

sumError = 0;
dEdW = zeros(size(W));
dEdV = zeros(size(V));

%% For convenience, define e
e = exp(1);
for pat = 1:N  
    %%%%% forward pass determines what the outputs are on this example %%%%%
    X = [inputs_train(pat,:),[1]]';
    
    hidNetIn = V * X;
    hidAct = sigmoid(hidNetIn);    %hidden layer output
    hidActBias = [[hidAct]',[1]]'; %take bias as fixed input
    outNetIn = W * hidActBias;
    %% This has been changed to softmax
    normalize = sum(e.^outNetIn);
    outAct = (e.^outNetIn)/normalize;   %outlayer output
    
    %%%%% backward pass updates the gradient %%%%%
    target = target_train(pat, :);
    %% change in gradient of the error contributed by this example
    outDel = (outAct - target') * hidActBias';
    % outDel = 2 * error .* outAct .* (1-outAct);
    dEdW = dEdW + outDel;
    hidDel = (W' * outDel) * (hidActBias .* (1-hidActBias));
    dEdV = dEdV + hidDel(1:h,:) * X';
    %% Find the maximum probability output by the model
    [m, i] = max(outAct);
    t = find(target, 1);
    %% Compare it to the actual test value
    sumError = sumError + (target(i) == 0);
end
MSE = sumError / N;
