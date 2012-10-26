clear all

%% load training and test data
% Note that the code assumes the input data is a DxN matrix, where
% D is the dimensionality of the data and N is the number of examples.
% The targets are of dimension nClasses x N.
load hw2v6.mat

inputs_train = traindata;
target_train = traintargets;
%% draw a random sample of size ValidSize for a validation set
%% and remove those from the input set
ValidSize = 300;

a = randperm(size(traindata, 1));
inputs_valid = traindata(a(1:ValidSize), :);
inputs_train(a(1:ValidSize), :) = [];
target_valid = traintargets(a(1:ValidSize), :);
target_train(a(1:ValidSize), :) = [];
inputs_test = testdata;
target_test = testtargets;

%% Flag that checks whether or not we will evaluate the test
%% set
EvalTest = true;

eta = 0.001;  %% the learning rate 
alpha = 0.35;   %% the momentum coefficient

NumEpochs = 100; %% number of learning epochs (number of passes through the
                 %% training set) each time runbp is called.

totalNumEpochs = 0; %% number of learning epochs so far. This is incremented 
                    %% by numEpochs each time runbp is called.

%%% For plotting learning curves:
minEpochsPerErrorPlot = 200;
errorsPerEpoch = zeros(1,minEpochsPerErrorPlot);
TestErrorsPerEpoch = zeros(1,minEpochsPerErrorPlot);
epochs = [1:minEpochsPerErrorPlot];

%% initialize the net structure.
numIn = size(inputs_train, 2);
numHid = 113;
numOut = 10; 

%%% make random initial weights smaller, and include bias weights
V = 0.1 * (rand(numHid,numIn+1) - ones(numHid,numIn+1) * .5);
W = 0.1 * (rand(numOut,numHid+1) - ones(numOut,numHid+1) * .5);