function h = hiddenUnitSearch
%% load training and test data
% Note that the code assumes the input data is a DxN matrix, where
% D is the dimensionality of the data and N is the number of examples.
% The targets are of dimension nClasses x N.
load hw2v6.mat

inputs_train = traindata;
target_train = traintargets;


%% initialize the net structure.
numIn = size(inputs_train, 2);
numOut = 10;

%% Flag that checks whether or not we will evaluate the test
%% set
EvalTest = false;

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

%% BEGIN FINDING GOOD HIDDEN UNIT NUMBER
numRuns = 10;

ErrorsLastRun = zeros(1,NumEpochs);
ValidErrorsLastRun = zeros(1,NumEpochs);


e = exp(1);
runValidationErrors = zeros(numRuns,1);

%% Start with an estimate of the number of hidden units
%% Don't go beyond the upper bound or below the lower bound.
hidLb = 20;
hidUb = 200;
minHid = inf;
oldPotential = inf;

for numHid = hidLb:3:hidUb
    %% draw a random sample of size ValidSize for a validation set
    %% and remove those from the input set
    ValidSize = 300;

    a = randperm(size(traindata,  1));
    inputs_valid = traindata(a(1:ValidSize), :);
    inputs_train(a(1:ValidSize), :) = [];
    target_valid = traintargets(a(1:ValidSize), :);
    target_train(a(1:ValidSize), :) = [];
    potential = trainNetwork(inputs_train, target_train, inputs_valid, ...
        target_valid, NumEpochs, numIn, numHid, numOut, numRuns);
    if potential < oldPotential
        minHid = numHid;
        oldPotential = potential;
    end
end
fprintf(1, 'Optimal Hidden: %g\n', minHid);