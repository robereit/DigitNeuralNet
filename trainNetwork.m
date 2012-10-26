function potential = trainNetwork(inputs_train, target_train, inputs_valid, ...
    target_valid, NumEpochs, numIn, numHid, numOut, numRuns)

beta = 0.8;
eta = 0.001;
alpha = 0.35;

e = exp(1);
for i = 1:numRuns
    %% train the network using inputs_train and return the error
    %% on a validation set defined by inputs_valid
    %%% make random initial weights smaller, and include bias weights
    V = 0.1 * (rand(numHid,numIn+1) - ones(numHid,numIn+1) * .5);
    W = 0.1 * (rand(numOut,numHid+1) - ones(numOut,numHid+1) * .5);

    ErrorsLastRun = zeros(1,NumEpochs);
    ValidErrorsLastRun = zeros(1,NumEpochs);
    NValid = size(inputs_valid,1);
    dWold = zeros(size(W));
    dVold = zeros(size(V));

    for epoch = 1:NumEpochs
        %%%%% Calculate the gradient of the objective function %%%
        [dEdW,dEdV,MSE] = calcGrad(inputs_train,target_train,W,V);

        %%%%% Update the weights at the end of the epoch %%%%%%
        [W, dW, V, dV] = updateWts(W,dEdW,dWold,V,dEdV,dVold,eta,alpha);
        dWold = dW;   dVold = dV;

        %%%%% Test network's performance on the validation patterns %%%%%
        ValidError = 0;
        for pat = 1:NValid
            %%%%% forward pass %%%%%
            X = [inputs_valid(pat, :),[1]]';
            hidNetIn = V * X;
            hidAct = sigmoid(hidNetIn);
            hidActBias = [[hidAct]',[1]]';
            outNetIn = W * hidActBias;
            %% This has been changed to softmax
            normalize = sum(e.^outNetIn);
            outAct = (e.^outNetIn)/normalize;   %outlayer output
            target = target_valid(pat, :);
            [m,i] = max(outAct);
            ValidError = ValidError + (target(i) == 0);
        end;
        gradSize = norm([dEdV(:);dEdW(:)]);
    end
    runValidationErrors(i) = ValidError/NValid;
end
varErr = var(runValidationErrors);
minErr = min(runValidationErrors);
potential = beta*minErr + (1-beta)*varErr;
fprintf(1, 'NumHid: %g, Potential:%g\n', numHid, potential);


