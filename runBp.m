%% To run this program:
%%   First run initBp
%%   Then repeatedly call runBp until convergence.

ErrorsLastRun = zeros(1,NumEpochs);
ValidErrorsLastRun = zeros(1,NumEpochs);
startEpoch = totalNumEpochs + 1;
NValid = size(inputs_valid,1);
dWold = zeros(size(W));
dVold = zeros(size(V));

e = exp(1);
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
  
  %%%%%% Print out summary statistics at the end of the epoch %%%%%
  gradSize = norm([dEdV(:);dEdW(:)]);
  totalNumEpochs = totalNumEpochs + 1;
  ValidMSE = ValidError/NValid;
  if totalNumEpochs == 1
      startError = MSE;
  end
  ErrorsLastNumEpochs(1,epoch) = MSE;
  ValidErrorsLastNumEpochs(1,epoch) = ValidMSE;
  fprintf(1,'%d  TrainError=%f, ValidError=%f, |G|=%f\n',...
            totalNumEpochs,MSE,ValidMSE,gradSize);
end

fprintf(1, 'TrainError=%g, ValidError=%g', MSE, ValidMSE);
if EvalTest
    TestError = 0;
    NTest = size(inputs_test, 1);
    %% Evaluate the network on the Test Set
    for pat = 1:NTest
        %%%%% forward pass %%%%%
        X = [inputs_test(pat, :),[1]]';
        hidNetIn = V * X;
        hidAct = sigmoid(hidNetIn);
        hidActBias = [[hidAct]',[1]]';
        outNetIn = W * hidActBias;
        %% This has been changed to softmax
        normalize = sum(e.^outNetIn);
        outAct = (e.^outNetIn)/normalize;   %outlayer output
        target = target_test(pat, :);
        [m,i] = max(outAct);
        t = find(target, 1);
        TestError = TestError + (target(i) == 0);
    end;
    MTestError = TestError/NTest;
    fprintf(1, ', TestError=%g\n', MTestError)
else
    fprintf(1, '\n');
end

clf; 
if totalNumEpochs > minEpochsPerErrorPlot
  epochs = [1:totalNumEpochs];
end

%%%%%%%%% Plot the learning curve for the training set patterns %%%%%%%%%
errorsPerEpoch(1,startEpoch:totalNumEpochs) = ErrorsLastNumEpochs;
subplot(2,1,1), ...
  axis([1 max(minEpochsPerErrorPlot,totalNumEpochs) 0 startError]),  hold on, ...
  plot(epochs(1,1:totalNumEpochs),errorsPerEpoch(1,1:totalNumEpochs)),...
  title('Misclassification Error % on the Training Set'), ...
  xlabel('Learning Epoch'), ...
  ylabel('Error');


%%%%%%%%% Plot the learning curve for the test set patterns %%%%%%%%%
ValidErrorsPerEpoch(1,startEpoch:totalNumEpochs) = ValidErrorsLastNumEpochs;
subplot(2,1,2), ...
  axis([1 max(minEpochsPerErrorPlot,totalNumEpochs) 0 startError]),  hold on, ...
  plot(epochs(1,1:totalNumEpochs),ValidErrorsPerEpoch(1,1:totalNumEpochs)), ...
  title('Misclassification Error % on the Validation Set'), ...
  xlabel('Learning Epoch'), ...
  ylabel('Error');


