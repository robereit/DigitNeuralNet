function [newW, dW, newV, dV] = updateWts(W,dEdW,dWold,V,dEdV,dVold,eta,alpha)
% [newW, dW, newV, dV] = updateWts(W,dWold,V,dVold,eta,alpha)
% UPDATE WEIGHTS -- GRADIENT DESCENT WITH MOMENTUM
%
%   eta is the learning rate
%   alpha is the momentum decay coefficient
%
%   the weight update is  dwt = -eta*dEdWt + alpha*dWtOld
%
%   newW and newV are the network weights after the updates

dV = - eta*dEdV + alpha * dVold;
dW = - eta*dEdW + alpha * dWold;
newV = V + dV;
newW = W + dW;
