function [A,B]=single_svm(XTrain,yTrain,XTest,yTest)
[a,b]=size(XTrain);
% if b>1
%     XTrain=diff(XTrain,1,2);
%     XTest=diff(XTest,1,2);
% end

Mdlsvm = fitrsvm(XTrain,yTrain, ...
    'KernelFunction','gaussian', ...
    'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions', ...
    struct('Optimizer','bayesopt','Verbose',0,'ShowPlots',false));
A=predict(Mdlsvm,XTrain);
B=predict(Mdlsvm,XTest);