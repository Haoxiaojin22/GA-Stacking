function [A,B]=single_gpr(XTrain,yTrain,XTest,yTest)

sigma0 = std(yTrain);
sigmaF0 = sigma0;
d = size(XTrain,2);
sigmaM0 = 10*ones(d,1);
kparams0 = [sigmaM0;sigmaF0];

gprMdl = fitrgp(XTrain,yTrain, ...
    'KernelFunction','ardsquaredexponential', ...
    'KernelParameters',kparams0, ...
    'Sigma',sigma0);

A=predict(gprMdl,XTrain);
B=predict(gprMdl,XTest);