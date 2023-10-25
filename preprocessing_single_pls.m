function [A,B]=single_pls(XTrain,yTrain,XTest,yTest,ncomp)

if(nargin<=4)
    ncomp=10;
end

XTrain=SNV(XTrain);
XTest=SNV(XTest);

[~,~,~,~,beta] = plsregress(XTrain,yTrain,ncomp);
A = [ones(size(XTrain,1),1) XTrain]*beta;
B = [ones(size(XTest,1),1) XTest]*beta;
