function [A,B]=single_rf(XTrain,yTrain,XTest,yTest,treeNum,nodeNum)
%调用随机森林

if(nargin<=4)
    treeNum=100;
    nodeNum=30;
end

[XTrain,XTest]=MSC(XTrain,XTest);

isCategorical = false(size(XTrain,2),1);% Categorical variable flag
b= TreeBagger(treeNum,XTrain,yTrain, ...
    'Method','R','OOBPredictorImportance','On',...
    'CategoricalPredictors',isCategorical,...
    'MinLeafSize',nodeNum);

A=predict(b,XTrain);
B=predict(b,XTest);
