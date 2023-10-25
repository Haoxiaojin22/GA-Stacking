function [A,B]=single_Adaboost(XTrain,yTrain,XTest,yTest)

ABtreeNum=200;
ABnodeNum=8;
learnRate=0.01;
% learnRate=0.02;
% [XTrain,XTest]=MSC(XTrain,XTest);

t = templateTree('MaxNumSplits',ABnodeNum,'Surrogate','on');
RTreeEns=fitensemble(XTrain,yTrain,'LSBoost',ABtreeNum,t,...
    'Type','regression','LearnRate',learnRate);

A=predict(RTreeEns,XTrain);
B=predict(RTreeEns,XTest);