function [A,B,r2A,rmseA]=base_level_adaboost(XTrain,yTrain,XTest,yTest,c)
% %适合用MSC
[XTrain,XTest]=MSC(XTrain,XTest);
% XTrain=FirstDerivative(XTrain);
% XTest=FirstDerivative(XTest);

A=zeros(length(yTrain),1);
B=zeros(length(yTest),1);

ABtreeNum=100;
ABnodeNum=5;
learnRate=0.02;

for ModelIndex=1:c.NumTestSets
    trainidx5=training(c,ModelIndex);
    testidx5=test(c,ModelIndex);

    XTrain5=XTrain(trainidx5,:);
    yTrain5=yTrain(trainidx5,:);

    XTest5 =XTrain(testidx5,:);
        yTest5 =yTrain(testidx5,:);

    t = templateTree('MaxNumSplits',ABnodeNum,'Surrogate','on');
    RTreeEns=fitensemble(XTrain5,yTrain5,'LSBoost',ABtreeNum,t,...
        'Type','regression','LearnRate',learnRate);

    a=predict(RTreeEns,XTest5);
    A(testidx5)=a;
    b=predict(RTreeEns,XTest);
    [r2(ModelIndex),rmse(ModelIndex)] = rsquare(yTest5,a);
    B=B+b;
end
r2A=mean(r2);
rmseA=mean(rmse);
B=B/c.NumTestSets;
