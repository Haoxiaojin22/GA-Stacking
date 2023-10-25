function [A,B,r2A,rmseA]=base_level_pls(XTrain,yTrain,XTest,yTest,c)
% 使用XTrain,yTrain进行5折交叉建模,
% 使用建立的模型计算XTest预测值，
% yTest在次函数中用于计算样本集大小，没有参与计算
% c是5折划分

% A是PLS生成的新特征
% B是XTest五次回归的平均值
% % 适合用MSC
XTrain=SNV(XTrain);
XTest=SNV(XTest);

A=zeros(length(yTrain),1);
B=zeros(length(yTest),1);

for ModelIndex=1:c.NumTestSets
    trainidx5=training(c,ModelIndex);
    testidx5=test(c,ModelIndex);

    XTrain5=XTrain(trainidx5,:);
    yTrain5=yTrain(trainidx5,:);

    XTest5 =XTrain(testidx5,:);
    yTest5 =yTrain(testidx5,:);

    [~,~,~,~,beta] = plsregress(XTrain5,yTrain5,10);

    a = [ones(size(XTest5,1),1) XTest5]*beta;
    b = [ones(size(XTest,1),1) XTest]*beta;
    [r2(ModelIndex),rmse(ModelIndex)] = rsquare(yTest5,a);
    A(testidx5)=a;
    B=B+b;
end
r2A=mean(r2);
rmseA=mean(rmse);
B=B/c.NumTestSets;