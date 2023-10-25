function [A,B,r2A,rmseA]=base_level_rf(XTrain,yTrain,XTest,yTest,c)
% 使用XTrain,yTrain进行5折交叉建模,
% 使用建立的模型计算XTest预测值，
% yTest在次函数中用于计算样本集大小，没有参与计算
% c是5折划分

% A是PLS生成的新特征
% B是XTest五次回归的平均值
% %适合用MSC
XTrain=MSC(XTrain);
XTest=MSC(XTest);
A=zeros(length(yTrain),1);
start=1;
B=zeros(length(yTest),1);

for ModelIndex=1:c.NumTestSets   
    trainidx5=training(c,ModelIndex);
    testidx5=test(c,ModelIndex);

    XTrain5=XTrain(trainidx5,:);
    yTrain5=yTrain(trainidx5,:);

    XTest5 =XTrain(testidx5,:);
    yTest5 =yTrain(testidx5,:);

    treeNum=80;
    nodeNum=32;
    beta=TreeBagger(treeNum,XTrain5,yTrain5,'Method','regression', ...
        'OOBPredictorImportance','on',...
        'CategoricalPredictors',false(size(XTrain5,2),1),...
        'MinLeafSize',nodeNum);
    
    a=predict(beta,XTest5);
    b=predict(beta,XTest);
    [r2(ModelIndex),rmse(ModelIndex)] = rsquare(yTest5,a);
    len=size(XTest5,1);
    A(testidx5)=a;
    start=start+len;
    B=B+b;
end
r2A=mean(r2);
rmseA=mean(rmse);
B=B/c.NumTestSets;