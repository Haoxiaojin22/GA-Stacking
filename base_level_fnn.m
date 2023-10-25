function [A,B,r2A,rmseA]=base_level_fnn(XTrain,yTrain,XTest,yTest,c,hiddenSizes)

if nargin<=5
    hiddenSizes =10;
end

XTrain=XTrain(:,700:1700)';
XTest=XTest(:,700:1700)';
% XTrain=SNV(XTrain);
% XTest=SNV(XTest);
% % %适合用MSC
XTrain=MSC(XTrain);
XTest=MSC(XTest);
A=zeros(length(yTrain),1);
B=zeros(length(yTest),1);
%%
% 1. 训练集
[XTrain,inputps] = mapminmax(XTrain);
XTest = mapminmax('apply',XTest,inputps);
%%
% 2. 测试集
[yTrain,outputps] = mapminmax(yTrain');
yTest = mapminmax('apply',yTest',outputps);

% parpool(c.NumTestSets);
for ModelIndex=1:c.NumTestSets
    X=XTrain;
    Y=yTrain;
    trainidx5=training(c,ModelIndex);
    testidx5=test(c,ModelIndex);

    XTrain5=X(:,trainidx5);
    yTrain5=Y(:,trainidx5);

    XTest5 =X(:,testidx5);
        yTest5 =yTrain(:,testidx5);

    net=feedforwardnet(hiddenSizes,'trainlm');
    net.trainParam.showWindow=0;
    NN=train(net,XTrain5,yTrain5);
    
    a=NN(XTest5);
    b=NN(XTest);
    % 1. 反归一化
    a = mapminmax('reverse',a ,outputps);
    b = mapminmax('reverse',b ,outputps);
     [r2(ModelIndex),rmse(ModelIndex)] = rsquare(yTest5,a);
    A(testidx5)=a;
    B=B+b';
end
r2A=mean(r2);
rmseA=mean(rmse);
B=B/c.NumTestSets;

