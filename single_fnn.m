function [A,B]=single_fnn(XTrain,yTrain,XTest,yTest,hiddenSizes )
if nargin<=4
    hiddenSizes =10;
end

XTrain=XTrain(:,700:1700);
XTest=XTest(:,700:1700);

XTrain=XTrain';
XTest=XTest';
% % %适合用MSC
% XTrain=MSC(XTrain');
% XTest=MSC(XTest');
%%
% 1. 训练集
[XTrain,inputps] = mapminmax(XTrain);
XTest = mapminmax('apply',XTest,inputps);
%%
% 2. 测试集
[yTrain,outputps] = mapminmax(yTrain');
yTest = mapminmax('apply',yTest',outputps);

net = feedforwardnet(hiddenSizes,'trainlm');
net.trainParam.showWindow=0;
NN=train(net,XTrain,yTrain);
A=NN(XTrain);
B=NN(XTest);
  % 1. 反归一化
    A = mapminmax('reverse',A ,outputps);
    B = mapminmax('reverse',B ,outputps);
    A = A';
    B = B';
