function [A,B,r2A,rmseA]=base_level_svm(XTrain,yTrain,XTest,yTest,c)
% %一阶导数适合
XTrain=diff(XTrain,1,2);
XTest=diff(XTest,1,2);
XTrain=XTrain(:,700:1800);
XTest=XTest(:,700:1800);
% XTrain=SNV(XTrain);
% XTest=SNV(XTest);
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

    MdlSvm = fitrsvm(XTrain5,yTrain5, ...
        'KernelFunction','rbf', ...
        'OptimizeHyperparameters','auto',...
        'HyperparameterOptimizationOptions', ...
        struct('Optimizer','gridsearch','ShowPlots',false,'Verbose',0,'MaxObjectiveEvaluations',60));

    a=predict(MdlSvm,XTest5);
    b=predict(MdlSvm,XTest);
    [r2(ModelIndex),rmse(ModelIndex)] = rsquare(yTest5,a);
    len=size(XTest5,1);
    A(testidx5)=a;
    start=start+len;
    B=B+b;
end
r2A=mean(r2);
rmseA=mean(rmse);
B=B./c.NumTestSets;