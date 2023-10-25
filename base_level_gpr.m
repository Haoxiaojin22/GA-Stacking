function [A,B,r2A,rmseA]=base_level_gpr(XTrain,yTrain,XTest,yTest,c)

a=cell(c.NumTestSets,1);
B=zeros(length(yTest),1);
A=zeros(length(yTrain),1);
delete(gcp('nocreate'));
% parpool(c.NumTestSets);
XTrain=XTrain(:,700:1800);
XTest=XTest(:,700:1800);
% XTrain=SNV(XTrain);
% XTest=SNV(XTest);
for ModelIndex=1:c.NumTestSets
    trainidx5=training(c,ModelIndex);
    testidx5=test(c,ModelIndex);
    XTrain5=XTrain(trainidx5,:);
    yTrain5=yTrain(trainidx5,:);
    XTest5 =XTrain(testidx5,:);
        yTest5 =yTrain(testidx5,:);

    sigma0 = std(yTrain);
    sigmaF0 = sigma0;

    d = size(XTrain5,2);
    sigmaM0 = 10*ones(d,1);
    kparams0 = [sigmaM0;sigmaF0];

    gprMdl = fitrgp(XTrain5,yTrain5, ...
        'KernelFunction','ardsquaredexponential',...
        'KernelParameters',kparams0, ...
        'Sigma',sigma0);

   a{ModelIndex}=predict(gprMdl,XTest5);
   
    b=predict(gprMdl,XTest);
    B=B+b;
end
B=B./c.NumTestSets;
aa=[];
for i=1:c.NumTestSets
    aa=cat(1,aa,[a{i,:}]);
end
L=0;
for i=1:c.NumTestSets
    testidx5=test(c,i);
     yTest5 =yTrain(testidx5,:);
    temp=sum(testidx5);
    A(testidx5)=aa(L+1:L+temp);
     [r2(ModelIndex),rmse(ModelIndex)] = rsquare(yTest5,aa(L+1:L+temp));
        L=L+temp;
end
r2A=mean(r2);
rmseA=mean(rmse);
delete(gcp('nocreate'));
   
