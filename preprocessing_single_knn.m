function [A,B]=single_knn(XTrain,yTrain,XTest,yTest,k)

if nargin<=4
    k=24;
end

[a,b]=size(XTrain);
if b>1
    XTrain=diff(XTrain,1,2);
    XTest=diff(XTest,1,2);
end

A=zeros(size(XTrain,1),1);
B=zeros(size(XTest,1),1);

Mdl = KDTreeSearcher(XTrain);

idxa = knnsearch(Mdl,XTrain(:,:),'k',k);

for j = 1:size(idxa,1)
    A(j,1)=mean(yTrain(idxa(j,:),:));
end

idxb = knnsearch(Mdl,XTest(:,:),'k',k);
for j = 1:size(idxb,1)
    B(j,1)=mean(yTrain(idxb(j,:),:));
end