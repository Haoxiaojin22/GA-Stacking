%%Selection of preprocessing methods 
data=load('cornMCCV.mat');
X=data.data(:,1:end-1);
y=data.data(:,end);
%%preprocessing methods
% X=MSC(X);
% X=SNV(X);
% X=FirstDerivative(X);
% X=MSC(X);

datasize=length(y);
trainsetsize=round(datasize*0.75);

%%single model 
MyIteration=10;
PLSresult=zeros(4,MyIteration);
for i=1:MyIteration
% rndperm=randperm(datasize);
% trainidx=rndperm(1:trainsetsize);
% testidx=rndperm(trainsetsize+1:end);
% XTrain=X(trainidx,:);
% yTrain=y(trainidx,:);
% XTest=X(testidx,:);
% yTest=y(testidx,:);
rndperm=MyRandi(i,:);
trainidx=rndperm(1:trainsetsize);
testidx=rndperm(trainsetsize+1:end);
XTrain=X(trainidx,:);
yTrain=y(trainidx,:);
XTest=X(testidx,:);
yTest=y(testidx,:);
tic;
[A_s_pls,B_s_pls]=single_pls(XTrain,yTrain,XTest,yTest,10);
toc;
[rc2_s_pls,rmsec_s_pls] = rsquare(yTrain,A_s_pls);
[rp2_s_pls,rmsep_s_pls] = rsquare(yTest,B_s_pls);
PLSresult(:,i)=[rmsec_s_pls,rc2_s_pls,rmsep_s_pls,rp2_s_pls];
end
PLSRMSECeven=mean(PLSresult(1,:));
PLSR2Ceven=mean(PLSresult(2,:));
PLSRMSEeven=mean(PLSresult(3,:));
PLSR2even=mean(PLSresult(4,:));

RFresult=zeros(4,MyIteration);
for i=1:MyIteration
% rndperm=randperm(datasize);
% trainidx=rndperm(1:trainsetsize);
% testidx=rndperm(trainsetsize+1:end);
% XTrain=X(trainidx,:);
% yTrain=y(trainidx,:);
% XTest=X(testidx,:);
% yTest=y(testidx,:);
rndperm=MyRandi(i,:);
trainidx=rndperm(1:trainsetsize);
testidx=rndperm(trainsetsize+1:end);
XTrain=X(trainidx,:);
yTrain=y(trainidx,:);
XTest=X(testidx,:);
yTest=y(testidx,:);
tic;
[A_s_rf,B_s_rf]=single_rf(XTrain,yTrain,XTest,yTest);
toc;
[rc2_s_rf,rmsec_s_rf] = rsquare(yTrain,A_s_rf);
[rp2_s_rf,rmsep_s_rf] = rsquare(yTest,B_s_rf);
RFresult(:,i)=[rmsec_s_rf,rc2_s_rf,rmsep_s_rf,rp2_s_rf];
end
RFRMSECeven=mean(RFresult(1,:));
RFR2Ceven=mean(RFresult(2,:));
RFRMSEeven=mean(RFresult(3,:));
RFR2even=mean(RFresult(4,:));

SVMresult=zeros(4,MyIteration);
for i=1:MyIteration
% rndperm=randperm(datasize);
% trainidx=rndperm(1:trainsetsize);
% testidx=rndperm(trainsetsize+1:end);
% XTrain=X(trainidx,:);
% yTrain=y(trainidx,:);
% XTest=X(testidx,:);
% yTest=y(testidx,:);
rndperm=MyRandi(i,:);
trainidx=rndperm(1:trainsetsize);
testidx=rndperm(trainsetsize+1:end);
XTrain=X(trainidx,:);
yTrain=y(trainidx,:);
XTest=X(testidx,:);
yTest=y(testidx,:);
tic;
[A_s_svm,B_s_svm]=single_svm(XTrain,yTrain,XTest,yTest);
toc;
[rc2_s_svm,rmsec_s_svm] = rsquare(yTrain,A_s_svm);
[rp2_s_svm,rmsep_s_svm] = rsquare(yTest,B_s_svm);
SVMresult(:,i)=[rmsec_s_svm,rc2_s_svm,rmsep_s_svm,rp2_s_svm];
end
SVMRMSECeven=mean(SVMresult(1,:));
SVMR2Ceven=mean(SVMresult(2,:));
SVMRMSEeven=mean(SVMresult(3,:));
SVMR2even=mean(SVMresult(4,:));

KNNresult=zeros(4,MyIteration);
for i=1:MyIteration
% rndperm=randperm(datasize);
% trainidx=rndperm(1:trainsetsize);
% testidx=rndperm(trainsetsize+1:end);
% XTrain=X(trainidx,:);
% yTrain=y(trainidx,:);
% XTest=X(testidx,:);
% yTest=y(testidx,:);
rndperm=MyRandi(i,:);
trainidx=rndperm(1:trainsetsize);
testidx=rndperm(trainsetsize+1:end);
XTrain=X(trainidx,:);
yTrain=y(trainidx,:);
XTest=X(testidx,:);
yTest=y(testidx,:);
tic;
[A_s_knn,B_s_knn]=single_knn(XTrain,yTrain,XTest,yTest);
toc;
[rc2_s_knn,rmsec_s_knn] = rsquare(yTrain,A_s_knn);
[rp2_s_knn,rmsep_s_knn] = rsquare(yTest,B_s_knn);
KNNresult(:,i)=[rmsec_s_knn,rc2_s_knn,rmsep_s_knn,rp2_s_knn];
end
KNNRMSECeven=mean(KNNresult(1,:));
KNNR2Ceven=mean(KNNresult(2,:));
KNNRMSEeven=mean(KNNresult(3,:));
KNNR2even=mean(KNNresult(4,:));

ABresult=zeros(4,MyIteration);
for i=1:MyIteration
% rndperm=randperm(datasize);
% trainidx=rndperm(1:trainsetsize);
% testidx=rndperm(trainsetsize+1:end);
% XTrain=X(trainidx,:);
% yTrain=y(trainidx,:);
% XTest=X(testidx,:);
% yTest=y(testidx,:);
rndperm=MyRandi(i,:);
trainidx=rndperm(1:trainsetsize);
testidx=rndperm(trainsetsize+1:end);
XTrain=X(trainidx,:);
yTrain=y(trainidx,:);
XTest=X(testidx,:);
yTest=y(testidx,:);
tic;
[A_s_boo,B_s_boo]=single_Adaboost(XTrain,yTrain,XTest,yTest);
toc;
[rc2_s_boo,rmsec_s_boo] = rsquare(yTrain,A_s_boo);
[rp2_s_boo,rmsep_s_boo] = rsquare(yTest,B_s_boo);
ABresult(:,i)=[rmsec_s_boo,rc2_s_boo,rmsep_s_boo,rp2_s_boo];
end
ABRMSECeven=mean(ABresult(1,:));
ABR2Ceven=mean(ABresult(2,:));
ABRMSEeven=mean(ABresult(3,:));
ABR2even=mean(ABresult(4,:));

GPRresult=zeros(4,MyIteration);
for i=1:MyIteration
% rndperm=randperm(datasize);
% trainidx=rndperm(1:trainsetsize);
% testidx=rndperm(trainsetsize+1:end);
% XTrain=X(trainidx,:);
% yTrain=y(trainidx,:);
% XTest=X(testidx,:);
% yTest=y(testidx,:);
rndperm=MyRandi(i,:);
trainidx=rndperm(1:trainsetsize);
testidx=rndperm(trainsetsize+1:end);
XTrain=X(trainidx,:);
yTrain=y(trainidx,:);
XTest=X(testidx,:);
yTest=y(testidx,:);
tic;
[A_s_gpr,B_s_gpr]=single_gpr(XTrain,yTrain,XTest,yTest);
toc;
[rc2_s_gpr,rmsec_s_gpr] = rsquare(yTrain,A_s_gpr);
[rp2_s_gpr,rmsep_s_gpr] = rsquare(yTest,B_s_gpr);
GPRresult(:,i)=[rmsec_s_gpr,rc2_s_gpr,rmsep_s_gpr,rp2_s_gpr];
end
GPRRMSECeven=mean(GPRresult(1,:));
GPRR2Ceven=mean(GPRresult(2,:));
GPRRMSEeven=mean(GPRresult(3,:));
GPRR2even=mean(GPRresult(4,:));

FNNresult=zeros(4,MyIteration);
for i=1:MyIteration
% rndperm=randperm(datasize);
% trainidx=rndperm(1:trainsetsize);
% testidx=rndperm(trainsetsize+1:end);
% XTrain=X(trainidx,:);
% yTrain=y(trainidx,:);
% XTest=X(testidx,:);
% yTest=y(testidx,:);
rndperm=MyRandi(i,:);
trainidx=rndperm(1:trainsetsize);
testidx=rndperm(trainsetsize+1:end);
XTrain=X(trainidx,:);
yTrain=y(trainidx,:);
XTest=X(testidx,:);
yTest=y(testidx,:);
tic;
[A_s_fnn,B_s_fnn]=single_fnn(XTrain,yTrain,XTest,yTest,20);
toc;
[rc2_s_fnn,rmsec_s_fnn] = rsquare(yTrain,A_s_fnn);
[rp2_s_fnn,rmsep_s_fnn] = rsquare(yTest,B_s_fnn);
FNNresult(:,i)=[rmsec_s_fnn,rc2_s_fnn,rmsep_s_fnn,rp2_s_fnn];
end
FNNRMSECeven=mean(FNNresult(1,:));
FNNR2Ceven=mean(FNNresult(2,:));
FNNRMSEeven=mean(FNNresult(3,:));
FNNR2even=mean(FNNresult(4,:));