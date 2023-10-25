

data=load('cornMCCV.mat');
X=data.data(:,1:end-1);
y=data.data(:,end);
% X=MSC(X);
% X=SNV(X);
% X=FirstDerivative(X);
% X=MSC(X);

datasize=length(y);
trainsetsize=round(datasize*0.75);

% rndperm=randperm(datasize);
% trainidx=rndperm(1:trainsetsize);
% testidx=rndperm(trainsetsize+1:end);
% XTrain=X(trainidx,:);
% yTrain=y(trainidx,:);
% XTest=X(testidx,:);
% yTest=y(testidx,:);
c=cvpartition(trainsetsize,'Kfold',5);

% yTrain5=[];
% yCell=cell(c.NumTestSets,1);
% for i=1:c.NumTestSets
%     testidx5=test(c,i);
%     yCell{i}=yTrain(testidx5,:);
% end
% for i=1:c.NumTestSets
%     yTrain5=cat(1,yTrain5,yCell{i});
% end
MyIteration=3;
PLSA=zeros(trainsetsize,MyIteration);
PLSB=zeros(datasize-trainsetsize,MyIteration);
PLSresult=zeros(4,MyIteration);
RFA=zeros(trainsetsize,MyIteration);
RFB=zeros(datasize-trainsetsize,MyIteration);
RFresult=zeros(4,MyIteration);
SVMA=zeros(trainsetsize,MyIteration);
SVMB=zeros(datasize-trainsetsize,MyIteration);
SVMresult=zeros(4,MyIteration);
KNNA=zeros(trainsetsize,MyIteration);
KNNB=zeros(datasize-trainsetsize,MyIteration);
KNNresult=zeros(4,MyIteration);
ABA=zeros(trainsetsize,MyIteration);
ABB=zeros(datasize-trainsetsize,MyIteration);
ABresult=zeros(4,MyIteration);
GPRA=zeros(trainsetsize,MyIteration);
GPRB=zeros(datasize-trainsetsize,MyIteration);
GPRresult=zeros(4,MyIteration);
FNNA=zeros(trainsetsize,MyIteration);
FNNB=zeros(datasize-trainsetsize,MyIteration);
FNNresult=zeros(4,MyIteration);

for i=1:MyIteration
c=cvpartition(trainsetsize,'Kfold',10);
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
[A_pls,B_pls]=base_level_pls(XTrain,yTrain,XTest,yTest,c);
PLSA(:,i)=A_pls;
PLSB(:,i)=B_pls;
toc;
[rc2_pls,rmsec_pls] = rsquare(yTrain,A_pls);
[rp2_pls,rmsep_pls] = rsquare(yTest,B_pls);
PLSresult(:,i)=[rmsec_pls,rc2_pls,rmsep_pls,rp2_pls];

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
[A_rf,B_rf]=base_level_rf(XTrain,yTrain,XTest,yTest,c);
toc;
RFA(:,i)=A_rf;
RFB(:,i)=B_rf;
[rc2_rf,rmsec_rf] = rsquare(yTrain,A_rf);
[rp2_rf,rmsep_rf] = rsquare(yTest,B_rf);
RFresult(:,i)=[rmsec_rf,rc2_rf,rmsep_rf,rp2_rf];

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
[A_svm,B_svm]=base_level_svm(XTrain,yTrain,XTest,yTest,c);
SVMA(:,i)=A_svm;
SVMB(:,i)=B_svm;
toc;
[rc2_svm,rmsec_svm] = rsquare(yTrain,A_svm);
[rp2_svm,rmsep_svm] = rsquare(yTest,B_svm);
SVMresult(:,i)=[rmsec_svm,rc2_svm,rmsep_svm,rp2_svm];

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
[A_knn,B_knn]=base_level_knn(XTrain,yTrain,XTest,yTest,c);
KNNA(:,i)=A_knn;
KNNB(:,i)=B_knn;
toc;
[rc2_knn,rmsec_knn] = rsquare(yTrain,A_knn);
[rp2_knn,rmsep_knn] = rsquare(yTest,B_knn);
KNNresult(:,i)=[rmsec_knn,rc2_knn,rmsep_knn,rp2_knn];

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
[A_boo,B_boo]=base_level_adaboost(XTrain,yTrain,XTest,yTest,c);
ABA(:,i)=A_boo;
ABB(:,i)=B_boo;
toc;
[rc2_boo,rmsec_boo] = rsquare(yTrain,A_boo);
[rp2_boo,rmsep_boo] = rsquare(yTest,B_boo);
ABresult(:,i)=[rmsec_boo,rc2_boo,rmsep_boo,rp2_boo];

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
[A_gpr,B_gpr]=base_level_gpr(XTrain,yTrain,XTest,yTest,c);
GPRA(:,i)=A_gpr;
GPRB(:,i)=B_gpr;
toc;
[rc2_gpr,rmsec_gpr] = rsquare(yTrain,A_gpr);
[rp2_gpr,rmsep_gpr] = rsquare(yTest,B_gpr);
GPRresult(:,i)=[rmsec_gpr,rc2_gpr,rmsep_gpr,rp2_gpr];

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
[A_fnn,B_fnn]=base_level_fnn(XTrain,yTrain,XTest,yTest,c,20);
FNNA(:,i)=A_fnn;
FNNB(:,i)=B_fnn;
toc;
[rc2_fnn,rmsec_fnn] = rsquare(yTrain,A_fnn);
[rp2_fnn,rmsep_fnn] = rsquare(yTest,B_fnn);
FNNresult(:,i)=[rmsec_fnn,rc2_fnn,rmsep_fnn,rp2_fnn];

% % rndperm=randperm(datasize);
% % trainidx=rndperm(1:trainsetsize);
% % testidx=rndperm(trainsetsize+1:end);
% % XTrain=X(trainidx,:);
% % yTrain=y(trainidx,:);
% % XTest=X(testidx,:);
% % yTest=y(testidx,:);
% rndperm=MyRandi50(i,:);
% trainidx=rndperm(1:trainsetsize);
% testidx=rndperm(trainsetsize+1:end);
% XTrain=X(trainidx,:);
% yTrain=y(trainidx,:);
% XTest=X(testidx,:);
% yTest=y(testidx,:);
% tic;
% [A_bp,B_bp]=base_level_bp(XTrain,yTrain,XTest,yTest,c);
% BPA(:,i)=A_bp;
% BPB(:,i)=B_bp;
% toc;
% [rc2_bp,rmsec_bp] = rsquare(yTrain,A_bp);
% [rp2_bp,rmsep_bp] = rsquare(yTest,B_bp);
% BPresult(:,i)=[rmsec_bp,rc2_bp,rmsep_bp,rp2_bp];
% 
% % rndperm=randperm(datasize);
% % trainidx=rndperm(1:trainsetsize);
% % testidx=rndperm(trainsetsize+1:end);
% % XTrain=X(trainidx,:);
% % yTrain=y(trainidx,:);
% % XTest=X(testidx,:);
% % yTest=y(testidx,:);
% rndperm=MyRandi50(i,:);
% trainidx=rndperm(1:trainsetsize);
% testidx=rndperm(trainsetsize+1:end);
% XTrain=X(trainidx,:);
% yTrain=y(trainidx,:);
% XTest=X(testidx,:);
% yTest=y(testidx,:);
% tic;
% [A_rr,B_rr]=base_level_rr(XTrain,yTrain,XTest,yTest,c);
% RRA(:,i)=A_rr;
% RRB(:,i)=B_rr;
% toc;
% [rc2_rr,rmsec_rr] = rsquare(yTrain,A_rr);
% [rp2_rr,rmsep_rr] = rsquare(yTest,B_rr);
% RRresult(:,i)=[rmsec_rr,rc2_rr,rmsep_rr,rp2_rr];
end

%% GA-Stacking
MetaResult=zeros(4,MyIteration);
RFMetaresult=zeros(4,MyIteration);
SVMMetaresult=zeros(4,MyIteration);
PLSMetaresult=zeros(4,MyIteration);
GPRMetaresult=zeros(4,MyIteration);
for i=1:MyIteration
    rndperm=MyRandi(i,:);
    trainidx=rndperm(1:trainsetsize);
    testidx=rndperm(trainsetsize+1:end);
    XTrain=X(trainidx,:);
    yTrain=y(trainidx,:);
    XTest=X(testidx,:);
    yTest=y(testidx,:);
    A0=[];  B0=[];  A1=[]; B1=[];
    A0=[RFA(:,i) SVMA(:,i) GPRA(:,i) PLSA(:,i) ABA(:,i) KNNA(:,i) FNNA(:,i)];
    B0=[RFB(:,i) SVMB(:,i) GPRB(:,i) PLSB(:,i) ABB(:,i) KNNB(:,i) FNNB(:,i)];
    A1=[SVMA(:,i) PLSA(:,i) FNNA(:,i)];
    B1=[SVMB(:,i) PLSB(:,i) FNNB(:,i)];

    yy=[];
    %迭代轮次
    iter=50;
    popsize=30;                                       %群体大小
    chromlength=9;                                   %字符串长度（个体长度）
    count = zeros(iter,chromlength);
    pc=0.6;                                           %交叉概率
    pm=0.001;                                         %变异概率
    pop=initpop(popsize,chromlength);                 %随机产生初始群体
    bestindividuals=zeros(iter,chromlength);     %每轮最好的特征选择
    for ii=1:iter                                    %20为迭代次数
        %     objvalue(i,1)=R2/abs(1-(R2*R2)/(R2C*R2C));
        [objvalue]=calobjvalue3(pop,A0,yTrain,B0,yTest);                      %计算目标函数
        fitvalue=objvalue;                   %计算群体中每个个体的适应度
        [newpop]=selection(pop,fitvalue);                 %复制
        [newpop]=crossover_multiv(newpop,pc);                       %交叉
        [newpop]=mutation(newpop,pc);                        %变异
        %     pop=newpop;
        [bestindividual,bestfit]=best(pop,fitvalue);%求出群体中适应值最大的个体及其适应值
        yy(ii)=max(bestfit);
        bestindividuals(ii,:)=bestindividual;
        nn(ii)=ii;
        %     pop5=bestindividual;
        %     x(ii)=decodechrom(pop5,1,chromlength)*10/1023;
        pop=newpop;
        count(ii,:)=sum(pop,1);
    end
    [M ,I]=max(yy);
    Bestindividual=bestindividuals(I,:);
    ind=logical(Bestindividual(1:7));
    newSpec=A0(:,ind);
    newtestSpec=B0(:,ind);
    num=num2str(Bestindividual(8:9));%次级模型编号
    temp1=bin2dec(num) %将编号转化成十进制数
    if temp1==0
        result=MetaGPR(newSpec, newtestSpec,yTrain, yTest);
    end
    if temp1==1
        result=MetaSVM(newSpec, newtestSpec, yTrain, yTest);       
    end
    if temp1==2
        result=MetaPLS(newSpec, newtestSpec,yTrain, yTest);    
    end
    if temp1==3
        result=MetaRF(newSpec, newtestSpec,yTrain, yTest);      
    end
    MetaResult(:,i)=result;
%     metaName = {'RMSEC';'R²C';'RMSEP';'R²P'};   
%     Tabl=table(metaName,MetaResult);
%     writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");   

%% Manual selection of model combinations
%% 选三个最好的单学习器次级随机森林
isCategorical = zeros(size(X,2),1);
treeNum=100;
nodeNum=15;
Stacking = TreeBagger(treeNum,A1,yTrain,'Method','R','OOBPredictorImportance','On',...
    'CategoricalPredictors',find(isCategorical == 1),...
    'MinLeafSize',nodeNum);
Ypredict5=predict(Stacking,B1);
YpredictC=predict(Stacking,A1);
%模型分析
co=corr(yTest,Ypredict5,'type','Pearson');
RMSE=sqrt(sum((yTest-Ypredict5).^2)/size(Ypredict5,1));
R2 = corr(yTest,Ypredict5)^2;
RMSEC=sqrt(sum((yTrain-YpredictC).^2)/size(YpredictC,1));
R2C = corr(yTrain,YpredictC)^2;
result=[RMSEC;R2C;RMSE;R2];
co=num2str(roundn(co,-4));
RMSE=num2str(roundn(RMSE,-4));
R2=num2str(roundn(R2,-4));
RMSEC=num2str(roundn(RMSEC,-4));
R2C=num2str(roundn(R2C,-4));

 RFMetaresult(:,i)=result;
     
     %% 选三个效果好的单模型次级模型SVM
Stacking = fitrsvm(A1,yTrain,'KernelFunction','gaussian','OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('Optimizer','gridsearch','NumGridDivisions',15,...
    'ShowPlots',false,'MaxObjectiveEvaluations',60));
Ypredict5=predict(Stacking,B1);
YpredictC=predict(Stacking,A1);
%模型分析
co=corr(yTest,Ypredict5,'type','Pearson');
RMSE=sqrt(sum((yTest-Ypredict5).^2)/size(Ypredict5,1));
R2 = corr(yTest,Ypredict5)^2;
RMSEC=sqrt(sum((yTrain-YpredictC).^2)/size(YpredictC,1));
R2C = corr(yTrain,YpredictC)^2;
result=[RMSEC;R2C;RMSE;R2];
co=num2str(roundn(co,-4));
RMSE=num2str(roundn(RMSE,-4));
R2=num2str(roundn(R2,-4));
RMSEC=num2str(roundn(RMSEC,-4));
R2C=num2str(roundn(R2C,-4));

 SVMMetaresult(:,i)=result;
 
 %% 选三个效果好的单模型次级模型PLS
Stacking = fitlm(A1,yTrain);
Ypredict5=predict(Stacking,B1);
YpredoctA=predict(Stacking,A1);
%模型分析
co=corr(yTest,Ypredict5,'type','Pearson');
RMSE=sqrt(sum((yTest-Ypredict5).^2)/size(Ypredict5,1));
R2 = corr(yTest,Ypredict5)^2;
coP=corr(yTrain,YpredoctA,'type','Pearson');
RMSEC=sqrt(sum((yTrain-YpredoctA).^2)/size(YpredoctA,1));
R2C = corr(yTrain,YpredoctA)^2;
result=[RMSEC;R2C;RMSE;R2];
co=num2str(roundn(co,-4));
RMSE=num2str(roundn(RMSE,-4));
R2=num2str(roundn(R2,-4));
RMSEC=num2str(roundn(RMSEC,-4));
R2C=num2str(roundn(R2C,-4));

 PLSMetaresult(:,i)=result;
 
 %% 选三个效果好的单模型次级模型高斯回归
sigma0 = 0.2;
kparams0 = [3.5, 6.2];
stacking = fitrgp(A1,yTrain,'KernelFunction','squaredexponential',...
        'KernelParameters',kparams0,'Sigma',sigma0);
Ypredict5=predict(stacking,B1);
YpredictC=predict(stacking,A1);
%模型分析
co=corr(yTest,Ypredict5,'type','Pearson');
RMSE=sqrt(sum((yTest-Ypredict5).^2)/size(Ypredict5,1));
R2 = corr(yTest,Ypredict5)^2;
RMSEC=sqrt(sum((yTrain-YpredictC).^2)/size(YpredictC,1));
R2C = corr(yTrain,YpredictC)^2;
result=[RMSEC;R2C;RMSE;R2];
co=num2str(roundn(co,-4));
RMSE=num2str(roundn(RMSE,-4));
R2=num2str(roundn(R2,-4));
RMSEC=num2str(roundn(RMSEC,-4));
R2C=num2str(roundn(R2C,-4));
msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
    ['R2C：',R2C,'R2：',R2]},...
    '次级模型高斯回归');
 GPRMetaresult(:,i)=result;
end

GASRMSECeven=mean(MetaResult(1,:));
GASR2Ceven=mean(MetaResult(2,:));
GASRMSEeven=mean(MetaResult(3,:));
GASR2even=mean(MetaResult(4,:));
RFMetaRMSECeven=mean(RFMetaresult(1,:));
RFMetaR2Ceven=mean(RFMetaresult(2,:));
RFMetaRMSEeven=mean(RFMetaresult(3,:));
RFMetaR2even=mean(RFMetaresult(4,:));
PLSMetaRMSECeven=mean(PLSMetaresult(1,:));
PLSMetaR2Ceven=mean(PLSMetaresult(2,:));
PLSMetaRMSEeven=mean(PLSMetaresult(3,:));
PLSMetaR2even=mean(PLSMetaresult(4,:));
SVMMetaRMSECeven=mean(SVMMetaresult(1,:));
SVMMetaR2Ceven=mean(SVMMetaresult(2,:));
SVMMetaRMSEeven=mean(SVMMetaresult(3,:));
SVMMetaR2even=mean(SVMMetaresult(4,:));
GPRMetaRMSECeven=mean(GPRMetaresult(1,:));
GPRMetaR2Ceven=mean(GPRMetaresult(2,:));
GPRMetaRMSEeven=mean(GPRMetaresult(3,:));
GPRMetaR2even=mean(GPRMetaresult(4,:));


%% 单一模型
%PLS
% MyIteration=10;
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
[A_s_pls,B_s_pls]=preprocessing_single_pls(XTrain,yTrain,XTest,yTest,10);
toc;
[rc2_s_pls,rmsec_s_pls] = rsquare(yTrain,A_s_pls);
[rp2_s_pls,rmsep_s_pls] = rsquare(yTest,B_s_pls);
PLSresult(:,i)=[rmsec_s_pls,rc2_s_pls,rmsep_s_pls,rp2_s_pls];
end
PLSRMSECeven=mean(PLSresult(1,:));
PLSR2Ceven=mean(PLSresult(2,:));
PLSRMSEeven=mean(PLSresult(3,:));
PLSR2even=mean(PLSresult(4,:));

%RF
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
[A_s_rf,B_s_rf]=preprocessing_single_rf(XTrain,yTrain,XTest,yTest);
toc;
[rc2_s_rf,rmsec_s_rf] = rsquare(yTrain,A_s_rf);
[rp2_s_rf,rmsep_s_rf] = rsquare(yTest,B_s_rf);
RFresult(:,i)=[rmsec_s_rf,rc2_s_rf,rmsep_s_rf,rp2_s_rf];
end
RFRMSECeven=mean(RFresult(1,:));
RFR2Ceven=mean(RFresult(2,:));
RFRMSEeven=mean(RFresult(3,:));
RFR2even=mean(RFresult(4,:));

%SVM
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
[A_s_svm,B_s_svm]=preprocessing_single_svm(XTrain,yTrain,XTest,yTest);
toc;
[rc2_s_svm,rmsec_s_svm] = rsquare(yTrain,A_s_svm);
[rp2_s_svm,rmsep_s_svm] = rsquare(yTest,B_s_svm);
SVMresult(:,i)=[rmsec_s_svm,rc2_s_svm,rmsep_s_svm,rp2_s_svm];
end
SVMRMSECeven=mean(SVMresult(1,:));
SVMR2Ceven=mean(SVMresult(2,:));
SVMRMSEeven=mean(SVMresult(3,:));
SVMR2even=mean(SVMresult(4,:));

%KNN
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
[A_s_knn,B_s_knn]=preprocessing_single_knn(XTrain,yTrain,XTest,yTest);
toc;
[rc2_s_knn,rmsec_s_knn] = rsquare(yTrain,A_s_knn);
[rp2_s_knn,rmsep_s_knn] = rsquare(yTest,B_s_knn);
KNNresult(:,i)=[rmsec_s_knn,rc2_s_knn,rmsep_s_knn,rp2_s_knn];
end
KNNRMSECeven=mean(KNNresult(1,:));
KNNR2Ceven=mean(KNNresult(2,:));
KNNRMSEeven=mean(KNNresult(3,:));
KNNR2even=mean(KNNresult(4,:));

%AdaBoost
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
[A_s_boo,B_s_boo]=preprocessing_single_Adaboost(XTrain,yTrain,XTest,yTest);
toc;
[rc2_s_boo,rmsec_s_boo] = rsquare(yTrain,A_s_boo);
[rp2_s_boo,rmsep_s_boo] = rsquare(yTest,B_s_boo);
ABresult(:,i)=[rmsec_s_boo,rc2_s_boo,rmsep_s_boo,rp2_s_boo];
end
ABRMSECeven=mean(ABresult(1,:));
ABR2Ceven=mean(ABresult(2,:));
ABRMSEeven=mean(ABresult(3,:));
ABR2even=mean(ABresult(4,:));

%GPR
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
[A_s_gpr,B_s_gpr]=preprocessing_single_gpr(XTrain,yTrain,XTest,yTest);
toc;
[rc2_s_gpr,rmsec_s_gpr] = rsquare(yTrain,A_s_gpr);
[rp2_s_gpr,rmsep_s_gpr] = rsquare(yTest,B_s_gpr);
GPRresult(:,i)=[rmsec_s_gpr,rc2_s_gpr,rmsep_s_gpr,rp2_s_gpr];
end
GPRRMSECeven=mean(GPRresult(1,:));
GPRR2Ceven=mean(GPRresult(2,:));
GPRRMSEeven=mean(GPRresult(3,:));
GPRR2even=mean(GPRresult(4,:));

%FNN
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
[A_s_fnn,B_s_fnn]=preprocessing_single_fnn(XTrain,yTrain,XTest,yTest,20);
toc;
[rc2_s_fnn,rmsec_s_fnn] = rsquare(yTrain,A_s_fnn);
[rp2_s_fnn,rmsep_s_fnn] = rsquare(yTest,B_s_fnn);
FNNresult(:,i)=[rmsec_s_fnn,rc2_s_fnn,rmsep_s_fnn,rp2_s_fnn];
end
FNNRMSECeven=mean(FNNresult(1,:));
FNNR2Ceven=mean(FNNresult(2,:));
FNNRMSEeven=mean(FNNresult(3,:));
FNNR2even=mean(FNNresult(4,:));


%% PLS参数选择
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
n=NCOMP_BestNumber_Search_RMSECV(XTrain,yTrain,5,1,20);
end

% BPresult=zeros(4,MyIteration);
% for i=1:MyIteration
% % rndperm=randperm(datasize);
% % trainidx=rndperm(1:trainsetsize);
% % testidx=rndperm(trainsetsize+1:end);
% % XTrain=X(trainidx,:);
% % yTrain=y(trainidx,:);
% % XTest=X(testidx,:);
% % yTest=y(testidx,:);
% rndperm=MyRandi(i,:);
% trainidx=rndperm(1:trainsetsize);
% testidx=rndperm(trainsetsize+1:end);
% XTrain=X(trainidx,:);
% yTrain=y(trainidx,:);
% XTest=X(testidx,:);
% yTest=y(testidx,:);
% tic;
% [A_s_bp,B_s_bp]=preprocessing_single_bp(XTrain,yTrain,XTest,yTest);
% toc;
% [rc2_s_bp,rmsec_s_bp] = rsquare(yTrain,A_s_bp);
% [rp2_s_bp,rmsep_s_bp] = rsquare(yTest,B_s_bp);
% BPresult(:,i)=[rmsec_s_bp,rc2_s_bp,rmsep_s_bp,rp2_s_bp];
% end
% BPRMSECeven=mean(BPresult(1,:));
% BPR2Ceven=mean(BPresult(2,:));
% BPRMSEeven=mean(BPresult(3,:));
% BPR2even=mean(BPresult(4,:));
% 
% RRresult=zeros(4,MyIteration);
% for i=1:MyIteration
% % rndperm=randperm(datasize);
% % trainidx=rndperm(1:trainsetsize);
% % testidx=rndperm(trainsetsize+1:end);
% % XTrain=X(trainidx,:);
% % yTrain=y(trainidx,:);
% % XTest=X(testidx,:);
% % yTest=y(testidx,:);
% rndperm=MyRandi(i,:);
% trainidx=rndperm(1:trainsetsize);
% testidx=rndperm(trainsetsize+1:end);
% XTrain=X(trainidx,:);
% yTrain=y(trainidx,:);
% XTest=X(testidx,:);
% yTest=y(testidx,:);
% [A_s_rr,B_s_rr]=preprocessing_single_rr(XTrain,yTrain,XTest,yTest);
% [rc2_s_rr,rmsec_s_rr] = rsquare(yTrain,A_s_rr);
% [rp2_s_rr,rmsep_s_rr] = rsquare(yTest,B_s_rr);
% RRresult(:,i)=[rmsec_s_rr,rc2_s_rr,rmsep_s_rr,rp2_s_rr];
% end
% toc;
% RRRMSECeven=mean(RRresult(1,:));
% RRR2Ceven=mean(RRresult(2,:));
% RRRMSEeven=mean(RRresult(3,:));
% RRR2even=mean(RRresult(4,:));
