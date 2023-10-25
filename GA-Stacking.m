%将次级模型一起编码
D1=[];
D2=[];
D11=[];
D21=[];
choos=0;
% %剔除异常值
% % data=csvread('D:\HXJ\研究生\数据\玉米发芽\玉米发芽率数据\玉米发芽率数据\laohuaYM_noAvg.csv');
% % data=data(2:end,:);
% data=[diesel_spec.data(:,:),diesel_prop.data(:,1)];
% outval=4;
% outliers = (isnan(data(:,end))|abs(data(:,end))==inf);
% outliers=~outliers;
% [n,m]=size(data);
% ind=logical(outliers);
% data=data(ind,:);
% % if m>1
% %     data(any(outliers'),:)=[];
% % else
% %     data(find(outliers'),:)=[];
% % end
% % [n,m]=size(data);
% % mu=mean(data(:,end));
% % sigma=std(data(:,end));
% % outliers=(abs(data(:,end)-ones(n,1)*mu)>outval*ones(n,1)*sigma);
% % % if m>1
% % %     data(any(outliers'),:)=[];
% % % else
% % %     data(find(outliers'),:)=[];
% % % end
% data=[rockNIR,TOC];
% plot(data(:,1:end-1)');
MyIteration=5;
ABa=zeros();
for i = 1:MyIteration
data=load('3品种MCCV.mat').data;
ind = round(0.75 * size(data,1)); %按比例分
RFa=zeros(ind,1);
RFb=zeros(size(data,1)-ind,1);
PLSa=zeros(ind,1);
PLSb=zeros(size(data,1)-ind,1);
GSa=zeros(ind,1);
GSb=zeros(size(data,1)-ind,1);
ABa=zeros(ind,1);
ABb=zeros(size(data,1)-ind,1);
SVMa=zeros(ind,1);
SVMb=zeros(size(data,1)-ind,1);
KNNa=zeros(ind,1);
KNNb=zeros(size(data,1)-ind,1);
NNa=zeros(ind,1);
NNb=zeros(size(data,1)-ind,1);
X=data(:,1:end-1);
% X=Xsnv;
Y=data(:,end);
% %划分数据集
% data=[X Y];

randi=randperm(size(data,1))

datarand = data(randi,:); %随机划分

% data= data(randperm(length(data)));
trainData = datarand(1:ind, 1:end); %训练集
testData = datarand(ind+1:end, 1:end); %测试集
SpecTrain=trainData(:,1:end-1);
SpecTest=testData(:,1:end-1);
ProbTrain=trainData(:,end);
ProbTest=testData(:,end);
% 
% SpecTrain=trainData(:,1:end-1);
% SpecTest=testData(:,1:end-1);
% ProbTrain=trainData(:,end);
% ProbTest=testData(:,end);

% SpecTrain=XTrain;
% SpecTest=XTest;
% ProbTrain=YTrain;
% ProbTest=YTest;

C=[];
% X=data(:,1:end-1);
% Y=data(:,end);
PointCount=size(SpecTrain,1);
c=cvpartition(PointCount,'Kfold',5);
for ModelIndex=1:c.NumTestSets
    TestIndex=test(c,ModelIndex);
    TestProp=ProbTrain(TestIndex,:);
    if(ModelIndex==1)
        C=TestProp;
    else
        C=cat(1,C,TestProp);
    end
end
%MSC
[m,n]=size(X);
x=4000:8000/1844:12000;
y=X;Me=mean(X);
Xmsc=ones(m,n);
for i=1:m
    p=polyfit(Me,X(i,:),1);
    Xmsc(i,:)=(X(i,:)-p(2)*ones(1,n))./(p(1)*ones(1,n));
end
%划分数据
data = [];
data=[Xmsc Y];
datarand = data(randi,:); %随机划分
ind = round(0.75 * size(datarand,1)); %按比例分
% data= data(randperm(length(data)));
trainData = []; %训练集
testData = []; %测试集
SpecTrain=[];
SpecTest=[];
ProbTrain=[];
ProbTest=[];
trainData = datarand(1:ind, 1:end); %训练集
testData = datarand(ind+1:end, 1:end); %测试集
SpecTrain=trainData(:,1:end-1);
SpecTest=testData(:,1:end-1);
ProbTrain=trainData(:,end);
ProbTest=testData(:,end);
%调用随机森林
isCategorical = zeros(size(X,2),1);% Categorical variable flag
treeNum=80;
nodeNum=32;
b= TreeBagger(treeNum,SpecTrain,ProbTrain,'Method','R','OOBPredictorImportance','On',...
    'CategoricalPredictors',find(isCategorical == 1),...
    'MinLeafSize',nodeNum);
Ytest=ProbTest(:,1);
Ypredict=predict(b,SpecTest);
YpredictP=predict(b,SpecTrain);
%模型分析
co=corr(Ytest,Ypredict,'type','Pearson');
RMSE=sqrt(sum((Ytest-Ypredict).^2)/size(Ypredict,1));
R2 = corr(Ytest,Ypredict)^2;
RMSEP=sqrt(sum((ProbTrain-YpredictP).^2)/size(YpredictP,1));
R2P = corr(ProbTrain,YpredictP)^2;
result=[RMSEP;R2P;RMSE;R2];
co=num2str(co);
RMSE=num2str(RMSE);
R2=num2str(R2);
RMSEP=num2str(RMSEP);
R2P=num2str(R2P);
% msgbox({'您运算的结果为:',['RMSEP：',RMSEP,'RMSE：',RMSE],...
%     ['R2P：',R2P,'R2：',R2]},...
%     '随机森林');
% 将运算结果写入文件
RFname = {'RMSEC';'R²C';'RMSEP';'R²P'};
% result=[RMSEP;R2P;RMSE;R2];
Tabl=table(RFname,result);
% writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
RFresult=zeros(4,MyIteration);
RFresult(:,i)=result;
for ModelIndex=1:c.NumTestSets
    TrainIndex=training(c,ModelIndex);
    TestIndex=test(c,ModelIndex);
    TrainData=SpecTrain(TrainIndex,:);
    TrainProp=ProbTrain(TrainIndex,:);
    TestData=SpecTrain(TestIndex,:);
    TestProp=ProbTrain(TestIndex,:);
    b=TreeBagger(treeNum,TrainData,TrainProp,'Method','R','OOBPredictorImportance','On',...
    'CategoricalPredictors',find(isCategorical == 1),...
    'MinLeafSize',nodeNum);
%     b = TreeBagger(treeNum,TrainData,TrainProp,'Method','R','OOBPredictorImportance','On',...
%     'CategoricalPredictors',find(isCategorical == 1),...
%     'MinLeafSize',nodeNum);
    a1=predict(b,TestData);
    b1=predict(b,SpecTest);
    if(ModelIndex==1)
        RFa=a1;
        RFb=b1;
    else
        RFa=cat(1,RFa,a1);
        RFb=RFb+b1;
    end
end
RFb=RFb./c.NumTestSets;
% end

%SNV
% Y = fy;
% X = fx;
[m,n]=size(X);
x=4000:8000/1844:12000;
y=X;
Xm=mean(X,2);
dX=X-repmat(Xm,1,n);
Xsnv=dX./repmat(sqrt(sum(dX.^2,2)/(n-1)),1,n);
%划分数据
data = [];
data=[Xsnv Y];
datarand = data(randi,:); %随机划分
ind = round(0.75 * size(datarand,1)); %按比例分
% data= data(randperm(length(data)));
trainData = []; %训练集
testData = []; %测试集
SpecTrain=[];
SpecTest=[];
ProbTrain=[];
ProbTest=[];
trainData = datarand(1:ind, 1:end); %训练集
testData = datarand(ind+1:end, 1:end); %测试集
SpecTrain=trainData(:,1:end-1);
SpecTest=testData(:,1:end-1);
ProbTrain=trainData(:,end);
ProbTest=testData(:,end);
%PLS
% for count0=1:5
%     data = data(randperm(size(data,1)),:); %随机划分
% ind = round(0.75 * size(data,1)); %按比例分
% % data= data(randperm(length(data)));
% trainData = data(1:ind, 1:end); %训练集
% testData = data(ind+1:end, 1:end); %测试集
% SpecTrain=trainData(:,1:end-1);
% SpecTest=testData(:,1:end-1);
% ProbTrain=trainData(:,end);
% ProbTest=testData(:,end);
[XL,yl,XS,YS,beta] = plsregress(SpecTrain,ProbTrain,10);
Ypredict3 = [ones(size(SpecTest,1),1) SpecTest]*beta;
YpredictP = [ones(size(SpecTrain,1),1) SpecTrain]*beta;
%模型分析
co=corr(ProbTest,Ypredict3,'type','Pearson');
RMSE=sqrt(sum((ProbTest-Ypredict3).^2)/size(Ypredict3,1));
R2 = corr(ProbTest,Ypredict3)^2;
RMSEP=sqrt(sum((ProbTrain-YpredictP).^2)/size(YpredictP,1));
R2P = corr(ProbTrain,YpredictP)^2;
result=[RMSEP;R2P;RMSE;R2];
co=num2str(co);
RMSE=num2str(RMSE);
R2=num2str(R2);
RMSEP=num2str(RMSEP);
R2P=num2str(R2P);
% msgbox({'您运算的结果为:',['RMSEP：',RMSEP,'RMSE：',RMSE],...
%     ['R2P：',R2P,'R2：',R2]},...
%     'PLS');
% 将运算结果写入文件
PLSname = {'RMSEC';'R²C';'RMSEP';'R²P'};
% result=[RMSEP;R2P;RMSE;R2];
Tabl=table(PLSname,result);
% writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
PLSresult=zeros(4,MyIteration);
PLSresult(:,i)=result;
for ModelIndex=1:c.NumTestSets
    TrainIndex=training(c,ModelIndex);
    TestIndex=test(c,ModelIndex);
    TrainData=SpecTrain(TrainIndex,:);
    TrainProp=ProbTrain(TrainIndex,:);
    TestData=SpecTrain(TestIndex,:);
    TestProp=ProbTrain(TestIndex,:);
    [XL,yl,XS,YS,beta] = plsregress(TrainData,TrainProp,10);
    a1 = [ones(size(TestData,1),1) TestData]*beta;
    b1 = [ones(size(SpecTest,1),1) SpecTest]*beta;
    if(ModelIndex==1)
        PLSa=a1;
        PLSb=b1;
    else
        PLSa=cat(1,PLSa,a1);
        PLSb=PLSb+b1;
    end
end
PLSb=PLSb./c.NumTestSets;
% end

% %标准正态变换SNV
% % Y = fy;
% % X = fx;
% [m,n]=size(X);
% x=4000:8000/1844:12000;
% y=X;
% Xm=mean(X,2);
% dX=X-repmat(Xm,1,n);
% Xsnv=dX./repmat(sqrt(sum(dX.^2,2)/(n-1)),1,n);
% 
% %划分数据
% data = [];
% data=[Xsnv Y];
data = [];
data=[X Y];
datarand = data(randi,:); %随机划分
ind = round(0.75 * size(datarand,1)); %按比例分
% data= data(randperm(length(data)));
trainData = []; %训练集
testData = []; %测试集
SpecTrain=[];
SpecTest=[];
ProbTrain=[];
ProbTest=[];
trainData = datarand(1:ind, 1:end); %训练集
testData = datarand(ind+1:end, 1:end); %测试集
SpecTrain=trainData(:,1:end-1);
SpecTest=testData(:,1:end-1);
ProbTrain=trainData(:,end);
ProbTest=testData(:,end);

%高斯过程回归
% for count0=1:5
%     data = data(randperm(size(data,1)),:); %随机划分
% ind = round(0.75 * size(data,1)); %按比例分
% % data= data(randperm(length(data)));
% trainData = data(1:ind, 1:end); %训练集
% testData = data(ind+1:end, 1:end); %测试集
% SpecTrain=trainData(:,1:end-1);
% SpecTest=testData(:,1:end-1);
% ProbTrain=trainData(:,end);
% ProbTest=testData(:,end);
sigma0 = std(ProbTrain);
sigmaF0 = sigma0;
d = size(SpecTrain,2);
sigmaM0 = 10*ones(d,1);
kparams0 = [sigmaM0;sigmaF0];
% sigma0 = 0.2;
% kparams0 = [3.5, 6.2];
gprMdl2 = fitrgp(SpecTrain,ProbTrain,'KernelFunction','ardsquaredexponential', 'KernelParameters',kparams0,'Sigma',sigma0);
Ypredict4=predict(gprMdl2,SpecTest);
YpredictC=predict(gprMdl2,SpecTrain);
%模型分析
co=corr(ProbTest,Ypredict4,'type','Pearson');
RMSE=sqrt(sum((ProbTest-Ypredict4).^2)/size(Ypredict4,1));
R2 = corr(ProbTest,Ypredict4)^2;
RMSEC=sqrt(sum((ProbTrain-YpredictC).^2)/size(YpredictC,1));
R2C = corr(ProbTrain,YpredictC)^2;
result=[RMSEC;R2C;RMSE;R2];
co=num2str(co);
RMSE=num2str(RMSE);
R2=num2str(R2);
RMSEC=num2str(RMSEC);
R2C=num2str(R2C);
% msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
%     ['R2C：',R2C,'R2：',R2]},...
%     'GS');
% 将运算结果写入文件
GPRname = {'RMSEC';'R²C';'RMSEP';'R²P'};
% result=[RMSEP;R2P;RMSE;R2];
Tabl=table(GPRname,result);
% writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
GPRresult=zeros(4,MyIteration);
GPRresult(:,i)=result;
for ModelIndex=1:c.NumTestSets
    TrainIndex=training(c,ModelIndex);
    TestIndex=test(c,ModelIndex);
    TrainData=SpecTrain(TrainIndex,:);
    TrainProp=ProbTrain(TrainIndex,:);
    TestData=SpecTrain(TestIndex,:);
    TestProp=ProbTrain(TestIndex,:);
    gprMdl2 = fitrgp(TrainData,TrainProp,'KernelFunction','ardsquaredexponential',...
        'KernelParameters',kparams0,'Sigma',sigma0);
    a1=predict(gprMdl2,TestData);
    b1=predict(gprMdl2,SpecTest);
    if(ModelIndex==1)
        GSa=a1;
        GSb=b1;
    else
        GSa=cat(1,GSa,a1);
        GSb=GSb+b1;
    end
end
GSb=GSb./c.NumTestSets;
% end

% % 数据预处理
% X2st=diff(X,2,2);
% 
% %划分数据
% data = [];
% data=[Xsnv Y];
%MSC
[m,n]=size(X);
x=4000:8000/1844:12000;
y=X;Me=mean(X);
Xmsc=ones(m,n);
for i=1:m
    p=polyfit(Me,X(i,:),1);
    Xmsc(i,:)=(X(i,:)-p(2)*ones(1,n))./(p(1)*ones(1,n));
end
%划分数据
data = [];
data=[Xmsc Y];

datarand = data(randi,:); %随机划分
ind = round(0.75 * size(datarand,1)); %按比例分
% data= data(randperm(length(data)));
trainData = []; %训练集
testData = []; %测试集
SpecTrain=[];
SpecTest=[];
ProbTrain=[];
ProbTest=[];
trainData = datarand(1:ind, 1:end); %训练集
testData = datarand(ind+1:end, 1:end); %测试集
SpecTrain=trainData(:,1:end-1);
SpecTest=testData(:,1:end-1);
ProbTrain=trainData(:,end);
ProbTest=testData(:,end);
%Adaboost
% for count0=1:5
%     data = data(randperm(size(data,1)),:); %随机划分
% ind = round(0.75 * size(data,1)); %按比例分
% % data= data(randperm(length(data)));
% trainData = data(1:ind, 1:end); %训练集
% testData = data(ind+1:end, 1:end); %测试集
% SpecTrain=trainData(:,1:end-1);
% SpecTest=testData(:,1:end-1);
% ProbTrain=trainData(:,end);
% ProbTest=testData(:,end);
ABtreeNum=100;
ABnodeNum=5;
learnRate=0.02;
t = templateTree('MaxNumSplits',ABnodeNum,'Surrogate','on');
RTreeEns=fitensemble(SpecTrain,ProbTrain,'LSBoost',ABtreeNum,t,...
            'Type','regression','LearnRate',learnRate);
Ytest=ProbTest(:,1);
Ypredict=predict(RTreeEns,SpecTest);
YpredictC=predict(RTreeEns,SpecTrain);
co=corr(Ytest,Ypredict,'type','Pearson');
RMSE=sqrt(sum((Ytest-Ypredict).^2)/size(Ypredict,1));
R2 = corr(Ytest,Ypredict)^2;
RMSEC=sqrt(sum((ProbTrain-YpredictC).^2)/size(YpredictC,1));
R2C = corr(ProbTrain,YpredictC)^2;
result=[RMSEC;R2C;RMSE;R2];
co=num2str(co);
RMSE=num2str(RMSE);
R2=num2str(R2);
RMSEC=num2str(RMSEC);
R2C=num2str(R2C);
% msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
%     ['R2C：',R2C,'R2：',R2]},...
%     'AdaBoost');
% 将运算结果写入文件
Adaboostname = {'RMSEC';'R²C';'RMSEP';'R²P'};
% result=[RMSEP;R2P;RMSE;R2];
Tabl=table(Adaboostname,result);
% writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
ABresult=zeros(4,MyIteration);
ABresult(:,i)=result;
for ModelIndex=1:c.NumTestSets
    TrainIndex=training(c,ModelIndex);
    TestIndex=test(c,ModelIndex);
    TrainData=SpecTrain(TrainIndex,:);
    TrainProp=ProbTrain(TrainIndex,:);
    TestData=SpecTrain(TestIndex,:);
    TestProp=ProbTrain(TestIndex,:);  
    t = templateTree('MaxNumSplits',ABnodeNum,'Surrogate','on');
    RTreeEns=fitensemble(TrainData,TrainProp,'LSBoost',ABtreeNum,t,...
            'Type','regression','LearnRate',learnRate);
    a1=predict(RTreeEns,TestData);
    b1=predict(RTreeEns,SpecTest);
    if(ModelIndex==1)
        ABa=a1;
        ABb=b1;
    else
        ABa=cat(1,ABa,a1);
        ABb=ABb+b1;
    end
end
ABb=ABb./c.NumTestSets;
% end

% First derivative preprocessing
X1st=FirstDerivative(X);
%划分数据
data = [];
data=[X1st Y];

datarand = data(randi,:); %随机划分
ind = round(0.75 * size(datarand,1)); %按比例分
% data= data(randperm(length(data)));
trainData = []; %训练集
testData = []; %测试集
SpecTrain=[];
SpecTest=[];
ProbTrain=[];
ProbTest=[];
trainData = datarand(1:ind, 1:end); %训练集
testData = datarand(ind+1:end, 1:end); %测试集
SpecTrain=trainData(:,1:end-1);
SpecTest=testData(:,1:end-1);
ProbTrain=trainData(:,end);
ProbTest=testData(:,end);
%SVM
% for count0=1:5
%     data = data(randperm(size(data,1)),:); %随机划分
% ind = round(0.75 * size(data,1)); %按比例分
% % data= data(randperm(length(data)));
% trainData = data(1:ind, 1:end); %训练集
% testData = data(ind+1:end, 1:end); %测试集
% SpecTrain=trainData(:,1:end-1);
% SpecTest=testData(:,1:end-1);
% ProbTrain=trainData(:,end);
% ProbTest=testData(:,end);
MdlGau = fitrsvm(SpecTrain,ProbTrain,'KernelFunction','gaussian','OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('Optimizer','gridsearch',...
    'ShowPlots',false));
Ypredict3=predict(MdlGau,SpecTest);
YpredictP=predict(MdlGau,SpecTrain);
%模型分析
co=corr(ProbTest,Ypredict3,'type','Pearson');
RMSE=sqrt(sum((ProbTest-Ypredict3).^2)/size(Ypredict3,1));
R2 = corr(ProbTest,Ypredict3)^2;
RMSEP=sqrt(sum((ProbTrain-YpredictP).^2)/size(YpredictP,1));
R2P = corr(ProbTrain,YpredictP)^2;
result=[RMSEP;R2P;RMSE;R2];
co=num2str(co);
RMSE=num2str(RMSE);
R2=num2str(R2);
RMSEP=num2str(RMSEP);
R2P=num2str(R2P);
% msgbox({'您运算的结果为:',['RMSEP：',RMSEP,'RMSE：',RMSE],...
%     ['R2P：',R2P,'R2：',R2]},...
%     'SVM');
% 将运算结果写入文件
SVMname = {'RMSEC';'R²C';'RMSEP';'R²P'};
% result=[RMSEP;R2P;RMSE;R2];
Tabl=table(SVMname,result);
% writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
SVMresult=zeros(4,MyIteration);
SVMresult(:,i)=result;
for ModelIndex=1:c.NumTestSets
    TrainIndex=training(c,ModelIndex);
    TestIndex=test(c,ModelIndex);
    TrainData=SpecTrain(TrainIndex,:);
    TrainProp=ProbTrain(TrainIndex,:);
    TestData=SpecTrain(TestIndex,:);
    TestProp=ProbTrain(TestIndex,:);
    MdlGau = fitrsvm(TrainData,TrainProp,'KernelFunction','gaussian','OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('Optimizer','gridsearch','ShowPlots',false,'MaxObjectiveEvaluations',60));
    a1=predict(MdlGau,TestData);
    b1=predict(MdlGau,SpecTest);
     if(ModelIndex==1)
        SVMa=a1;
        SVMb=b1;
    else
        SVMa=cat(1,SVMa,a1);
        SVMb=SVMb+b1;
     end
end
SVMb=SVMb./c.NumTestSets;
% end


% %标准正态变换SNV
% % Y = fy;
% % X = fx;
% [m,n]=size(X);
% x=4000:8000/1844:12000;
% y=X;
% Xm=mean(X,2);
% dX=X-repmat(Xm,1,n);
% Xsnv=dX./repmat(sqrt(sum(dX.^2,2)/(n-1)),1,n);
% 
% %划分数据
% data = [];
% data=[Xsnv Y];
datarand = data(randi,:); %随机划分
ind = round(0.75 * size(datarand,1)); %按比例分
% data= data(randperm(length(data)));
trainData = []; %训练集
testData = []; %测试集
SpecTrain=[];
SpecTest=[];
ProbTrain=[];
ProbTest=[];
trainData = datarand(1:ind, 1:end); %训练集
testData = datarand(ind+1:end, 1:end); %测试集
SpecTrain=trainData(:,1:end-1);
SpecTest=testData(:,1:end-1);
ProbTrain=trainData(:,end);
ProbTest=testData(:,end);
%KNN
% for k= 11:2:33
    k=24;
    x = SpecTrain(:,:);
    Mdl = KDTreeSearcher(x);
    [n,~] = knnsearch(Mdl,SpecTest(:,:),'k',k);
    resultClass=zeros(size(n,1),1);
    for j = 1:size(n,1)
        tempClass = ProbTrain(n(j,:),:);
    %     result = mode(tempClass);
    %     resultClass(j,1) = result;
        result = sum(tempClass)./k;
        resultClass(j,1) = result;
     end
    %  validate = sum( ProbTest(:,:) == resultClass )./ size(SpecTest,1) * 100;
    %  R2=corr(ProbTest,resultClass)^2;
     RMSE=sqrt(sum((ProbTest-resultClass).^2)/size(resultClass,1));
    R2 = corr(ProbTest,resultClass)^2;
    result=[RMSE;R2];
    % RMSEP=sqrt(sum((ProbTrain-YpredictP).^2)/size(YpredictP,1));
    % R2P = corr(ProbTrain,YpredictP)^2;
    % co=num2str(co);
    RMSE=num2str(RMSE);
    R2=num2str(R2);
    % RMSEP=num2str(RMSEP);
    % R2P=num2str(R2P);
%     msgbox({'您运算的结果为:',['K:',k],['RMSE：',RMSE],...
%         ['R2：',R2]},...
%         'KNN');
    % 将运算结果写入文件
KNNname = {'RMSEP';'R²P'};
% result=[RMSEP;R2P;RMSE;R2];
Tabl=table(KNNname,result);
% writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
KNNresult=zeros(2,MyIteration);
KNNresult(:,i)=result;
% end

resultClassa=[];
resultClassb=[];
for ModelIndex=1:c.NumTestSets
    TrainIndex=training(c,ModelIndex);
    TestIndex=test(c,ModelIndex);
    TrainData=SpecTrain(TrainIndex,:);
    TrainProp=ProbTrain(TrainIndex,:);
    TestData=SpecTrain(TestIndex,:);
    TestProp=ProbTrain(TestIndex,:);
     x = TrainData(:,:);
    Mdl = KDTreeSearcher(x);
    [n,~] = knnsearch(Mdl,TestData(:,:),'k',k);
    for j = 1:size(n,1)
        tempClassa = TrainProp(n(j,:),:);
    %     result = mode(tempClass);
    %     resultClass(j,1) = result;
        resulta = sum(tempClassa)./k;
        resultClassa(j,1) = resulta;
    end
   
    [n,~] = knnsearch(Mdl,SpecTest(:,:),'k',k);
    for j = 1:size(n,1)
          tempClassb = TrainProp(n(j,:),:);
    %     result = mode(tempClass);
    %     resultClass(j,1) = result;
        resultb = sum(tempClassb)./k;
        resultClassb(j,1) = resultb;
     end
    a1=resultClassa;
    b1=resultClassb;
    resultClassa=[];
    resultClassb=[];
     if(ModelIndex==1)
        KNNa=a1;
        KNNb=b1;
    else
        KNNa=cat(1,KNNa,a1);
        KNNb=KNNb+b1;
     end
end
KNNb=KNNb./c.NumTestSets;

%多元散射校正MSC
[m,n]=size(X);
x=4000:8000/1844:12000;
y=X;Me=mean(X);
Xmsc=ones(m,n);
for e=1:m
   p=polyfit(Me,X(e,:),1); 
   Xmsc(e,:)=(X(e,:)-p(2)*ones(1,n))./(p(1)*ones(1,n));
end
%划分数据
data = [];
data=[Xmsc Y];
datarand = data(randi,:); %随机划分
ind = round(0.75 * size(datarand,1)); %按比例分
% data= data(randperm(length(data)));
trainData = []; %训练集
testData = []; %测试集
SpecTrain=[];
SpecTest=[];
ProbTrain=[];
ProbTest=[];
trainData = datarand(1:ind, 1:end); %训练集
testData = datarand(ind+1:end, 1:end); %测试集
SpecTrain=trainData(:,1:end-1);
SpecTest=testData(:,1:end-1);
ProbTrain=trainData(:,end);
ProbTest=testData(:,end);
%前馈神经网络初级模型
% for count0=1:5
%     data = data(randperm(size(data,1)),:); %随机划分
% ind = round(0.75 * size(data,1)); %按比例分
% % data= data(randperm(length(data)));
% trainData = data(1:ind, 1:end); %训练集
% testData = data(ind+1:end, 1:end); %测试集
% SpecTrain=trainData(:,1:end-1);
% SpecTest=testData(:,1:end-1);
% ProbTrain=trainData(:,end);
% ProbTest=testData(:,end);
net1 = feedforwardnet(5,'trainbr');
net1.trainParam.showWindow=0;
% net1 = configure(net1,SpecTrain,ProbTrain);
NN=train(net1,SpecTrain',ProbTrain','showResources','yes');
Ypredict=NN(SpecTest');
YpredictC=NN(SpecTrain');
%模型分析
RMSE=sqrt(sum((ProbTest-Ypredict').^2)/size(Ypredict',1));
R2 = corr(ProbTest,Ypredict')^2;
RMSEC=sqrt(sum((ProbTrain-YpredictC').^2)/size(YpredictC',1));
R2C = corr(ProbTrain,YpredictC')^2;
result=[RMSEC;R2C;RMSE;R2];
RMSE=num2str(RMSE);
R2=num2str(R2);
RMSEC=num2str(RMSEC);
R2C=num2str(R2C);
% msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
%     ['R2C：',R2C,'R2：',R2]},...
%     '神经网络');
 % 将运算结果写入文件
FNNname = {'RMSEC';'R²C';'RMSEP';'R²P'};
% result=[RMSEP;R2P;RMSE;R2];
Tabl=table(FNNname,result);
% writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
FNNresult=zeros(4,MyIteration);
FNNresult(:,i)=result;
for ModelIndex=1:c.NumTestSets
    TrainIndex=training(c,ModelIndex);
    TestIndex=test(c,ModelIndex);
    TrainData=SpecTrain(TrainIndex,:);
    TrainProp=ProbTrain(TrainIndex,:);
    TestData=SpecTrain(TestIndex,:);
    TestProp=ProbTrain(TestIndex,:);
    net2 = feedforwardnet(5,'trainbr');
    net2.trainParam.showWindow=0;
%     net2 = configure(net2,TrainData,TrainProp);
    NN=train(net2,TrainData',TrainProp');
    a1=NN(TestData');
    b1=NN(SpecTest');
    if(ModelIndex==1)
        NNa=a1';
        NNb=b1';
    else
        NNa=cat(1,NNa,a1');
        NNb=NNb+b1';
    end
end
NNb=NNb./c.NumTestSets;
% end
 
A0=[RFa SVMa GSa PLSa ABa KNNa NNa];
B0=[RFb SVMb GSb PLSb ABb KNNb NNb];
A1=[SVMa GSa PLSa];
B1=[SVMb GSb PLSb];
RFA=zeros(ind,MyIteration);
RFA(:,i)=RFa;
RFB=zeros(size(data,1)-ind,MyIteration);
RFB(:,i)=RFb;
SVMA=zeros(ind,MyIteration);
SVMA(:,i)=SVMa;
SVMB=zeros(size(data,1)-ind,MyIteration);
SVMB(:,i)=SVMb;
GSA=zeros(ind,MyIteration);
GSA(:,i)=GSa;
GSB=zeros(size(data,1)-ind,MyIteration);
GSB(:,i)=GSb;
PLSA=zeros(ind,MyIteration);
PLSA(:,i)=PLSa;
PLSB=zeros(size(data,1)-ind,MyIteration);
PLSB(:,i)=PLSb;
ABA=zeros(ind,MyIteration);
ABA(:,i)=ABa;
ABB=zeros(size(data,1)-ind,MyIteration);
ABB(:,i)=ABb;
KNNA=zeros(ind,MyIteration);
KNNA(:,i)=KNNa;
KNNB=zeros(size(data,1)-ind,MyIteration);
KNNB(:,i)=KNNb;
FNNA=zeros(ind,MyIteration);
FNNA(:,i)=NNa;
FNNB=zeros(size(data,1)-ind,MyIteration);
FNNB(:,i)=NNb;

% y=[];
%迭代轮次
iteration=30;
%Name:genmain05.m
% clear
% clf
% popsize=9;                                       %群体大小
% chromlength=7;                                   %字符串长度（个体长度）
popsize=30;                                       %群体大小
chromlength=9;                                   %字符串长度（个体长度）
count = zeros(iteration,chromlength);
pc=0.6;                                           %交叉概率
pm=0.001;                                         %变异概率
pop=initpop(popsize,chromlength);                 %随机产生初始群体
bestindividuals=zeros(iteration,chromlength);     %每轮最好的特征选择
 Metaresult=zeros(4,popsize);
  bestindividual=zeros(1,chromlength);
     bestfitNum=1;
for ii=1:iteration                                    %20为迭代次数
%     objvalue(i,1)=R2/abs(1-(R2*R2)/(R2C*R2C));
       [objvalue,Metaresult]=calobjvalue3(pop,A0,C,B0,ProbTest);                      %计算目标函数
    fitvalue=objvalue;                   %计算群体中每个个体的适应度
    [newpop]=selection(pop,fitvalue);                 %复制
    [newpop]=crossover_multiv(newpop,pc);                       %交叉
    [newpop]=mutation(newpop,pc);                        %变异
%     pop=newpop;
        [bestindividual,bestfit,bestfitNum]=best(pop,fitvalue);%求出群体中适应值最大的个体及其适应值
%     bestp(ii)=max(bestfit);
%     bestindividuals(ii,:)=bestindividual;
%     nn(ii)=ii;
% %     pop5=bestindividual;
% %     x(ii)=decodechrom(pop5,1,chromlength)*10/1023;
    pop=newpop;
%     count(ii,:)=sum(pop,1);
end
% [M ,I]=max(bestp);
% Bestindividual=bestindividuals(I,:);
ind=logical(bestindividual(1:7));
%     newSpec=A0(:,ind);
%     newtestSpec=B0(:,ind);
    num=num2str(bestindividual(8:9));%次级模型编号
    temp1=bin2dec(num) %将编号转化成十进制数
     result=Metaresult(:,bestfitNum);
        RMSE=result(1);
        R2 = result(2);
        RMSEC=result(3);
        R2C = result(4);
    if temp1==0
        RMSE=num2str(roundn(RMSE,-4));
        R2=num2str(roundn(R2,-4));
        RMSEC=num2str(roundn(RMSEC,-4));
        R2C=num2str(roundn(R2C,-4));
         msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
            ['R2C：',R2C,'R2：',R2]},...
            '次级模型高斯回归');
    end
    if temp1==1
        RMSE=num2str(roundn(RMSE,-4));
    R2=num2str(roundn(R2,-4));
    RMSEC=num2str(roundn(RMSEC,-4));
    R2C=num2str(roundn(R2C,-4));
    msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
        ['R2C：',R2C,'R2：',R2]},...
        '次级模型SVM');  
    end
     if temp1==2
          RMSE=num2str(roundn(RMSE,-4));
    R2=num2str(roundn(R2,-4));
    RMSEC=num2str(roundn(RMSEC,-4));
    R2C=num2str(roundn(R2C,-4));
    msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
        ['R2C：',R2C,'R2：',R2]},...
        '次级模型PLS');
     end
     if temp1==3
          RMSE=num2str(roundn(RMSE,-4));
    R2=num2str(roundn(R2,-4));
    RMSEC=num2str(roundn(RMSEC,-4));
    R2C=num2str(roundn(R2C,-4));
    msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
        ['R2C：',R2C,'R2：',R2]},...
        '次级随机森林');
     end
     GAStackingresult=zeros(4,MyIteration);
     GAStackingresult(:,i)=result;
     
     %选三个最好的单学习器次级随机森林
isCategorical = zeros(size(X,2),1);
treeNum=100;
nodeNum=15;
Stacking = TreeBagger(treeNum,A1,C,'Method','R','OOBPredictorImportance','On',...
    'CategoricalPredictors',find(isCategorical == 1),...
    'MinLeafSize',nodeNum);
Ypredict5=predict(Stacking,B1);
YpredictC=predict(Stacking,A1);
%模型分析
co=corr(ProbTest,Ypredict5,'type','Pearson');
RMSE=sqrt(sum((ProbTest-Ypredict5).^2)/size(Ypredict5,1));
R2 = corr(ProbTest,Ypredict5)^2;
RMSEC=sqrt(sum((C-YpredictC).^2)/size(YpredictC,1));
R2C = corr(C,YpredictC)^2;
result=[RMSEC;R2C;RMSE;R2];
co=num2str(roundn(co,-4));
RMSE=num2str(roundn(RMSE,-4));
R2=num2str(roundn(R2,-4));
RMSEC=num2str(roundn(RMSEC,-4));
R2C=num2str(roundn(R2C,-4));
msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
    ['R2C：',R2C,'R2：',R2]},...
    '次级RF1');
 % 将运算结果写入文件
FNNname = {'RMSEC';'R²C';'RMSEP';'R²P'};
% result=[RMSEP;R2P;RMSE;R2];
Tabl=table(FNNname,result);
% writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
RFMetaresult=zeros(4,MyIteration);
     RFMetaresult(:,i)=result;
    %选三个效果好的单模型次级模型SVM
Stacking = fitrsvm(A1,C,'KernelFunction','gaussian','OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('Optimizer','gridsearch','NumGridDivisions',15,...
    'ShowPlots',false,'MaxObjectiveEvaluations',60));
Ypredict5=predict(Stacking,B1);
YpredictC=predict(Stacking,A1);
%模型分析
co=corr(ProbTest,Ypredict5,'type','Pearson');
RMSE=sqrt(sum((ProbTest-Ypredict5).^2)/size(Ypredict5,1));
R2 = corr(ProbTest,Ypredict5)^2;
RMSEC=sqrt(sum((C-YpredictC).^2)/size(YpredictC,1));
R2C = corr(C,YpredictC)^2;
result=[RMSEC;R2C;RMSE;R2];
co=num2str(roundn(co,-4));
RMSE=num2str(roundn(RMSE,-4));
R2=num2str(roundn(R2,-4));
RMSEC=num2str(roundn(RMSEC,-4));
R2C=num2str(roundn(R2C,-4));
msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
    ['R2C：',R2C,'R2：',R2]},...
    '次级模型SVM');
 % 将运算结果写入文件
FNNname = {'RMSEC';'R²C';'RMSEP';'R²P'};
% result=[RMSEP;R2P;RMSE;R2];
Tabl=table(FNNname,result);
% writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
SVMMetaresult=zeros(4,MyIteration);
     SVMMetaresult(:,i)=result;

%选三个效果好的单模型次级模型PLS
Stacking = fitlm(A1,C);
Ypredict5=predict(Stacking,B1);
YpredoctA=predict(Stacking,A1);
%模型分析
co=corr(ProbTest,Ypredict5,'type','Pearson');
RMSE=sqrt(sum((ProbTest-Ypredict5).^2)/size(Ypredict5,1));
R2 = corr(ProbTest,Ypredict5)^2;
coP=corr(C,YpredoctA,'type','Pearson');
RMSEC=sqrt(sum((C-YpredoctA).^2)/size(YpredoctA,1));
R2C = corr(C,YpredoctA)^2;
result=[RMSEC;R2C;RMSE;R2];
co=num2str(roundn(co,-4));
RMSE=num2str(roundn(RMSE,-4));
R2=num2str(roundn(R2,-4));
RMSEC=num2str(roundn(RMSEC,-4));
R2C=num2str(roundn(R2C,-4));
msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
    ['R2C：',R2C,'R2：',R2]},...
    '次级模型PLS');
% 将运算结果写入文件
FNNname = {'RMSEC';'R²C';'RMSEP';'R²P'};
% result=[RMSEP;R2P;RMSE;R2];
Tabl=table(FNNname,result);
% writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
PLSMetaresult=zeros(4,MyIteration);
    PLSMetaresult(:,i)=result;

%选三个效果好的单模型次级模型高斯回归
sigma0 = 0.2;
kparams0 = [3.5, 6.2];
stacking = fitrgp(A1,C,'KernelFunction','squaredexponential',...
        'KernelParameters',kparams0,'Sigma',sigma0);
Ypredict5=predict(stacking,B1);
YpredictC=predict(stacking,A1);
%模型分析
co=corr(ProbTest,Ypredict5,'type','Pearson');
RMSE=sqrt(sum((ProbTest-Ypredict5).^2)/size(Ypredict5,1));
R2 = corr(ProbTest,Ypredict5)^2;
RMSEC=sqrt(sum((C-YpredictC).^2)/size(YpredictC,1));
R2C = corr(C,YpredictC)^2;
result=[RMSEC;R2C;RMSE;R2];
co=num2str(roundn(co,-4));
RMSE=num2str(roundn(RMSE,-4));
R2=num2str(roundn(R2,-4));
RMSEC=num2str(roundn(RMSEC,-4));
R2C=num2str(roundn(R2C,-4));
msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
    ['R2C：',R2C,'R2：',R2]},...
    '次级模型高斯回归');
% 将运算结果写入文件
FNNname = {'RMSEC';'R²C';'RMSEP';'R²P'};
% result=[RMSEP;R2P;RMSE;R2];
Tabl=table(FNNname,result);
writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
RFMetaresult=zeros(4,MyIteration);
     RFMetaresult(:,i)=result;
end
for i=1:80
    co=corr(ABA80(1:36,i)',CMatrix80(1:36,i)','type','Pearson');
    RMSEP(1,i)=sqrt(sum((ABA80(1:36,i)-CMatrix80(1:36,i)).^2)/size(CMatrix80(1:36,i),1));
    R2P(1,i) = corr(ABA80(1:36,i),CMatrix80(1:36,i))^2;
end
for i=1:80
    co=corr(ABA80(37:72,i),CMatrix80(37:72,i),'type','Pearson');
    RMSEP(2,i)=sqrt(sum((ABA80(37:72,i)-CMatrix80(37:72,i)).^2)/size(CMatrix80(37:72,i),1));
    R2P(2,i) = corr(ABA80(37:72,i),CMatrix80(37:72,i))^2;
end
for i=1:80
    co=corr(ABA80(73:108,i),CMatrix80(73:108,i),'type','Pearson');
    RMSEP(3,i)=sqrt(sum((ABA80(73:108,i)-CMatrix80(73:108,i)).^2)/size(CMatrix80(73:108,i),1));
    R2P(3,i) = corr(ABA80(73:108,i),CMatrix80(73:108,i))^2;
end
for i=1:80
    co=corr(ABA80(109:144,i),CMatrix80(109:144,i),'type','Pearson');
    RMSEP(4,i)=sqrt(sum((ABA80(109:144,i)-CMatrix80(109:144,i)).^2)/size(CMatrix80(109:144,i),1));
    R2P(4,i) = corr(ABA80(109:144,i),CMatrix80(109:144,i))^2;
end
for i=1:80
    co=corr(ABA80(145:end,i),CMatrix80(145:end,i),'type','Pearson');
    RMSEP(5,i)=sqrt(sum((ABA80(145:end,i)-CMatrix80(145:end,i)).^2)/size(CMatrix80(145:end,i),1));
    R2P(5,i) = corr(ABA80(145:end,i),CMatrix80(145:end,i))^2;
end
RMSEPAB=mean((RMSEP(1,:)+RMSEP(2,:)+RMSEP(3,:)+RMSEP(4,:)+RMSEP(5,:))./5)
R2PAB=mean((R2P(1,:)+R2P(2,:)+R2P(3,:)+R2P(4,:)+R2P(5,:))./5)
RMSEC=sqrt(sum((ProbTrain-YpredictC).^2)/size(YpredictC,1));
R2C = corr(ProbTrain,YpredictC)^2;
result=[RMSEC;R2C;RMSEP;R2P];
co=num2str(co);
RMSEP=num2str(RMSEP);
R2P=num2str(R2P);
RMSEC=num2str(RMSEC);
R2C=num2str(R2C);
% % fplot('10*sin(5*x)+7*cos(4*x)',[0 10])
% % plot();
% % hold on
%   
% % hold off
% [M ,I]=max(bestp);
% Bestindividual=bestindividuals(I,:);
% ind=logical(Bestindividual(1:7));
%     newSpec=A0(:,ind);
%     newtestSpec=B0(:,ind);
%     num=num2str(Bestindividual(8:9));%次级模型编号
%     temp1=bin2dec(num) %将编号转化成十进制数
%     if temp1==0
%         sigma0 = 0.2;
%         kparams0 = [3.5, 6.2];
%         stacking = fitrgp(newSpec,C,'KernelFunction','squaredexponential',...
%             'KernelParameters',kparams0,'Sigma',sigma0);
%         Ypredict5=predict(stacking,newtestSpec);
%         YpredictC=predict(stacking,newSpec);
%         %模型分析
%         co=corr(ProbTest,Ypredict5,'type','Pearson');
%         RMSE=sqrt(sum((ProbTest-Ypredict5).^2)/size(Ypredict5,1));
%         R2 = corr(ProbTest,Ypredict5)^2;
%         RMSEC=sqrt(sum((C-YpredictC).^2)/size(YpredictC,1));
%         R2C = corr(C,YpredictC)^2;
% %         RMSE=num2str(roundn(RMSE,-4));
%         result=[RMSEC;R2C;RMSE;R2];
%         co=num2str(roundn(co,-4));
%         RMSE=num2str(roundn(RMSE,-4));
%         R2=num2str(roundn(R2,-4));
%         RMSEC=num2str(roundn(RMSEC,-4));
%         R2C=num2str(roundn(R2C,-4));
%         msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
%             ['R2C：',R2C,'R2：',R2]},...
%             '次级模型高斯回归');
% %         次级模型运算结果写入文件
%         GPRmetaName = {'RMSEC';'R²C';'RMSEP';'R²P'};
%         % result=[RMSEP;R2P;RMSE;R2];
%         Tabl=table(GPRmetaName,result);
% %         writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
% %         writetable(Tabl,'m.xls','Range','C34:D38','WriteVariableNames',true);
%     end
%     if temp1==1
%         Stacking = fitrsvm(newSpec,C,'KernelFunction','gaussian','OptimizeHyperparameters','auto',...
%     'HyperparameterOptimizationOptions',struct('Optimizer','gridsearch','NumGridDivisions',15,...
%     'ShowPlots',false,'MaxObjectiveEvaluations',60));
%     Ypredict5=predict(Stacking,newtestSpec);
%     YpredictC=predict(Stacking,newSpec);
%     %模型分析
%     co=corr(ProbTest,Ypredict5,'type','Pearson');
%     RMSE=sqrt(sum((ProbTest-Ypredict5).^2)/size(Ypredict5,1));
%     R2 = corr(ProbTest,Ypredict5)^2;
%     RMSEC=sqrt(sum((C-YpredictC).^2)/size(YpredictC,1));
%     R2C = corr(C,YpredictC)^2;
%     result=[RMSEC;R2C;RMSE;R2];
%     co=num2str(roundn(co,-4));
%     RMSE=num2str(roundn(RMSE,-4));
%     R2=num2str(roundn(R2,-4));
%     RMSEC=num2str(roundn(RMSEC,-4));
%     R2C=num2str(roundn(R2C,-4));
%     msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
%         ['R2C：',R2C,'R2：',R2]},...
%         '次级模型SVM');
%     %         次级模型运算结果写入文件
%         SVMmetaName = {'RMSEC';'R²C';'RMSEP';'R²P'};
%         % result=[RMSEP;R2P;RMSE;R2];
%         Tabl=table(SVMmetaName,result);
% %               writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
% %         writetable(Tabl,'m.xls','Range','C34:D38','WriteVariableNames',true);
%     end
%     if temp1==2
%          Stacking = fitlm(newSpec,C);
%     Ypredict5=predict(Stacking,newtestSpec);
%     YpredoctA=predict(Stacking,newSpec);
%     %模型分析
%     co=corr(ProbTest,Ypredict5,'type','Pearson');
%     RMSE=sqrt(sum((ProbTest-Ypredict5).^2)/size(Ypredict5,1));
%     R2 = corr(ProbTest,Ypredict5)^2;
% %     coP=corr(prob,YpredoctA,'type','Pearson');
%     RMSEP=sqrt(sum((C-YpredoctA).^2)/size(YpredoctA,1));
%     R2P = corr(C,YpredoctA)^2;
%     result=[RMSEP;R2P;RMSE;R2];
%     co=num2str(roundn(co,-4));
%     RMSE=num2str(roundn(RMSE,-4));
%     R2=num2str(roundn(R2,-4));
%     RMSEP=num2str(roundn(RMSEP,-4));
%     R2P=num2str(roundn(R2P,-4));
%     msgbox({'您运算的结果为:',['RMSEC：',RMSEP,'RMSE：',RMSE],...
%         ['R2C：',R2P,'R2：',R2]},...
%         '次级模型PLS');
%      %         次级模型运算结果写入文件
%         SVMmetaName = {'RMSEC';'R²C';'RMSEP';'R²P'};
%         % result=[RMSEP;R2P;RMSE;R2];
%         Tabl=table(SVMmetaName,result);
% %                 writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
% %         writetable(Tabl,'m.xls','Range','C34:D38','WriteVariableNames',true);
%     end
%     if temp1==3
%     isCategorical = zeros(size(newSpec,2),1);
%     treeNum=100;
%     nodeNum=15;
%     Stacking = TreeBagger(treeNum,newSpec,C,'Method','R','OOBPredictorImportance','On',...
%     'CategoricalPredictors',find(isCategorical == 1),...
%     'MinLeafSize',nodeNum);
%     Ypredict5=predict(Stacking,newtestSpec);
%     YpredictC=predict(Stacking,newSpec);
%     %模型分析
%     co=corr(ProbTest,Ypredict5,'type','Pearson');
%     RMSE=sqrt(sum((ProbTest-Ypredict5).^2)/size(Ypredict5,1));
%     R2 = corr(ProbTest,Ypredict5)^2;
%     RMSEC=sqrt(sum((C-YpredictC).^2)/size(YpredictC,1));
%     R2C = corr(C,YpredictC)^2;
%     result=[RMSEC;R2C;RMSE;R2];
%         RMSE=num2str(roundn(RMSE,-4));
%     R2=num2str(roundn(R2,-4));
%     RMSEC=num2str(roundn(RMSEC,-4));
%     R2C=num2str(roundn(R2C,-4));
%     msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
%         ['R2C：',R2C,'R2：',R2]},...
%         '次级随机森林');
%      %         次级模型运算结果写入文件
%        RFmetaName = {'RMSEC';'R²C';'RMSEP';'R²P'};
%         % result=[RMSEP;R2P;RMSE;R2];
%         Tabl=table(RFmetaName,result);
% %                writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
% %         writetable(Tabl,'m.xls','Range','C34:D38','WriteVariableNames',true);
%     end
%     Metaresult=zeros(4,iteration);
%     Metaresult(:,i)=result;
% % end
% %迭代轮次
% iteration=50;
% %Name:genmain05.m
% % clear
% % clf
% % popsize=9;                                       %群体大小
% % chromlength=7;                                   %字符串长度（个体长度）
% popsize=30;                                       %群体大小
% chromlength=9;                                   %字符串长度（个体长度）
% count = zeros(iteration,chromlength);
% pc=0.6;                                           %交叉概率
% pm=0.001;                                         %变异概率
% pop=initpop(popsize,chromlength);                 %随机产生初始群体
% bestindividuals=zeros(iteration,chromlength);     %每轮最好的特征选择
% for ii=1:iteration                                    %20为迭代次数
% %     objvalue(i,1)=(R2*R2) /abs(Im*(1-(R2*R2)/(R2C*R2C)))
%     Im=0.5;
%     [objvalue]=calobjvalue3(pop,A0,C,B0,ProbTest,Im);                      %计算目标函数
%     fitvalue=objvalue;                   %计算群体中每个个体的适应度
%     [newpop]=selection(pop,fitvalue);                 %复制
%     [newpop]=crossover_multiv(newpop,pc);                       %交叉
%     [newpop]=mutation(newpop,pc);                        %变异
% %     pop=newpop;
%     [bestindividual,bestfit]=best(pop,fitvalue);%求出群体中适应值最大的个体及其适应值
%     y(ii)=max(bestfit);
%     bestindividuals(ii,:)=bestindividual;
%     nn(ii)=ii;
% %     pop5=bestindividual;
% %     x(ii)=decodechrom(pop5,1,chromlength)*10/1023;
%     pop=newpop;
%     count(ii,:)=sum(pop,1);
% end
% % fplot('10*sin(5*x)+7*cos(4*x)',[0 10])
% % plot();
% % hold on
%   
% % hold off
% [M ,I]=max(y);
% Bestindividual=bestindividuals(I,:);
% ind=logical(Bestindividual(1:7));
%     newSpec=A0(:,ind);
%     newtestSpec=B0(:,ind);
%     num=num2str(Bestindividual(8:9));%次级模型编号
%     temp1=bin2dec(num) %将编号转化成十进制数
%     if temp1==0
%         sigma0 = 0.2;
%         kparams0 = [3.5, 6.2];
%         stacking = fitrgp(newSpec,C,'KernelFunction','squaredexponential',...
%             'KernelParameters',kparams0,'Sigma',sigma0);
%         Ypredict5=predict(stacking,newtestSpec);
%         YpredictC=predict(stacking,newSpec);
%         %模型分析
%         co=corr(ProbTest,Ypredict5,'type','Pearson');
%         RMSE=sqrt(sum((ProbTest-Ypredict5).^2)/size(Ypredict5,1));
%         R2 = corr(ProbTest,Ypredict5)^2;
%         RMSEC=sqrt(sum((C-YpredictC).^2)/size(YpredictC,1));
%         R2C = corr(C,YpredictC)^2;
% %         RMSE=num2str(roundn(RMSE,-4));
%         result=[RMSEC;R2C;RMSE;R2];
%         co=num2str(roundn(co,-4));
%         RMSE=num2str(roundn(RMSE,-4));
%         R2=num2str(roundn(R2,-4));
%         RMSEC=num2str(roundn(RMSEC,-4));
%         R2C=num2str(roundn(R2C,-4));
%         msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
%             ['R2C：',R2C,'R2：',R2]},...
%             '次级模型高斯回归');
% %         次级模型运算结果写入文件
%         GPRmetaName = {'RMSEC';'R²C';'RMSEP';'R²P'};
%         % result=[RMSEP;R2P;RMSE;R2];
%         Tabl=table(GPRmetaName,result);
%         writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
% %         writetable(Tabl,'m.xls','Range','E34:F38','WriteVariableNames',true);
%     end
%     if temp1==1
%         Stacking = fitrsvm(newSpec,C,'KernelFunction','gaussian','OptimizeHyperparameters','auto',...
%     'HyperparameterOptimizationOptions',struct('Optimizer','gridsearch','NumGridDivisions',15,...
%     'ShowPlots',false,'MaxObjectiveEvaluations',60));
%     Ypredict5=predict(Stacking,newtestSpec);
%     YpredictC=predict(Stacking,newSpec);
%     %模型分析
%     co=corr(ProbTest,Ypredict5,'type','Pearson');
%     RMSE=sqrt(sum((ProbTest-Ypredict5).^2)/size(Ypredict5,1));
%     R2 = corr(ProbTest,Ypredict5)^2;
%     RMSEC=sqrt(sum((C-YpredictC).^2)/size(YpredictC,1));
%     R2C = corr(C,YpredictC)^2;
%     result=[RMSEC;R2C;RMSE;R2];
%     co=num2str(roundn(co,-4));
%     RMSE=num2str(roundn(RMSE,-4));
%     R2=num2str(roundn(R2,-4));
%     RMSEC=num2str(roundn(RMSEC,-4));
%     R2C=num2str(roundn(R2C,-4));
%     msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
%         ['R2C：',R2C,'R2：',R2]},...
%         '次级模型SVM');
%     %         次级模型运算结果写入文件
%         SVMmetaName = {'RMSEC';'R²C';'RMSEP';'R²P'};
%         % result=[RMSEP;R2P;RMSE;R2];
%         Tabl=table(SVMmetaName,result);
%               writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
% %         writetable(Tabl,'m.xls','Range','E34:F38','WriteVariableNames',true);
%     end
%     if temp1==2
%          Stacking = fitlm(newSpec,C);
%     Ypredict5=predict(Stacking,newtestSpec);
%     YpredoctA=predict(Stacking,newSpec);
%     %模型分析
%     co=corr(ProbTest,Ypredict5,'type','Pearson');
%     RMSE=sqrt(sum((ProbTest-Ypredict5).^2)/size(Ypredict5,1));
%     R2 = corr(ProbTest,Ypredict5)^2;
% %     coP=corr(prob,YpredoctA,'type','Pearson');
%     RMSEP=sqrt(sum((C-YpredoctA).^2)/size(YpredoctA,1));
%     R2P = corr(C,YpredoctA)^2;
%     result=[RMSEP;R2P;RMSE;R2];
%     co=num2str(roundn(co,-4));
%     RMSE=num2str(roundn(RMSE,-4));
%     R2=num2str(roundn(R2,-4));
%     RMSEP=num2str(roundn(RMSEP,-4));
%     R2P=num2str(roundn(R2P,-4));
%     msgbox({'您运算的结果为:',['RMSEC：',RMSEP,'RMSE：',RMSE],...
%         ['R2C：',R2P,'R2：',R2]},...
%         '次级模型PLS');
%      %         次级模型运算结果写入文件
%         SVMmetaName = {'RMSEC';'R²C';'RMSEP';'R²P'};
%         % result=[RMSEP;R2P;RMSE;R2];
%         Tabl=table(SVMmetaName,result);
%                 writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
% %         writetable(Tabl,'m.xls','Range','E34:F38','WriteVariableNames',true);
%     end
%     if temp1==3
%     isCategorical = zeros(size(newSpec,2),1);
%     treeNum=100;
%     nodeNum=15;
%     Stacking = TreeBagger(treeNum,newSpec,C,'Method','R','OOBPredictorImportance','On',...
%     'CategoricalPredictors',find(isCategorical == 1),...
%     'MinLeafSize',nodeNum);
%     Ypredict5=predict(Stacking,newtestSpec);
%     YpredictC=predict(Stacking,newSpec);
%     %模型分析
%     co=corr(ProbTest,Ypredict5,'type','Pearson');
%     RMSE=sqrt(sum((ProbTest-Ypredict5).^2)/size(Ypredict5,1));
%     R2 = corr(ProbTest,Ypredict5)^2;
%     RMSEC=sqrt(sum((C-YpredictC).^2)/size(YpredictC,1));
%     R2C = corr(C,YpredictC)^2;
%     result=[RMSEC;R2C;RMSE;R2];
%         RMSE=num2str(roundn(RMSE,-4));
%     R2=num2str(roundn(R2,-4));
%     RMSEC=num2str(roundn(RMSEC,-4));
%     R2C=num2str(roundn(R2C,-4));
%     msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
%         ['R2C：',R2C,'R2：',R2]},...
%         '次级随机森林');
%      %         次级模型运算结果写入文件
%        RFmetaName = {'RMSEC';'R²C';'RMSEP';'R²P'};
%         % result=[RMSEP;R2P;RMSE;R2];
%         Tabl=table(RFmetaName,result);
%                writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
% %         writetable(Tabl,'m.xls','Range','E34:F38','WriteVariableNames',true);
%     end
% 
%     %迭代轮次
% iteration=50;
% %Name:genmain05.m
% % clear
% % clf
% % popsize=9;                                       %群体大小
% % chromlength=7;                                   %字符串长度（个体长度）
% popsize=30;                                       %群体大小
% chromlength=9;                                   %字符串长度（个体长度）
% count = zeros(iteration,chromlength);
% pc=0.6;                                           %交叉概率
% pm=0.001;                                         %变异概率
% pop=initpop(popsize,chromlength);                 %随机产生初始群体
% bestindividuals=zeros(iteration,chromlength);     %每轮最好的特征选择
% for ii=1:iteration                                    %20为迭代次数
% %     objvalue=R2;
%     [objvalue]=calobjvalue3(pop,A0,C,B0,ProbTest);                      %计算目标函数
%     fitvalue=objvalue;                   %计算群体中每个个体的适应度
%     [newpop]=selection(pop,fitvalue);                 %复制
%     [newpop]=crossover_multiv(newpop,pc);                       %交叉
%     [newpop]=mutation(newpop,pc);                        %变异
% %     pop=newpop;
%     [bestindividual,bestfit]=best(pop,fitvalue);%求出群体中适应值最大的个体及其适应值
%     y(ii)=max(bestfit);
%     bestindividuals(ii,:)=bestindividual;
%     nn(ii)=ii;
% %     pop5=bestindividual;
% %     x(ii)=decodechrom(pop5,1,chromlength)*10/1023;
%     pop=newpop;
%     count(ii,:)=sum(pop,1);
% end
% % fplot('10*sin(5*x)+7*cos(4*x)',[0 10])
% % plot();
% % hold on
%   
% % hold off
% [M ,I]=max(y);
% Bestindividual=bestindividuals(I,:);
% ind=logical(Bestindividual(1:7));
%     newSpec=A0(:,ind);
%     newtestSpec=B0(:,ind);
%     num=num2str(Bestindividual(8:9));%次级模型编号
%     temp1=bin2dec(num) %将编号转化成十进制数
%     if temp1==0
%         sigma0 = 0.2;
%         kparams0 = [3.5, 6.2];
%         stacking = fitrgp(newSpec,C,'KernelFunction','squaredexponential',...
%             'KernelParameters',kparams0,'Sigma',sigma0);
%         Ypredict5=predict(stacking,newtestSpec);
%         YpredictC=predict(stacking,newSpec);
%         %模型分析
%         co=corr(ProbTest,Ypredict5,'type','Pearson');
%         RMSE=sqrt(sum((ProbTest-Ypredict5).^2)/size(Ypredict5,1));
%         R2 = corr(ProbTest,Ypredict5)^2;
%         RMSEC=sqrt(sum((C-YpredictC).^2)/size(YpredictC,1));
%         R2C = corr(C,YpredictC)^2;
% %         RMSE=num2str(roundn(RMSE,-4));
%         result=[RMSEC;R2C;RMSE;R2];
%         co=num2str(roundn(co,-4));
%         RMSE=num2str(roundn(RMSE,-4));
%         R2=num2str(roundn(R2,-4));
%         RMSEC=num2str(roundn(RMSEC,-4));
%         R2C=num2str(roundn(R2C,-4));
%         msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
%             ['R2C：',R2C,'R2：',R2]},...
%             '次级模型高斯回归');
% %         次级模型运算结果写入文件
%         GPRmetaName = {'RMSEC';'R²C';'RMSEP';'R²P'};
%         % result=[RMSEP;R2P;RMSE;R2];
%         Tabl=table(GPRmetaName,result);
%         writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
%     end
%     if temp1==1
%         Stacking = fitrsvm(newSpec,C,'KernelFunction','gaussian','OptimizeHyperparameters','auto',...
%     'HyperparameterOptimizationOptions',struct('Optimizer','gridsearch','NumGridDivisions',15,...
%     'ShowPlots',false,'MaxObjectiveEvaluations',60));
%     Ypredict5=predict(Stacking,newtestSpec);
%     YpredictC=predict(Stacking,newSpec);
%     %模型分析
%     co=corr(ProbTest,Ypredict5,'type','Pearson');
%     RMSE=sqrt(sum((ProbTest-Ypredict5).^2)/size(Ypredict5,1));
%     R2 = corr(ProbTest,Ypredict5)^2;
%     RMSEC=sqrt(sum((C-YpredictC).^2)/size(YpredictC,1));
%     R2C = corr(C,YpredictC)^2;
%     result=[RMSEC;R2C;RMSE;R2];
%     co=num2str(roundn(co,-4));
%     RMSE=num2str(roundn(RMSE,-4));
%     R2=num2str(roundn(R2,-4));
%     RMSEC=num2str(roundn(RMSEC,-4));
%     R2C=num2str(roundn(R2C,-4));
%     msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
%         ['R2C：',R2C,'R2：',R2]},...
%         '次级模型SVM');
%     %         次级模型运算结果写入文件
%         SVMmetaName = {'RMSEC';'R²C';'RMSEP';'R²P'};
%         % result=[RMSEP;R2P;RMSE;R2];
%         Tabl=table(SVMmetaName,result);
%         writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
%     end
%     if temp1==2
%          Stacking = fitlm(newSpec,C);
%     Ypredict5=predict(Stacking,newtestSpec);
%     YpredoctA=predict(Stacking,newSpec);
%     %模型分析
%     co=corr(ProbTest,Ypredict5,'type','Pearson');
%     RMSE=sqrt(sum((ProbTest-Ypredict5).^2)/size(Ypredict5,1));
%     R2 = corr(ProbTest,Ypredict5)^2;
% %     coP=corr(prob,YpredoctA,'type','Pearson');
%     RMSEP=sqrt(sum((C-YpredoctA).^2)/size(YpredoctA,1));
%     R2P = corr(C,YpredoctA)^2;
%     result=[RMSEP;R2P;RMSE;R2];
%     co=num2str(roundn(co,-4));
%     RMSE=num2str(roundn(RMSE,-4));
%     R2=num2str(roundn(R2,-4));
%     RMSEP=num2str(roundn(RMSEP,-4));
%     R2P=num2str(roundn(R2P,-4));
%     msgbox({'您运算的结果为:',['RMSEC：',RMSEP,'RMSE：',RMSE],...
%         ['R2C：',R2P,'R2：',R2]},...
%         '次级模型PLS');
%      %         次级模型运算结果写入文件
%         SVMmetaName = {'RMSEC';'R²C';'RMSEP';'R²P'};
%         % result=[RMSEP;R2P;RMSE;R2];
%         Tabl=table(SVMmetaName,result);
%         writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
%     end
%     if temp1==3
%     isCategorical = zeros(size(newSpec,2),1);
%     treeNum=100;
%     nodeNum=15;
%     Stacking = TreeBagger(treeNum,newSpec,C,'Method','R','OOBPredictorImportance','On',...
%     'CategoricalPredictors',find(isCategorical == 1),...
%     'MinLeafSize',nodeNum);
%     Ypredict5=predict(Stacking,newtestSpec);
%     YpredictC=predict(Stacking,newSpec);
%     %模型分析
%     co=corr(ProbTest,Ypredict5,'type','Pearson');
%     RMSE=sqrt(sum((ProbTest-Ypredict5).^2)/size(Ypredict5,1));
%     R2 = corr(ProbTest,Ypredict5)^2;
%     RMSEC=sqrt(sum((C-YpredictC).^2)/size(YpredictC,1));
%     R2C = corr(C,YpredictC)^2;
%     result=[RMSEC;R2C;RMSE;R2];
%         RMSE=num2str(roundn(RMSE,-4));
%     R2=num2str(roundn(R2,-4));
%     RMSEC=num2str(roundn(RMSEC,-4));
%     R2C=num2str(roundn(R2C,-4));
%     msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
%         ['R2C：',R2C,'R2：',R2]},...
%         '次级随机森林');
%      %         次级模型运算结果写入文件
%        RFmetaName = {'RMSEC';'R²C';'RMSEP';'R²P'};
%         % result=[RMSEP;R2P;RMSE;R2];
%         Tabl=table(RFmetaName,result);
%         writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
%     end
%     
%     
%     %选三个最好的单学习器次级随机森林
% isCategorical = zeros(size(X,2),1);
% treeNum=100;
% nodeNum=15;
% Stacking = TreeBagger(treeNum,A1,C,'Method','R','OOBPredictorImportance','On',...
%     'CategoricalPredictors',find(isCategorical == 1),...
%     'MinLeafSize',nodeNum);
% Ypredict5=predict(Stacking,B1);
% YpredictC=predict(Stacking,A1);
% %模型分析
% co=corr(ProbTest,Ypredict5,'type','Pearson');
% RMSE=sqrt(sum((ProbTest-Ypredict5).^2)/size(Ypredict5,1));
% R2 = corr(ProbTest,Ypredict5)^2;
% RMSEC=sqrt(sum((C-YpredictC).^2)/size(YpredictC,1));
% R2C = corr(C,YpredictC)^2;
% result=[RMSEC;R2C;RMSE;R2];
% co=num2str(roundn(co,-4));
% RMSE=num2str(roundn(RMSE,-4));
% R2=num2str(roundn(R2,-4));
% RMSEC=num2str(roundn(RMSEC,-4));
% R2C=num2str(roundn(R2C,-4));
% msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
%     ['R2C：',R2C,'R2：',R2]},...
%     '次级RF1');
%  % 将运算结果写入文件
% FNNname = {'RMSEC';'R²C';'RMSEP';'R²P'};
% % result=[RMSEP;R2P;RMSE;R2];
% Tabl=table(FNNname,result);
% writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
% d11=Ypredict5;
% d21=YpredictC;
%     
% 
% %选三个效果好的单模型次级模型SVM
% Stacking = fitrsvm(A1,C,'KernelFunction','gaussian','OptimizeHyperparameters','auto',...
%     'HyperparameterOptimizationOptions',struct('Optimizer','gridsearch','NumGridDivisions',15,...
%     'ShowPlots',false,'MaxObjectiveEvaluations',60));
% Ypredict5=predict(Stacking,B1);
% YpredictC=predict(Stacking,A1);
% %模型分析
% co=corr(ProbTest,Ypredict5,'type','Pearson');
% RMSE=sqrt(sum((ProbTest-Ypredict5).^2)/size(Ypredict5,1));
% R2 = corr(ProbTest,Ypredict5)^2;
% RMSEC=sqrt(sum((C-YpredictC).^2)/size(YpredictC,1));
% R2C = corr(C,YpredictC)^2;
% result=[RMSEC;R2C;RMSE;R2];
% co=num2str(roundn(co,-4));
% RMSE=num2str(roundn(RMSE,-4));
% R2=num2str(roundn(R2,-4));
% RMSEC=num2str(roundn(RMSEC,-4));
% R2C=num2str(roundn(R2C,-4));
% msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
%     ['R2C：',R2C,'R2：',R2]},...
%     '次级模型SVM');
%  % 将运算结果写入文件
% FNNname = {'RMSEC';'R²C';'RMSEP';'R²P'};
% % result=[RMSEP;R2P;RMSE;R2];
% Tabl=table(FNNname,result);
% writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
% % d1=Ypredict5;
% % d2=YpredictC;
% d11=d11+Ypredict5;
% d21=d21+YpredictC;
% 
% %选三个效果好的单模型次级模型PLS
% Stacking = fitlm(A1,C);
% Ypredict5=predict(Stacking,B1);
% YpredoctA=predict(Stacking,A1);
% %模型分析
% co=corr(ProbTest,Ypredict5,'type','Pearson');
% RMSE=sqrt(sum((ProbTest-Ypredict5).^2)/size(Ypredict5,1));
% R2 = corr(ProbTest,Ypredict5)^2;
% coP=corr(C,YpredoctA,'type','Pearson');
% RMSEC=sqrt(sum((C-YpredoctA).^2)/size(YpredoctA,1));
% R2C = corr(C,YpredoctA)^2;
% result=[RMSEC;R2C;RMSE;R2];
% co=num2str(roundn(co,-4));
% RMSE=num2str(roundn(RMSE,-4));
% R2=num2str(roundn(R2,-4));
% RMSEC=num2str(roundn(RMSEC,-4));
% R2C=num2str(roundn(R2C,-4));
% msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
%     ['R2C：',R2C,'R2：',R2]},...
%     '次级模型PLS');
% % 将运算结果写入文件
% FNNname = {'RMSEC';'R²C';'RMSEP';'R²P'};
% % result=[RMSEP;R2P;RMSE;R2];
% Tabl=table(FNNname,result);
% writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
% d11=d11+Ypredict5;
% d21=d21+YpredoctA;
% 
% %选三个效果好的单模型次级模型高斯回归
% sigma0 = 0.2;
% kparams0 = [3.5, 6.2];
% stacking = fitrgp(A1,C,'KernelFunction','squaredexponential',...
%         'KernelParameters',kparams0,'Sigma',sigma0);
% Ypredict5=predict(stacking,B1);
% YpredictC=predict(stacking,A1);
% %模型分析
% co=corr(ProbTest,Ypredict5,'type','Pearson');
% RMSE=sqrt(sum((ProbTest-Ypredict5).^2)/size(Ypredict5,1));
% R2 = corr(ProbTest,Ypredict5)^2;
% RMSEC=sqrt(sum((C-YpredictC).^2)/size(YpredictC,1));
% R2C = corr(C,YpredictC)^2;
% result=[RMSEC;R2C;RMSE;R2];
% co=num2str(roundn(co,-4));
% RMSE=num2str(roundn(RMSE,-4));
% R2=num2str(roundn(R2,-4));
% RMSEC=num2str(roundn(RMSEC,-4));
% R2C=num2str(roundn(R2C,-4));
% msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
%     ['R2C：',R2C,'R2：',R2]},...
%     '次级模型高斯回归');
% % 将运算结果写入文件
% FNNname = {'RMSEC';'R²C';'RMSEP';'R²P'};
% % result=[RMSEP;R2P;RMSE;R2];
% Tabl=table(FNNname,result);
% writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
% d11=d11+Ypredict5;
% d21=d21+YpredictC;
% 
% A=A0(:,logical(Bestindividual));
% B=B0(:,logical(Bestindividual));
% figure();
% plot(nn,y);
% figure();
% plot(1:7,sum(count,1));
% hold on;
% scatter(1:7,Bestindividual.*sum(count,1));
% 
% % A=A0(:,logical(Bestindividual));
% % B=B0(:,logical(Bestindividual));
% %次级随机森林
% isCategorical = zeros(size(X,2),1);
% treeNum=100;
% nodeNum=15;
% Stacking = TreeBagger(treeNum,A,C,'Method','R','OOBPredictorImportance','On',...
%     'CategoricalPredictors',find(isCategorical == 1),...
%     'MinLeafSize',nodeNum);
% Ypredict5=predict(Stacking,B);
% YpredictC=predict(Stacking,A);
% %模型分析
% co=corr(ProbTest,Ypredict5,'type','Pearson');
% RMSE=sqrt(sum((ProbTest-Ypredict5).^2)/size(Ypredict5,1));
% R2 = corr(ProbTest,Ypredict5)^2;
% RMSEC=sqrt(sum((C-YpredictC).^2)/size(YpredictC,1));
% R2C = corr(C,YpredictC)^2;
% co=num2str(roundn(co,-4));
% RMSE=num2str(roundn(RMSE,-4));
% R2=num2str(roundn(R2,-4));
% RMSEC=num2str(roundn(RMSEC,-4));
% R2C=num2str(roundn(R2C,-4));
% msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
%     ['R2C：',R2C,'R2：',R2]},...
%     '次级随机森林');
% d10=Ypredict5;
% d20=YpredictC;
% 
% 
% 
% %Name:genmain05.m
% % clear
% % clf
% popsize=10;                                       %群体大小
% % chromlength=7;                                   %字符串长度（个体长度）
% count = zeros(iteration,chromlength);
% pc=0.6;                                           %交叉概率
% pm=0.001;                                         %变异概率
% pop=initpop(popsize,chromlength);                 %随机产生初始群体
% bestindividuals=zeros(iteration,chromlength);     %每轮最好的特征选择
% for ii=1:iteration                                     %20为迭代次数
%     [objvalue]=calobjvalueSVM(pop,A0,C,B0,ProbTest);                      %计算目标函数
%     fitvalue=objvalue;                   %计算群体中每个个体的适应度
%     [newpop]=selection(pop,fitvalue);                 %复制
%     [newpop]=crossover_multiv(newpop,pc);                       %交叉
%     [newpop]=mutation(newpop,pc);                        %变异
% %     pop=newpop;
%     [bestindividual,bestfit]=best(pop,fitvalue);%求出群体中适应值最大的个体及其适应值
%     y(ii)=max(bestfit);
%     n(ii)=ii;
%     bestindividuals(ii,:)=bestindividual;
% %     pop5=bestindividual;
% %     x(ii)=decodechrom(pop5,1,chromlength)*10/1023;
%     pop=newpop;
%     count(ii,:)=sum(pop,1);
% end
% % fplot('10*sin(5*x)+7*cos(4*x)',[0 10])
% % plot();
% % hold on
%  [M ,I]=max(y);
% Bestindividual=bestindividuals(I,:);
%  figure();
% plot(nn,y);
% figure();
% plot(1:7,sum(count,1));
% hold on;
% scatter(1:7,Bestindividual.*sum(count,1));
% % hold off
% A=A0(:,logical(Bestindividual));
% B=B0(:,logical(Bestindividual));
% 
% %次级模型SVM
% Stacking = fitrsvm(A,C,'KernelFunction','gaussian','OptimizeHyperparameters','auto',...
%     'HyperparameterOptimizationOptions',struct('Optimizer','gridsearch','NumGridDivisions',15,...
%     'ShowPlots',false,'MaxObjectiveEvaluations',60));
% Ypredict5=predict(Stacking,B);
% YpredictC=predict(Stacking,A);
% %模型分析
% co=corr(ProbTest,Ypredict5,'type','Pearson');
% RMSE=sqrt(sum((ProbTest-Ypredict5).^2)/size(Ypredict5,1));
% R2 = corr(ProbTest,Ypredict5)^2;
% RMSEC=sqrt(sum((C-YpredictC).^2)/size(YpredictC,1));
% R2C = corr(C,YpredictC)^2;
% co=num2str(roundn(co,-4));
% RMSE=num2str(roundn(RMSE,-4));
% R2=num2str(roundn(R2,-4));
% RMSEC=num2str(roundn(RMSEC,-4));
% R2C=num2str(roundn(R2C,-4));
% msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
%     ['R2C：',R2C,'R2：',R2]},...
%     '次级模型SVM');
% % d1=Ypredict5;
% % d2=YpredictC;
% d10=d10+Ypredict5;
% d20=d20+YpredictC;
% 
% %选三个效果好的单模型次级模型SVM
% Stacking = fitrsvm(A1,C,'KernelFunction','gaussian','OptimizeHyperparameters','auto',...
%     'HyperparameterOptimizationOptions',struct('Optimizer','gridsearch','NumGridDivisions',15,...
%     'ShowPlots',false,'MaxObjectiveEvaluations',60));
% Ypredict5=predict(Stacking,B1);
% YpredictC=predict(Stacking,A1);
% %模型分析
% co=corr(ProbTest,Ypredict5,'type','Pearson');
% RMSE=sqrt(sum((ProbTest-Ypredict5).^2)/size(Ypredict5,1));
% R2 = corr(ProbTest,Ypredict5)^2;
% RMSEC=sqrt(sum((C-YpredictC).^2)/size(YpredictC,1));
% R2C = corr(C,YpredictC)^2;
% co=num2str(roundn(co,-4));
% RMSE=num2str(roundn(RMSE,-4));
% R2=num2str(roundn(R2,-4));
% RMSEC=num2str(roundn(RMSEC,-4));
% R2C=num2str(roundn(R2C,-4));
% msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
%     ['R2C：',R2C,'R2：',R2]},...
%     '次级模型SVM');
% % d1=Ypredict5;
% % d2=YpredictC;
% d11=d11+Ypredict5;
% d21=d21+YpredictC;
% 
% %Name:genmain05.m
% % clear
% % clf
% popsize=10;                                       %群体大小
% % chromlength=5;                                   %字符串长度（个体长度）
% count = zeros(iteration,chromlength);
% pc=0.6;                                           %交叉概率
% pm=0.001;                                         %变异概率
% pop=initpop(popsize,chromlength);                 %随机产生初始群体
% bestindividuals=zeros(iteration,chromlength);     %每轮最好的特征选择
% for ii=1:iteration                                     %20为迭代次数
%     [objvalue]=calobjvaluePLS(pop,A0,C,B0,ProbTest);                      %计算目标函数
%     fitvalue=objvalue;                   %计算群体中每个个体的适应度
%     [newpop]=selection(pop,fitvalue);                 %复制
%     [newpop]=crossover_multiv(newpop,pc);                       %交叉
%     [newpop]=mutation(newpop,pc);                        %变异
% %     pop=newpop;
%     [bestindividual,bestfit]=best(pop,fitvalue);%求出群体中适应值最大的个体及其适应值
%     y(ii)=max(bestfit);
%     n(ii)=ii;
%      bestindividuals(ii,:)=bestindividual;
% %     pop5=bestindividual;
% %     x(ii)=decodechrom(pop5,1,chromlength)*10/1023;
%     pop=newpop;
%     count(ii,:)=sum(pop,1);
% end
% % fplot('10*sin(5*x)+7*cos(4*x)',[0 10])
% % plot();
% % hold on
%  [M ,I]=max(y);
% Bestindividual=bestindividuals(I,:);
% figure();
% plot(nn,y);
% figure();
% plot(1:7,sum(count,1));
% hold on;
% scatter(1:7,Bestindividual.*sum(count,1));
% % hold off
% A=A0(:,logical(Bestindividual));
% B=B0(:,logical(Bestindividual));
% 
% %次级模型PLS
% Stacking = fitlm(A,C);
% Ypredict5=predict(Stacking,B);
% YpredoctA=predict(Stacking,A);
% %模型分析
% co=corr(ProbTest,Ypredict5,'type','Pearson');
% RMSE=sqrt(sum((ProbTest-Ypredict5).^2)/size(Ypredict5,1));
% R2 = corr(ProbTest,Ypredict5)^2;
% coP=corr(C,YpredoctA,'type','Pearson');
% RMSEP=sqrt(sum((C-YpredoctA).^2)/size(YpredoctA,1));
% R2P = corr(C,YpredoctA)^2;
% co=num2str(roundn(co,-4));
% RMSE=num2str(roundn(RMSE,-4));
% R2=num2str(roundn(R2,-4));
% RMSEP=num2str(roundn(RMSEP,-4));
% R2P=num2str(roundn(R2P,-4));
% msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
%     ['R2C：',R2C,'R2：',R2]},...
%     '次级模型PLS');
% d10=d10+Ypredict5;
% d20=d20+YpredoctA;
% 
% %选三个效果好的单模型次级模型PLS
% Stacking = fitlm(A1,C);
% Ypredict5=predict(Stacking,B1);
% YpredoctA=predict(Stacking,A1);
% %模型分析
% co=corr(ProbTest,Ypredict5,'type','Pearson');
% RMSE=sqrt(sum((ProbTest-Ypredict5).^2)/size(Ypredict5,1));
% R2 = corr(ProbTest,Ypredict5)^2;
% coP=corr(C,YpredoctA,'type','Pearson');
% RMSEP=sqrt(sum((C-YpredoctA).^2)/size(YpredoctA,1));
% R2P = corr(C,YpredoctA)^2;
% co=num2str(roundn(co,-4));
% RMSE=num2str(roundn(RMSE,-4));
% R2=num2str(roundn(R2,-4));
% RMSEP=num2str(roundn(RMSEP,-4));
% R2P=num2str(roundn(R2P,-4));
% msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
%     ['R2C：',R2C,'R2：',R2]},...
%     '次级模型PLS');
% d11=d11+Ypredict5;
% d21=d21+YpredoctA;
% 
% %Name:genmain05.m
% % clear
% % clf
% popsize=10;                                       %群体大小
% % chromlength=5;                                   %字符串长度（个体长度）
% count = zeros(iteration,chromlength);
% pc=0.6;                                           %交叉概率
% pm=0.001;                                         %变异概率
% pop=initpop(popsize,chromlength);                 %随机产生初始群体
% bestindividuals=zeros(iteration,chromlength);     %每轮最好的特征选择
% for ii=1:iteration                                      %20为迭代次数
%     [objvalue]=calobjvalueGPR(pop,A0,C,B0,ProbTest);                      %计算目标函数
%     fitvalue=objvalue;                   %计算群体中每个个体的适应度
%     [newpop]=selection(pop,fitvalue);                 %复制
%     [newpop]=crossover_multiv(newpop,pc);                       %交叉
%     [newpop]=mutation(newpop,pc);                        %变异
% %     pop=newpop;
%     [bestindividual,bestfit]=best(pop,fitvalue);%求出群体中适应值最大的个体及其适应值
%     y(ii)=max(bestfit);
%     n(ii)=ii;
%       bestindividuals(ii,:)=bestindividual;
% %     pop5=bestindividual;
% %     x(ii)=decodechrom(pop5,1,chromlength)*10/1023;
%     pop=newpop;
%     count(ii,:)=sum(pop,1);
% end
% % fplot('10*sin(5*x)+7*cos(4*x)',[0 10])
% % plot();
% % hold on
%  [M ,I]=max(y);
% Bestindividual=bestindividuals(I,:);
% figure();
% plot(nn,y);
% figure();
% plot(1:7,sum(count,1));
% hold on;
% scatter(1:7,Bestindividual.*sum(count,1));
% % hold off
% A=A0(:,logical(Bestindividual));
% B=B0(:,logical(Bestindividual));
% 
% %次级模型高斯回归
% sigma0 = 0.2;
% kparams0 = [3.5, 6.2];
% stacking = fitrgp(A,C,'KernelFunction','squaredexponential',...
%         'KernelParameters',kparams0,'Sigma',sigma0);
% Ypredict5=predict(stacking,B);
% YpredictC=predict(stacking,A);
% %模型分析
% co=corr(ProbTest,Ypredict5,'type','Pearson');
% RMSE=sqrt(sum((ProbTest-Ypredict5).^2)/size(Ypredict5,1));
% R2 = corr(ProbTest,Ypredict5)^2;
% RMSEC=sqrt(sum((C-YpredictC).^2)/size(YpredictC,1));
% R2C = corr(C,YpredictC)^2;
% co=num2str(roundn(co,-4));
% RMSE=num2str(roundn(RMSE,-4));
% R2=num2str(roundn(R2,-4));
% RMSEC=num2str(roundn(RMSEC,-4));
% R2C=num2str(roundn(R2C,-4));
% msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
%     ['R2C：',R2C,'R2：',R2]},...
%     '次级模型高斯回归');
% d10=d10+Ypredict5;
% d20=d20+YpredictC;
% 
% %选三个效果好的单模型次级模型高斯回归
% sigma0 = 0.2;
% kparams0 = [3.5, 6.2];
% stacking = fitrgp(A1,C,'KernelFunction','squaredexponential',...
%         'KernelParameters',kparams0,'Sigma',sigma0);
% Ypredict5=predict(stacking,B1);
% YpredictC=predict(stacking,A1);
% %模型分析
% co=corr(ProbTest,Ypredict5,'type','Pearson');
% RMSE=sqrt(sum((ProbTest-Ypredict5).^2)/size(Ypredict5,1));
% R2 = corr(ProbTest,Ypredict5)^2;
% RMSEC=sqrt(sum((C-YpredictC).^2)/size(YpredictC,1));
% R2C = corr(C,YpredictC)^2;
% co=num2str(roundn(co,-4));
% RMSE=num2str(roundn(RMSE,-4));
% R2=num2str(roundn(R2,-4));
% RMSEC=num2str(roundn(RMSEC,-4));
% R2C=num2str(roundn(R2C,-4));
% msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
%     ['R2C：',R2C,'R2：',R2]},...
%     '次级模型高斯回归');
% d11=d11+Ypredict5;
% d21=d21+YpredictC;
% 
% d10=d10./4; 
% d20=d20./4;
% RMSE=sqrt(sum((ProbTest-d10).^2)/size(d10,1));
% R2 = corr(ProbTest,d10)^2;
% RMSEC=sqrt(sum((C-d20).^2)/size(d20,1));
% R2C = corr(C,d20)^2;
% % co=num2str(roundn(co,-4));
% RMSE=num2str(roundn(RMSE,-4));
% R2=num2str(roundn(R2,-4));
% RMSEC=num2str(roundn(RMSEC,-4));
% R2C=num2str(roundn(R2C,-4));
% msgbox({['您的第',i,'运算的结果为:'],['RMSEC：',RMSEC,'RMSE：',RMSE],...
%     ['R2C：',R2C,'R2：',R2]},...
%     '简单平均法');
% 
% %选三个效果好的基模型
% d11=d11./4;
% d21=d21./4;
% RMSE=sqrt(sum((ProbTest-d11).^2)/size(d11,1));
% R2 = corr(ProbTest,d11)^2;
% RMSEC=sqrt(sum((C-d21).^2)/size(d21,1));
% R2C = corr(C,d21)^2;
% % co=num2str(roundn(co,-4));
% RMSE=num2str(roundn(RMSE,-4));
% R2=num2str(roundn(R2,-4));
% RMSEC=num2str(roundn(RMSEC,-4));
% R2C=num2str(roundn(R2C,-4));
% msgbox({['您的第',i,'运算的结果为:'],['RMSEC：',RMSEC,'RMSE：',RMSE],...
%     ['R2C：',R2C,'R2：',R2]},...
%     '简单平均法');
% 
% % if (str2num(R2)>0.7 && str2num(R2C)>0.7)
% %     choos=1;
% % else
% %     i=i-1;
% % end
% % if(choos==1)
% %     if(i==1)
% %         D1=d10;
% %         D2=d20;
% %     else
% %         D1= D1+d10;
% %         D2=D2+d20;
% %     end 
% %     choos=0;
% % end
% if(i==1)
%     D1=d10;
%     D2=d20;
% else
%     D1= D1+d10;
%     D2=D2+d20;
% end 
% 
% if(i==1)
%     D11=d11;
%     D21=d21;
% else
%     D11= D11+d11;
%     D21=D21+d21;
% end
% % end
% D1=D1./i;
% D2=D2./i;
% RMSE=sqrt(sum((ProbTest-D1).^2)/size(D1,1));
% R2 = corr(ProbTest,D1)^2;
% RMSEC=sqrt(sum((C-D2).^2)/size(D2,1));
% R2C = corr(C,D2)^2;
% % co=num2str(roundn(co,-4));
% RMSE=num2str(roundn(RMSE,-4));
% R2=num2str(roundn(R2,-4));
% RMSEC=num2str(roundn(RMSEC,-4));
% R2C=num2str(roundn(R2C,-4));
% msgbox({['您的最终运算的结果为:'],['RMSEC：',RMSEC,'RMSE：',RMSE],...
%     ['R2C：',R2C,'R2：',R2]},...
%     '最终结果');
% 
% D11=D11./i;
% D21=D21./i;
% RMSE=sqrt(sum((ProbTest-D11).^2)/size(D11,1));
% R2 = corr(ProbTest,D11)^2;
% RMSEC=sqrt(sum((C-D21).^2)/size(D21,1));
% R2C = corr(C,D21)^2;
% % co=num2str(roundn(co,-4));
% RMSE=num2str(roundn(RMSE,-4));
% R2=num2str(roundn(R2,-4));
% RMSEC=num2str(roundn(RMSEC,-4));
% R2C=num2str(roundn(R2C,-4));
% msgbox({['您的最终运算的结果为:'],['RMSEC：',RMSEC,'RMSE：',RMSE],...
%     ['R2C：',R2C,'R2：',R2]},...
%     '最终结果');
% 
%  