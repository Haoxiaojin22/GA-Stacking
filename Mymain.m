
D1=[];
D2=[];
D11=[];
D21=[];
choos=0;
% % %read data
% data=csvread('D:\HXJ\研究生\数据\玉米发芽\玉米发芽率数据\玉米发芽率数据\laohuaYM_noAvg.csv');
% data=data(2:end,:);
% % data=[diesel_spec.data(:,:),diesel_prop.data(:,1)];
% % %Remove outliers ,NULL and INF
% RemoveOutliersData=RemoveOutliers(data);
% % data=[rockNIR,TOC];
% % plot(data(:,1:end-1)');

% % Randomly divide the data set in proportion

csvwrite('diesel_spec.csv',diesel_spec.data(:,:))
csvwrite('diesel_prop.csv',diesel_prop.data(:,:))

ABA1=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\70\ABA.mat');
ABA2=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\0510\ABA.mat');
ABA80=[ABA1.ABA70 ABA2.ABA];
ABB1=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\70\ABB.mat');
ABB2=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\0510\ABB.mat');
ABB80=[ABB1.ABB70 ABB2.ABB];
ABresult1=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\70\ABresult.mat');
ABresult2=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\0510\ABresult.mat');
ABresult80=[ABresult1.ABresult70 ABresult2.ABresult];
FNNA1=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\70\FNNA.mat');
FNNA2=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\0510\FNNA.mat');
FNNA80=[FNNA1.FNNA70 FNNA2.FNNA];
FNNB1=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\70\FNNB.mat');
FNNB2=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\0510\FNNB.mat');
FNNB80=[FNNB1.FNNB70 FNNB2.FNNB];
FNNresult1=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\70\FNNresult.mat');
FNNresult2=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\0510\FNNresult.mat');
FNNresult80=[FNNresult1.FNNresult70 FNNresult2.FNNresult];
GPRA1=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\70\GPRA.mat');
GPRA2=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\0510\GPRA.mat');
GPRA80=[GPRA1.GPRA70 GPRA2.GSA];
GPRB1=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\70\GPRB.mat');
GPRB2=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\0510\GPRB.mat');
GPRB80=[GPRB1.GPRB70 GPRB2.GSB];
GPRresult1=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\70\GPRresult.mat');
GPRresult2=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\0510\GPRresult.mat');
GPRresult80=[GPRresult1.GPRresult70 GPRresult2.GPRresult];
KNNA1=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\70\KNNA.mat');
KNNA2=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\0510\KNNA.mat');
KNNA80=[KNNA1.KNNA70 KNNA2.KNNA];
KNNB1=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\70\KNNB.mat');
KNNB2=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\0510\KNNB.mat');
KNNB80=[KNNB1.KNNB70 KNNB2.KNNB];
KNNresult1=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\70\KNNresult.mat');
KNNresult2=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\0510\KNNresult.mat');
KNNresult80=[KNNresult1.KNNresult70 KNNresult2.KNNresult];
PLSA1=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\70\PLSA.mat');
PLSA2=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\0510\PLSA.mat');
PLSA80=[PLSA1.PLSA70 PLSA2.PLSA];
PLSB1=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\70\PLSB.mat');
PLSB2=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\0510\PLSB.mat');
PLSB80=[PLSB1.PLSB70 PLSB2.PLSB];
PLSresult1=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\70\PLSresult.mat');
PLSresult2=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\0510\PLSresult.mat');
PLSresult80=[PLSresult1.PLSresult70 PLSresult2.PLSresult];
RFA1=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\70\RFA.mat');
RFA2=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\0510\RFA.mat');
RFA80=[RFA1.RFA70 RFA2.RFA];
RFB1=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\70\RFB.mat');
RFB2=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\0510\RFB.mat');
RFB80=[RFB1.RFB70 RFB2.RFB];
RFresult1=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\70\RFresult.mat');
RFresult2=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\0510\RFresult.mat');
RFresult80=[RFresult1.RFresult70 RFresult2.RFresult];
SVMA1=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\70\SVMA.mat');
SVMA2=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\0510\SVMA.mat');
SVMA80=[SVMA1.SVMA70 SVMA2.SVMA];
SVMB1=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\70\SVMB.mat');
SVMB2=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\0510\SVMB.mat');
SVMB80=[SVMB1.SVMB70 SVMB2.SVMB];
SVMresult1=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\70\SVMresult.mat');
SVMresult2=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\0510\SVMresult.mat');
SVMresult80=[SVMresult1.SVMresult70 SVMresult2.SVMresult];

GPRMetaresult1=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\70\GPRMetaresult.mat');
GPRMetaresult2=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\0510\GPRMetaresult.mat');
GPRMetaresult80=[GPRMetaresult1.GPRMetaresult70 GPRMetaresult2.GPRMetaresult];
PLSMetaresult1=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\70\PLSMetaresult.mat');
PLSMetaresult2=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\0510\PLSMetaresult.mat');
PLSMetaresult80=[PLSMetaresult1.PLSMetaresult70 PLSMetaresult2.PLSMetaresult];
RFMetaresult1=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\70\RFMetaresult.mat');
RFMetaresult2=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\0510\RFMetaresult.mat');
RFMetaresult80=[RFMetaresult1.RFMetaresult70 RFMetaresult2.RFMetaresult];
SVMMetaresult1=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\70\SVMMetaresult.mat');
SVMMetaresult2=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\0510\SVMMetaresult.mat');
SVMMetaresult80=[SVMMetaresult1.SVMMetaresult70 SVMMetaresult2.SVMMetaresult];

CMatrix1=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\70\CMatrix.mat');
CMatrix2=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\0510\CMatrix.mat');
CMatrix80=[CMatrix1.CMatrix70 CMatrix2.CMatrix];
GAStackingresult1=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\70\GASresult.mat');
GAStackingresult2=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\0510\GASresult.mat');
GAStackingresult80=[GAStackingresult1.GAStackingresult70 GAStackingresult2.GAStackingresult];
RandMatrix1=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\70\RandMatrix.mat');
RandMatrix2=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\0510\RandMatrix.mat');
RandMatrix80=[RandMatrix1.RandMatrix70;RandMatrix2.RandMatrix];
ProbTestMatrix1=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\70\ProbTestMatrix.mat');
ProbTestMatrix2=load('D:\HXJ\研究生\代码\特征提取\连续投影算法\zxw\连续投影算法-教程版\连续投影算法-教程版\数据\0510\ProbTestMatrix.mat');
ProbTestMatrix80=[ProbTestMatrix1.ProbTestMatrix70 ProbTestMatrix2.ProbTestMatrix];
iteration = 50;
proportion=0.75; %set proportion
MyRandi50=zeros(iteration,size(data,1));
for i=1:iteration
    MyRandi50(i,:)=randperm(size(data,1));
end
C=[];
X=data(:,1:end-1);
Y=data(:,end);
% for i=1:iteration
%     [SpecTrain,SpecTest,ProbTrain,ProbTest]=DivideDataSets(data,MyRandi(i,:),proportion);
%     PointCount=size(SpecTrain,1);
%     c(i)=cvpartition(PointCount,'Kfold',5);
%     for ModelIndex=1:c(i).NumTestSets
%         TestIndex=test(c(i),ModelIndex);
%         TestProp=ProbTrain(TestIndex,:);
%         if(ModelIndex==1)
%             temp=TestProp;
%         else
%             temp=cat(1, temp,TestProp);
%         end
%     end
%     C(:,i)=temp;
% end
for i=1:iteration
    [SpecTrain,SpecTest,ProbTrain,ProbTest]=DivideDataSets(data,MyRandi(i,:),proportion);
    PointCount=size(SpecTrain,1);
    c(i)=cvpartition(PointCount,'Kfold',5);
end

for i=1:iteration
    for ModelIndex=1:c(i).NumTestSets
        TestIndex=test(c(i),ModelIndex);
        TestProp=ProbTrain(TestIndex,:);
        if(ModelIndex==1)
            temp=TestProp;
        else
            temp=cat(1, temp,TestProp);
        end
    end
    C(:,i)=temp;
end
% First derivative preprocessing
X1st=FirstDerivative(X);
%划分数据
data = [];
data=[X1st Y];

RFA=[];
RFB=[];
RFresult=[];

for i=1:iteration
    trainData = []; testData = [];%训练集 %测试集
    SpecTrain=[];SpecTest=[];ProbTrain=[];ProbTest=[];
    [SpecTrain, SpecTest, ProbTrain, ProbTest]=DivideDataSets(data,MyRandi(i,:),proportion);
    % %Random forest
    [objvalue] =RF(SpecTrain, SpecTest, ProbTrain, ProbTest,X,c(i));
    if(i==1)
%         RFA=RFa;
%         RFB=RFb;
        RFresult=objvalue;
    else
%         RFA=[RFA RFa];
%         RFB=[RFB RFb];
        RFresult=[RFresult objvalue];
    end
end
% Write results to file
RFname = {'RMSEC';'R²C';'RMSEP';'R²P'};
% result=[RMSEP;R2P;RMSE;R2];
Tabl=table(RFname,RFresult);
% writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
RFRMSECeven=mean(RFresult(1,:));
RFR2Ceven=mean(RFresult(2,:));
RFRMSEeven=mean(RFresult(3,:));
RFR2even=mean(RFresult(4,:));

% %划分数据
% data = [];
% data=[X Y];
PLSA=[];
PLSB=[];
PLSresult=[];
for i=1:iteration
    trainData = []; testData = [];%训练集 %测试集
    SpecTrain=[];SpecTest=[];ProbTrain=[];ProbTest=[];
    [SpecTrain, SpecTest, ProbTrain, ProbTest]=DivideDataSets(data,MyRandi(i,:),proportion);
    % %PLS
    [objvalue] =PLS(SpecTrain, SpecTest, ProbTrain, ProbTest,c(i));
    if(iteration==1)
%         PLSA=PLSa;
%         PLSB=PLSb;
        PLSresult=objvalue;
    else
%         PLSA=[PLSA PLSa];
%         PLSB=[PLSB PLSb];
        PLSresult=[PLSresult objvalue];
    end
end
PLSRMSECeven=mean(PLSresult(1,:));
PLSR2Ceven=mean(PLSresult(2,:));
PLSRMSEeven=mean(PLSresult(3,:));
PLSR2even=mean(PLSresult(4,:));
% Write results to file
PLSname = {'RMSEC';'R²C';'RMSEP';'R²P'};
% result=[RMSEP;R2P;RMSE;R2];
Tabl=table(PLSname,PLSresult);
% writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
% boxchart(PLSresult(3,:)');
% boxchart([RFresult(4,:)' PLSresult(4,:)']);

% %SNV
% [m,n]=size(X);
% x=4000:8000/1844:12000;
% y=X;
% Xm=mean(X,2);
% dX=X-repmat(Xm,1,n);
% Xsnv=dX./repmat(sqrt(sum(dX.^2,2)/(n-1)),1,n);
% %划分数据
% data = [];
% data=[Xsnv Y];

GPRA=[];
GPRB=[];
GPRresult=[];
for i=1:iteration
    trainData = []; testData = [];%训练集 %测试集
    SpecTrain=[];SpecTest=[];ProbTrain=[];ProbTest=[];
    [SpecTrain, SpecTest, ProbTrain, ProbTest]=DivideDataSets(data,MyRandi(i,:),proportion);
    % %Random forest
    [objvalue] =GPR(SpecTrain, SpecTest, ProbTrain, ProbTest,c(i));
    if(i==1)
%         GPRA=GPRa;
%         GPRB=GPRb;
        GPRresult=objvalue;
    else
%         GPRA=[GPRA GPRa];
%         GPRB=[GPRB GPRb];
        GPRresult=[GPRresult objvalue];
    end
end
% Write results to file
GPRname = {'RMSEC';'R²C';'RMSEP';'R²P'};
% result=[RMSEP;R2P;RMSE;R2];
Tabl=table(GPRname,GPRresult);
% writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
GPRRMSECeven=mean(GPRresult(1,:));
GPRR2Ceven=mean(GPRresult(2,:));
GPRRMSEeven=mean(GPRresult(3,:));
GPRR2even=mean(GPRresult(4,:));
% 
% % Second derivative preprocessing
% X2st=diff(X,2,2);
% %划分数据
% data = [];
% data=[X2st Y];

ABA=[];
ABB=[];
ABresult=[];
for i=1:iteration
    trainData = []; testData = [];%训练集 %测试集
    SpecTrain=[];SpecTest=[];ProbTrain=[];ProbTest=[];
    [SpecTrain, SpecTest, ProbTrain, ProbTest]=DivideDataSets(data,MyRandi(i,:),proportion);
    % %Random forest
    [objvalue] =Adaboost(SpecTrain, SpecTest, ProbTrain, ProbTest,c(i));
    if(i==1)
%         ABA=ABa;
%         ABB=ABb;
        ABresult=objvalue;
    else
%         ABA=[ABA ABa];
%         ABB=[ABB ABb];
        ABresult=[ABresult objvalue];
    end
end
% Write results to file
ABname = {'RMSEC';'R²C';'RMSEP';'R²P'};
% result=[RMSEP;R2P;RMSE;R2];
Tabl=table(ABname,ABresult);
% writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
ABRMSECeven=mean(ABresult(1,:));
ABR2Ceven=mean(ABresult(2,:));
ABRMSEeven=mean(ABresult(3,:));
ABR2even=mean(ABresult(4,:));

% %SNV
% % Y = fy;
% % X = fx;
% [m,n]=size(X);
% x=4000:8000/1844:12000;
% y=X;
% Xm=mean(X,2);
% dX=X-repmat(Xm,1,n);
% Xsnv=dX./repmat(sqrt(sum(dX.^2,2)/(n-1)),1,n);
% %划分数据
% data = [];
% data=[Xsnv Y];

SVMA=[];
SVMB=[];
SVMresult=[];
for i=1:iteration
    trainData = []; testData = [];%训练集 %测试集
    SpecTrain=[];SpecTest=[];ProbTrain=[];ProbTest=[];
    [SpecTrain, SpecTest, ProbTrain, ProbTest]=DivideDataSets(data,MyRandi(i,:),proportion);
    % %SVM
    [objvalue] =SVM(SpecTrain, SpecTest, ProbTrain, ProbTest,c(i));
    if(i==1)
%         SVMA=SVMa;
%         SVMB=SVMb;
        SVMresult=objvalue;
    else
%         SVMA=[SVMA SVMa];
%         SVMB=[SVMB SVMb];
        SVMresult=[SVMresult objvalue];
    end
end
% Write results to file
SVMname = {'RMSEC';'R²C';'RMSEP';'R²P'};
% result=[RMSEP;R2P;RMSE;R2];
Tabl=table(SVMname,SVMresult);
% writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
SVMRMSECeven=mean(SVMresult(1,:));
SVMR2Ceven=mean(SVMresult(2,:));
SVMRMSEeven=mean(SVMresult(3,:));
SVMR2even=mean(SVMresult(4,:)');
% %SNV
% % Y = fy;
% % X = fx;
% [m,n]=size(X);
% x=4000:8000/1844:12000;
% y=X;
% Xm=mean(X,2);
% dX=X-repmat(Xm,1,n);
% Xsnv=dX./repmat(sqrt(sum(dX.^2,2)/(n-1)),1,n);
% %划分数据
% data = [];
% data=[Xsnv Y];
MyRandi=RandMatrix100;
KNNA=[];
KNNB=[];
KNNresult=[];
for i=1:iteration
    trainData = []; testData = [];%训练集 %测试集
    SpecTrain=[];SpecTest=[];ProbTrain=[];ProbTest=[];
    [SpecTrain, SpecTest, ProbTrain, ProbTest]=DivideDataSets(data,MyRandi(i,:),proportion);
    % %Random forest
    [objvalue] =KNN(SpecTrain, SpecTest, ProbTrain, ProbTest,c(i));
    if(i==1)
%         KNNA=KNNa;
%         KNNB=KNNb;
        KNNresult=objvalue;
    else
%         KNNA=[KNNA KNNa];
%         KNNB=[KNNB KNNb];
        KNNresult=[KNNresult objvalue];
    end
end
% Write results to file
KNNname = {'RMSEC';'R²C';'RMSEP';'R²P'};
% result=[RMSEP;R2P;RMSE;R2];
Tabl=table(KNNname,KNNresult);
% writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
KNNRMSECeven=mean(KNNresult(1,:));
KNNR2Ceven=mean(KNNresult(2,:));
KNNRMSEPeven=mean(KNNresult(3,:));
KNNR2Peven=mean(KNNresult(4,:));
% %MSC
% [m,n]=size(X);
% x=4000:8000/1844:12000;
% y=X;Me=mean(X);
% Xmsc=ones(m,n);
% for i=1:m
%     p=polyfit(Me,X(i,:),1);
%     Xmsc(i,:)=(X(i,:)-p(2)*ones(1,n))./(p(1)*ones(1,n));
% end
% %划分数据
% data = [];
% data=[Xmsc Y];

NNA=[];
NNB=[];
NNresult=[];
for i=1:iteration
    trainData = []; testData = [];%训练集 %测试集
    SpecTrain=[];SpecTest=[];ProbTrain=[];ProbTest=[];
    [SpecTrain, SpecTest, ProbTrain, ProbTest]=DivideDataSets(data,MyRandi(i,:),proportion);
    % %Random forest
    [objvalue] =FNN(SpecTrain, SpecTest, ProbTrain, ProbTest,X,c(i));
    if(i==1)
%         NNA=NNa;
%         NNB=NNb;
        NNresult=objvalue;
    else
%         NNA=[NNA NNa];
%         NNB=[NNB NNb];
        NNresult=[NNresult objvalue];
    end
end
% Write results to file
NNname = {'RMSEC';'R²C';'RMSEP';'R²P'};
% result=[RMSEP;R2P;RMSE;R2];
Tabl=table(NNname,NNresult);
% writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
NNRMSECeven=mean(NNresult(1,:));
NNR2Ceven=mean(NNresult(2,:));
NNRMSEeven=mean(NNresult(3,:));
NNR2even=mean(NNresult(4,:));
figure;
boxplotX=[RFresult(3,:)' PLSresult(3,:)' GPRresult(3,:)' ABresult(3,:)' SVMresult(3,:)' KNNresult(1,:)' FNNresult(3,:)' GPRMetaresult(3,:)' SVMMetaresult(3,:)' PLSMetaresult(3,:)' RFMetaresult(3,:)' GAStackingresult(1,:)'];
boxplot(boxplotX, 'Labels', {'RF', 'PLS', 'GPR', 'Adaboost', 'SVM', 'KNN','FNN', 'FNNMeta', 'SVMMeta', 'PLSMeta','AdaboostMeta','GAStacking'},'LabelOrientation', 'inline');
ylabel('RMSEP');
figure;
boxplotX=[RFresult(4,:)' PLSresult(4,:)' GPRresult(4,:)' ABresult(4,:)' SVMresult(4,:)' KNNresult(2,:)' FNNresult(4,:)' GPRMetaresult(4,:)' SVMMetaresult(4,:)' PLSMetaresult(4,:)' RFMetaresult(4,:)' GAStackingresult(2,:)'];
boxplot(boxplotX, 'Labels', {'RF', 'PLS', 'GPR', 'Adaboost', 'SVM', 'KNN','FNN', 'FNNMeta', 'SVMMeta', 'PLSMeta','AdaboostMeta','GAStacking'},'LabelOrientation', 'inline');
ylabel('R²');

figure;
boxplotX=[RFresult80(3,:)' PLSresult80(3,:)' GPRresult80(3,:)' ABresult80(3,:)' SVMresult80(3,:)' KNNresult80(1,:)' FNNresult80(3,:)' GPRMetaresult80(3,:)' SVMMetaresult80(3,:)' PLSMetaresult80(3,:)' RFMetaresult80(3,:)' GAStackingresult80(1,:)'];
boxplot(boxplotX, 'Labels', {'RF', 'PLS', 'GPR', 'Adaboost', 'SVM', 'KNN','FNN', 'FNNMeta', 'SVMMeta', 'PLSMeta','AdaboostMeta','GAStacking'},'LabelOrientation', 'inline');
ylabel('RMSEP');
figure;
boxplotX=[RFresult80(4,:)' PLSresult80(4,:)' GPRresult80(4,:)' ABresult80(4,:)' SVMresult80(4,:)' KNNresult80(2,:)' FNNresult80(4,:)' GPRMetaresult80(4,:)' SVMMetaresult80(4,:)' PLSMetaresult80(4,:)' RFMetaresult80(4,:)' GAStackingresult80(2,:)'];
boxplot(boxplotX, 'Labels', {'RF', 'PLS', 'GPR', 'Adaboost', 'SVM', 'KNN','FNN', 'FNNMeta', 'SVMMeta', 'PLSMeta','AdaboostMeta','GAStacking'},'LabelOrientation', 'inline');
ylabel('R²');

% 'Notch','on',中位线是否缺口
figure;
boxplotX=[RFresult80(3,:)' PLSresult80(3,:)' GPRresult80(3,:)' ABresult80(3,:)' SVMresult80(3,:)' KNNresult80(1,:)' FNNresult80(3,:)' GPRMetaresult80(3,:)' SVMMetaresult80(3,:)' PLSMetaresult80(3,:)' RFMetaresult80(3,:)' GAStackingresult80(1,:)' PSOMetaresult(3,:)'];
boxplot(boxplotX);
ylabel('RMSE');
colors =[0.9654 0.1780 0.1840;0.9654 0.1780 0.1840;0.9290 0.6940 0.1250;0.9290 0.6940 0.1250;0.9290 0.6940 0.1250;0.9290 0.6940 0.1250;0,0.5,0.5;0,0.5,0.5;0,0.5,0.5;0,0.5,0.5;0,0.5,0.5;0,0.5,0.5;0,0.5,0.5];
h = findobj(gca,'Tag','Box');
set(gca,'TickLabelInterpreter','tex','FontName','Times','FontSize',13, 'xticklabels',  {'\color[rgb]{0,0.5,0.5} RF', '\color[rgb]{0,0.5,0.5} PLS', '\color[rgb]{0,0.5,0.5} GPR', '\color[rgb]{0,0.5,0.5} AdaBoost', '\color[rgb]{0,0.5,0.5} SVR', '\color[rgb]{0,0.5,0.5} KNN','\color[rgb]{0,0.5,0.5} FNN', '\color[rgb]{0.9290 0.6940 0.1250} GPR-Meta', '\color[rgb]{0.9290 0.6940 0.1250} SVR-Meta', '\color[rgb]{0.9290 0.6940 0.1250} PLS-Meta','\color[rgb]{0.9290 0.6940 0.1250} RF-Meta','\color[rgb]{0.9654 0.1780 0.1840} GA-Stacking','\color[rgb]{0.9654 0.1780 0.1840} PSO-Stacking'},'XTickLabelRotation',90);
for j = 1:length(h)
  patch(get(h(j) , 'XData' ),get(h(j), 'YData' ),colors(j, : ) , 'FaceAlpha',.5) ;
end
text_h = findobj(gca,'Type','text');
set(text_h,'FontSize', 13);
yshift = get(text_h, 'Position'); % stop  
% yshift = cellfun(@(y) [y(1) y(2)-10 y(3)],yshift,'un',0);  % set position
% set(text_h, {'Position'}, yshift);


figure;
boxplotX=[RFresult80(4,:)' PLSresult80(4,:)' GPRresult80(4,:)' ABresult80(4,:)' SVMresult80(4,:)' KNNresult80(2,:)' FNNresult80(4,:)' GPRMetaresult80(4,:)' SVMMetaresult80(4,:)' PLSMetaresult80(4,:)' RFMetaresult80(4,:)' GAStackingresult80(2,:)' PSOMetaresult(4,:)'];
boxplot( boxplotX);
ylabel('R²');
colors =[0.9654 0.1780 0.1840;0.9654 0.1780 0.1840;0.9290 0.6940 0.1250;0.9290 0.6940 0.1250;0.9290 0.6940 0.1250;0.9290 0.6940 0.1250;0,0.5,0.5;0,0.5,0.5;0,0.5,0.5;0,0.5,0.5;0,0.5,0.5;0,0.5,0.5;0,0.5,0.5];
h = findobj(gca,'Tag','Box');
set(gca,'TickLabelInterpreter','tex','FontName','Times','FontSize',13, 'xticklabels',  {'\color[rgb]{0,0.5,0.5} RF', '\color[rgb]{0,0.5,0.5} PLS', '\color[rgb]{0,0.5,0.5} GPR', '\color[rgb]{0,0.5,0.5} AdaBoost', '\color[rgb]{0,0.5,0.5} SVR', '\color[rgb]{0,0.5,0.5} KNN','\color[rgb]{0,0.5,0.5} FNN', '\color[rgb]{0.9290 0.6940 0.1250} GPR-Meta', '\color[rgb]{0.9290 0.6940 0.1250} SVR-Meta', '\color[rgb]{0.9290 0.6940 0.1250} PLS-Meta','\color[rgb]{0.9290 0.6940 0.1250} RF-Meta','\color[rgb]{0.9654 0.1780 0.1840} GA-Stacking','\color[rgb]{0.9654 0.1780 0.1840} PSO-Stacking'},'XTickLabelRotation',90);
for j = 1:length(h)
  patch(get(h(j) , 'XData' ),get(h(j), 'YData' ),colors(j, : ) , 'FaceAlpha',.5) ;
end
% set(gca,'ycolor',colors);
text_h = findobj(gca,'Type','text');
set(text_h,'FontSize', 13);
yshift = get(text_h, 'Position'); % stop  
% yshift = cellfun(@(y) [y(1) y(2)-10 y(3)],yshift,'un',0);  % set position
% set(text_h, {'Position'}, yshift);

% 'Notch','on',中位线是否缺口
figure;
boxplotX=[RFresult80(1,:)' PLSresult80(1,:)' GPRresult80(1,:)' ABresult80(1,:)' SVMresult80(1,:)' KNNresult80(1,:)' FNNresult80(1,:)' GPRMetaresult80(1,:)' SVMMetaresult80(1,:)' PLSMetaresult80(1,:)' RFMetaresult80(1,:)' GAStackingresult80(3,:)' PSOMetaresult(1,:)'];
boxplot(boxplotX);
ylabel('RMSE');
colors =[0.9654 0.1780 0.1840;0.9654 0.1780 0.1840;0.9290 0.6940 0.1250;0.9290 0.6940 0.1250;0.9290 0.6940 0.1250;0.9290 0.6940 0.1250;0,0.5,0.5;0,0.5,0.5;0,0.5,0.5;0,0.5,0.5;0,0.5,0.5;0,0.5,0.5;0,0.5,0.5];
h = findobj(gca,'Tag','Box');
set(gca,'TickLabelInterpreter','tex','FontName','Times','FontSize',13, 'xticklabels',  {'\color[rgb]{0,0.5,0.5} RF', '\color[rgb]{0,0.5,0.5} PLS', '\color[rgb]{0,0.5,0.5} GPR', '\color[rgb]{0,0.5,0.5} AdaBoost', '\color[rgb]{0,0.5,0.5} SVR', '\color[rgb]{0,0.5,0.5} KNN','\color[rgb]{0,0.5,0.5} FNN', '\color[rgb]{0.9290 0.6940 0.1250} GPR-Meta', '\color[rgb]{0.9290 0.6940 0.1250} SVR-Meta', '\color[rgb]{0.9290 0.6940 0.1250} PLS-Meta','\color[rgb]{0.9290 0.6940 0.1250} RF-Meta','\color[rgb]{0.9654 0.1780 0.1840} GA-Stacking','\color[rgb]{0.9654 0.1780 0.1840} PSO-Stacking'},'XTickLabelRotation',90);
for j = 1:length(h)
  patch(get(h(j) , 'XData' ),get(h(j), 'YData' ),colors(j, : ) , 'FaceAlpha',.5) ;
end
text_h = findobj(gca,'Type','text');
set(text_h,'FontSize', 13);
yshift = get(text_h, 'Position'); % stop  
% yshift = cellfun(@(y) [y(1) y(2)-10 y(3)],yshift,'un',0);  % set position
% set(text_h, {'Position'}, yshift);


figure;
boxplotX=[RFresult80(2,:)' PLSresult80(2,:)' GPRresult80(2,:)' ABresult80(2,:)' SVMresult80(2,:)' KNNresult80(2,:)' FNNresult80(2,:)' GPRMetaresult80(2,:)' SVMMetaresult80(2,:)' PLSMetaresult80(2,:)' RFMetaresult80(2,:)' GAStackingresult80(4,:)' PSOMetaresult(2,:)'];
boxplot( boxplotX);
ylabel('R²');
colors =[0.9654 0.1780 0.1840;0.9654 0.1780 0.1840;0.9290 0.6940 0.1250;0.9290 0.6940 0.1250;0.9290 0.6940 0.1250;0.9290 0.6940 0.1250;0,0.5,0.5;0,0.5,0.5;0,0.5,0.5;0,0.5,0.5;0,0.5,0.5;0,0.5,0.5;0,0.5,0.5];
h = findobj(gca,'Tag','Box');
set(gca,'TickLabelInterpreter','tex','FontName','Times','FontSize',13, 'xticklabels',  {'\color[rgb]{0,0.5,0.5} RF', '\color[rgb]{0,0.5,0.5} PLS', '\color[rgb]{0,0.5,0.5} GPR', '\color[rgb]{0,0.5,0.5} AdaBoost', '\color[rgb]{0,0.5,0.5} SVR', '\color[rgb]{0,0.5,0.5} KNN','\color[rgb]{0,0.5,0.5} FNN', '\color[rgb]{0.9290 0.6940 0.1250} GPR-Meta', '\color[rgb]{0.9290 0.6940 0.1250} SVR-Meta', '\color[rgb]{0.9290 0.6940 0.1250} PLS-Meta','\color[rgb]{0.9290 0.6940 0.1250} RF-Meta','\color[rgb]{0.9654 0.1780 0.1840} GA-Stacking','\color[rgb]{0.9654 0.1780 0.1840} PSO-Stacking'},'XTickLabelRotation',90);
for j = 1:length(h)
  patch(get(h(j) , 'XData' ),get(h(j), 'YData' ),colors(j, : ) , 'FaceAlpha',.5) ;
end
% set(gca,'ycolor',colors);
text_h = findobj(gca,'Type','text');
set(text_h,'FontSize', 13);
yshift = get(text_h, 'Position'); % stop  
% yshift = cellfun(@(y) [y(1) y(2)-10 y(3)],yshift,'un',0);  % set position
% set(text_h, {'Position'}, yshift);

figure;
boxplotX=[RC2(3,:)' RP2(3,:)' GAStackingresult80(3,1:30)' RPRC3(3,:)'];
boxplot(boxplotX, 'Labels', {'RC2', 'RP2', 'RP3RC', 'RPRC3'},'LabelOrientation', 'inline');
ylabel('RMSE');
figure;
boxplotX=[RC2(4,:)' RP2(4,:)' GAStackingresult80(4,1:30)' RPRC3(4,:)'];
boxplot(boxplotX, 'Labels', {'RC2', 'RP2', 'RP3RC', 'RPRC3'},'LabelOrientation', 'inline');
ylabel('R²');
figure;
boxplotX=[RC2(1,:)' RP2(1,:)' GAStackingresult80(1,1:30)' RPRC3(1,:)'];
boxplot(boxplotX, 'Labels', {'RC2', 'RP2', 'RP3RC', 'RPRC3'},'LabelOrientation', 'inline');
ylabel('RMSEP');
figure;
boxplotX=[RC2(2,:)' RP2(2,:)' GAStackingresult80(2,1:30)' RPRC3(2,:)'];
boxplot(boxplotX, 'Labels', {'RC2', 'RP2', 'RP3RC', 'RPRC3'},'LabelOrientation', 'inline');
ylabel('R²P');
RC2RMSEPmean=mean(RC2(1,:));
RC2R2Pmean=mean(RC2(2,:));
RC2RMSECmean=mean(RC2(3,:));
RC2R2Cmean=mean(RC2(4,:));

RP2RMSEPmean=mean(RP2(1,:));
RP2R2Pmean=mean(RP2(2,:));
RP2RMSECmean=mean(RP2(3,:));
RP2R2Cmean=mean(RP2(4,:));

RP3PCRMSEPmean=mean(GAStackingresult80(1,1:30));
RP3PCR2Pmean=mean(GAStackingresult80(2,1:30));
RP3PCRMSECmean=mean(GAStackingresult80(3,1:30));
RP3PCR2Cmean=mean(GAStackingresult80(4,1:30));

RPRC3RMSEPmean=mean(RPRC3(1,:));
RPRC3R2Pmean=mean(RPRC3(2,:));
RPRC3RMSECmean=mean(RPRC3(3,:));
RPRC3R2Cmean=mean(RPRC3(4,:));
% colors =[0.1 0.7 0.7;0.1 0.7 0.7;0.1 0.7 0.7;0.1 0.7 0.7;0.1 0.7 0.7;0.1 0.7 0.7;0.1 0.7 0.7;1 0.4 0.6;1 0.4 0.6;1 0.4 0.6;1 0.4 0.6;0.13,0.55,0.13];
colors=rand(4,3);
h = findobj(gca,'Tag','Box');
for j = 1:length(h)
  patch(get(h(j) , 'XData' ),get(h(j), 'YData' ),colors(j, : ) , 'FaceAlpha',.5) ;
end
RFRMSECmean=mean(RFresult80(1,:));
RFR2Cmean=mean(RFresult80(2,:));
RFRMSEPmean=mean(RFresult80(3,:));
RFR2Pmean=mean(RFresult80(4,:));

PLSRMSECmean=mean(PLSresult80(1,:));
PLSR2Cmean=mean(PLSresult80(2,:));
PLSRMSEPmean=mean(PLSresult80(3,:));
PLSR2Pmean=mean(PLSresult80(4,:));

GPRRMSECmean=mean(GPRresult80(1,:));
GPRR2Cmean=mean(GPRresult80(2,:));
GPRRMSEPmean=mean(GPRresult80(3,:));
GPRR2Pmean=mean(GPRresult80(4,:));

ABRMSECmean=mean(ABresult80(1,:));
ABR2Cmean=mean(ABresult80(2,:));
ABRMSEPmean=mean(ABresult80(3,:));
ABR2Pmean=mean(ABresult80(4,:));

SVMRMSECmean=mean(SVMresult80(1,:));
SVMR2Cmean=mean(SVMresult80(2,:));
SVMRMSEPmean=mean(SVMresult80(3,:));
SVMR2Pmean=mean(SVMresult80(4,:));

KNNRMSECmean=mean(KNNresult80(1,:));
KNNR2Cmean=mean(KNNresult80(2,:));


FNNRMSECmean=mean(FNNresult80(1,:));
FNNR2Cmean=mean(FNNresult80(2,:));
FNNRMSEPmean=mean(FNNresult80(3,:));
FNNR2Pmean=mean(FNNresult80(4,:));

GPRMetaRMSECmean=mean(GPRMetaresult80(1,:));
GPRMetaR2Cmean=mean(GPRMetaresult80(2,:));
GPRMetaRMSEPmean=mean(GPRMetaresult80(3,:));
GPRMetaR2Pmean=mean(GPRMetaresult80(4,:));

SVMMetaRMSECmean=mean(SVMMetaresult80(1,:));
SVMMetaR2Cmean=mean(SVMMetaresult80(2,:));
SVMMetaRMSEPmean=mean(SVMMetaresult80(3,:));
SVMMetaR2Pmean=mean(SVMMetaresult80(4,:));

PLSMetaRMSECmean=mean(PLSMetaresult80(1,:));
PLSMetaR2Cmean=mean(PLSMetaresult80(2,:));
PLSMetaRMSEPmean=mean(PLSMetaresult80(3,:));
PLSMetaR2Pmean=mean(PLSMetaresult80(4,:));

RFMetaRMSECmean=mean(RFMetaresult80(1,:));
RFMetaR2Cmean=mean(RFMetaresult80(2,:));
RFMetaRMSEPmean=mean(RFMetaresult80(3,:));
RFMetaR2Pmean=mean(RFMetaresult80(4,:));

GASMetaRMSEPmean=mean(GAStackingresult80(1,:));
GASMetaR2Pmean=mean(GAStackingresult80(2,:));
GASMetaRMSECmean=mean(GAStackingresult80(3,:));
GASMetaR2Cmean=mean(GAStackingresult80(4,:));
 MetaResult=zeros(4,10);
for i=1:10
    A0=[];  B0=[];  A1=[]; B1=[];
    A0=[RFA(:,i) SVMA(:,i) GSA(:,i) PLSA(:,i) ABA(:,i) KNNA(:,i) FNNA(:,i)];
    B0=[RFB(:,i) SVMB(:,i) GSB(:,i) PLSB(:,i) ABB(:,i) KNNB(:,i) FNNB(:,i)];
    A1=[SVMA(:,i) PLSA(:,i) FNNA(:,i)];
    B1=[SVMB(:,i) PLSB(:,i) FNNB(:,i)];
%     A0=[RFA(:,i) SVMA(:,i) PLSA(:,i) ABA(:,i) KNNA(:,i)];
%     B0=[RFB(:,i) SVMB(:,i) PLSB(:,i) ABB(:,i) KNNB(:,i)];
%     A1=[SVMA(:,i) PLSA(:,i)];
%     B1=[SVMB(:,i) PLSB(:,i)];
    y=[];
%     MetaResult=zeros(4,10);
    %迭代轮次
    iter=50;
    %Name:genmain05.m
    % clear
    % clf
    % popsize=9;                                       %群体大小
    % chromlength=7;                                   %字符串长度（个体长度）
    popsize=30;                                       %群体大小
    chromlength=9;                                   %字符串长度（个体长度）
    count = zeros(iter,chromlength);
    pc=0.6;                                           %交叉概率
    pm=0.001;                                         %变异概率
    pop=initpop(popsize,chromlength);                 %随机产生初始群体
    bestindividuals=zeros(iter,chromlength);     %每轮最好的特征选择
    for ii=1:iter                                    %20为迭代次数
        %     objvalue(i,1)=R2/abs(1-(R2*R2)/(R2C*R2C));
        [objvalue]=calobjvalue3(pop,A0,CMatrix(:,i),B0,ProbTestMatrix(:,i));                      %计算目标函数
        fitvalue=objvalue;                   %计算群体中每个个体的适应度
        [newpop]=selection(pop,fitvalue);                 %复制
        [newpop]=crossover_multiv(newpop,pc);                       %交叉
        [newpop]=mutation(newpop,pc);                        %变异
        %     pop=newpop;
        [bestindividual,bestfit]=best(pop,fitvalue);%求出群体中适应值最大的个体及其适应值
        y(ii)=max(bestfit);
        bestindividuals(ii,:)=bestindividual;
        nn(ii)=ii;
        %     pop5=bestindividual;
        %     x(ii)=decodechrom(pop5,1,chromlength)*10/1023;
        pop=newpop;
        count(ii,:)=sum(pop,1);
    end
    % fplot('10*sin(5*x)+7*cos(4*x)',[0 10])
    % plot();
    % hold on
    
    % hold off
    [M ,I]=max(y);
    Bestindividual=bestindividuals(I,:);
    ind=logical(Bestindividual(1:7));
    newSpec=A0(:,ind);
    newtestSpec=B0(:,ind);
    num=num2str(Bestindividual(8:9));%次级模型编号
    temp1=bin2dec(num) %将编号转化成十进制数
    if temp1==0
        result=MetaGPR(newSpec, newtestSpec,CMatrix(:,i), ProbTestMatrix(:,i));
        %        %         次级模型运算结果写入文件
        %         GPRmetaName = {'RMSEC';'R²C';'RMSEP';'R²P'};
        %         % result=[RMSEP;R2P;RMSE;R2];
        %         Tabl=table(GPRmetaName,result);
        %         writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
        % %         writetable(Tabl,'m.xls','Range','C34:D38','WriteVariableNames',true);
    end
    if temp1==1
        result=MetaSVM(newSpec, newtestSpec, CMatrix(:,i), ProbTestMatrix(:,i));
        %         %         次级模型运算结果写入文件
        %         SVMmetaName = {'RMSEC';'R²C';'RMSEP';'R²P'};
        %         % result=[RMSEP;R2P;RMSE;R2];
        %         Tabl=table(SVMmetaName,result);
        %               writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
        % %         writetable(Tabl,'m.xls','Range','C34:D38','WriteVariableNames',true);
    end
    if temp1==2
        result=MetaPLS(newSpec, newtestSpec,CMatrix(:,i), ProbTestMatrix(:,i));
        %         %         次级模型运算结果写入文件
        %         SVMmetaName = {'RMSEC';'R²C';'RMSEP';'R²P'};
        %         % result=[RMSEP;R2P;RMSE;R2];
        %         Tabl=table(SVMmetaName,result);
        %                 writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
        % %         writetable(Tabl,'m.xls','Range','C34:D38','WriteVariableNames',true);
    end
    if temp1==3
        result=MetaRF(newSpec, newtestSpec, CMatrix(:,i), ProbTestMatrix(:,i));
        %              %         次级模型运算结果写入文件
        %        RFmetaName = {'RMSEC';'R²C';'RMSEP';'R²P'};
        %         % result=[RMSEP;R2P;RMSE;R2];
        %         Tabl=table(RFmetaName,result);
        %                writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
        % %         writetable(Tabl,'m.xls','Range','C34:D38','WriteVariableNames',true);
    end
    MetaResult(:,i)=result;
%     if(i==1)
%         MetaResult=result;
%     else
%         MetaResult=[MetaResult result];
%     end
    %         次级模型运算结果写入文件
    metaName = {'RMSEC';'R²C';'RMSEP';'R²P'};
    % result=[RMSEP;R2P;RMSE;R2];
    Tabl=table(metaName,MetaResult);
    writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
    %         writetable(Tabl,'m.xls','Range','C34:D38','WriteVariableNames',true);
end

for i = 1:5
    
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
    
    
    A0=[RFa SVMa GSa PLSa ABa KNNa NNa];
    B0=[RFb SVMb GSb PLSb ABb KNNb NNb];
    A1=[SVMa GSa PLSa];
    B1=[SVMb GSb PLSb];
    
    y=[];
    %迭代轮次
    iter=50;
    %Name:genmain05.m
    % clear
    % clf
    % popsize=9;                                       %群体大小
    % chromlength=7;                                   %字符串长度（个体长度）
    popsize=30;                                       %群体大小
    chromlength=9;                                   %字符串长度（个体长度）
    count = zeros(iter,chromlength);
    pc=0.6;                                           %交叉概率
    pm=0.001;                                         %变异概率
    pop=initpop(popsize,chromlength);                 %随机产生初始群体
    bestindividuals=zeros(iter,chromlength);     %每轮最好的特征选择
    for ii=1:iter                                    %20为迭代次数
        %     objvalue(i,1)=R2/abs(1-(R2*R2)/(R2C*R2C));
        [objvalue]=calobjvalue2(pop,A0,C,B0,ProbTest);                      %计算目标函数
        fitvalue=objvalue;                   %计算群体中每个个体的适应度
        [newpop]=selection(pop,fitvalue);                 %复制
        [newpop]=crossover_multiv(newpop,pc);                       %交叉
        [newpop]=mutation(newpop,pc);                        %变异
        %     pop=newpop;
        [bestindividual,bestfit]=best(pop,fitvalue);%求出群体中适应值最大的个体及其适应值
        y(ii)=max(bestfit);
        bestindividuals(ii,:)=bestindividual;
        nn(ii)=ii;
        %     pop5=bestindividual;
        %     x(ii)=decodechrom(pop5,1,chromlength)*10/1023;
        pop=newpop;
        count(ii,:)=sum(pop,1);
    end
    % fplot('10*sin(5*x)+7*cos(4*x)',[0 10])
    % plot();
    % hold on
    
    % hold off
    [M ,I]=max(y);
    Bestindividual=bestindividuals(I,:);
    ind=logical(Bestindividual(1:7));
    newSpec=A0(:,ind);
    newtestSpec=B0(:,ind);
    num=num2str(Bestindividual(8:9));%次级模型编号
    temp1=bin2dec(num) %将编号转化成十进制数
    if temp1==0
        sigma0 = 0.2;
        kparams0 = [3.5, 6.2];
        stacking = fitrgp(newSpec,C,'KernelFunction','squaredexponential',...
            'KernelParameters',kparams0,'Sigma',sigma0);
        Ypredict5=predict(stacking,newtestSpec);
        YpredictC=predict(stacking,newSpec);
        %模型分析
        co=corr(ProbTest,Ypredict5,'type','Pearson');
        RMSE=sqrt(sum((ProbTest-Ypredict5).^2)/size(Ypredict5,1));
        R2 = corr(ProbTest,Ypredict5)^2;
        RMSEC=sqrt(sum((C-YpredictC).^2)/size(YpredictC,1));
        R2C = corr(C,YpredictC)^2;
        %         RMSE=num2str(roundn(RMSE,-4));
        result=[RMSEC;R2C;RMSE;R2];
        co=num2str(roundn(co,-4));
        RMSE=num2str(roundn(RMSE,-4));
        R2=num2str(roundn(R2,-4));
        RMSEC=num2str(roundn(RMSEC,-4));
        R2C=num2str(roundn(R2C,-4));
        msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
            ['R2C：',R2C,'R2：',R2]},...
            '次级模型高斯回归');
        %         次级模型运算结果写入文件
        GPRmetaName = {'RMSEC';'R²C';'RMSEP';'R²P'};
        % result=[RMSEP;R2P;RMSE;R2];
        Tabl=table(GPRmetaName,result);
        writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
        %         writetable(Tabl,'m.xls','Range','C34:D38','WriteVariableNames',true);
    end
    if temp1==1
        Stacking = fitrsvm(newSpec,C,'KernelFunction','gaussian','OptimizeHyperparameters','auto',...
            'HyperparameterOptimizationOptions',struct('Optimizer','gridsearch','NumGridDivisions',15,...
            'ShowPlots',false,'MaxObjectiveEvaluations',60));
        Ypredict5=predict(Stacking,newtestSpec);
        YpredictC=predict(Stacking,newSpec);
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
        %         次级模型运算结果写入文件
        SVMmetaName = {'RMSEC';'R²C';'RMSEP';'R²P'};
        % result=[RMSEP;R2P;RMSE;R2];
        Tabl=table(SVMmetaName,result);
        writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
        %         writetable(Tabl,'m.xls','Range','C34:D38','WriteVariableNames',true);
    end
    if temp1==2
        Stacking = fitlm(newSpec,C);
        Ypredict5=predict(Stacking,newtestSpec);
        YpredoctA=predict(Stacking,newSpec);
        %模型分析
        co=corr(ProbTest,Ypredict5,'type','Pearson');
        RMSE=sqrt(sum((ProbTest-Ypredict5).^2)/size(Ypredict5,1));
        R2 = corr(ProbTest,Ypredict5)^2;
        %     coP=corr(prob,YpredoctA,'type','Pearson');
        RMSEP=sqrt(sum((C-YpredoctA).^2)/size(YpredoctA,1));
        R2P = corr(C,YpredoctA)^2;
        result=[RMSEP;R2P;RMSE;R2];
        co=num2str(roundn(co,-4));
        RMSE=num2str(roundn(RMSE,-4));
        R2=num2str(roundn(R2,-4));
        RMSEP=num2str(roundn(RMSEP,-4));
        R2P=num2str(roundn(R2P,-4));
        msgbox({'您运算的结果为:',['RMSEC：',RMSEP,'RMSE：',RMSE],...
            ['R2C：',R2P,'R2：',R2]},...
            '次级模型PLS');
        %         次级模型运算结果写入文件
        SVMmetaName = {'RMSEC';'R²C';'RMSEP';'R²P'};
        % result=[RMSEP;R2P;RMSE;R2];
        Tabl=table(SVMmetaName,result);
        writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
        %         writetable(Tabl,'m.xls','Range','C34:D38','WriteVariableNames',true);
    end
    if temp1==3
        isCategorical = zeros(size(newSpec,2),1);
        treeNum=100;
        nodeNum=15;
        Stacking = TreeBagger(treeNum,newSpec,C,'Method','R','OOBPredictorImportance','On',...
            'CategoricalPredictors',find(isCategorical == 1),...
            'MinLeafSize',nodeNum);
        Ypredict5=predict(Stacking,newtestSpec);
        YpredictC=predict(Stacking,newSpec);
        %模型分析
        co=corr(ProbTest,Ypredict5,'type','Pearson');
        RMSE=sqrt(sum((ProbTest-Ypredict5).^2)/size(Ypredict5,1));
        R2 = corr(ProbTest,Ypredict5)^2;
        RMSEC=sqrt(sum((C-YpredictC).^2)/size(YpredictC,1));
        R2C = corr(C,YpredictC)^2;
        result=[RMSEC;R2C;RMSE;R2];
        RMSE=num2str(roundn(RMSE,-4));
        R2=num2str(roundn(R2,-4));
        RMSEC=num2str(roundn(RMSEC,-4));
        R2C=num2str(roundn(R2C,-4));
        msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
            ['R2C：',R2C,'R2：',R2]},...
            '次级随机森林');
        %         次级模型运算结果写入文件
        RFmetaName = {'RMSEC';'R²C';'RMSEP';'R²P'};
        % result=[RMSEP;R2P;RMSE;R2];
        Tabl=table(RFmetaName,result);
        writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
        %         writetable(Tabl,'m.xls','Range','C34:D38','WriteVariableNames',true);
    end
    
    %迭代轮次
    iteration=50;
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
    for ii=1:iteration                                    %20为迭代次数
        %     objvalue(i,1)=(R2*R2) /abs(Im*(1-(R2*R2)/(R2C*R2C)))
        Im=0.5;
        [objvalue]=calobjvalue4(pop,A0,C,B0,ProbTest,Im);                      %计算目标函数
        fitvalue=objvalue;                   %计算群体中每个个体的适应度
        [newpop]=selection(pop,fitvalue);                 %复制
        [newpop]=crossover_multiv(newpop,pc);                       %交叉
        [newpop]=mutation(newpop,pc);                        %变异
        %     pop=newpop;
        [bestindividual,bestfit]=best(pop,fitvalue);%求出群体中适应值最大的个体及其适应值
        y(ii)=max(bestfit);
        bestindividuals(ii,:)=bestindividual;
        nn(ii)=ii;
        %     pop5=bestindividual;
        %     x(ii)=decodechrom(pop5,1,chromlength)*10/1023;
        pop=newpop;
        count(ii,:)=sum(pop,1);
    end
    % fplot('10*sin(5*x)+7*cos(4*x)',[0 10])
    % plot();
    % hold on
    
    % hold off
    [M ,I]=max(y);
    Bestindividual=bestindividuals(I,:);
    ind=logical(Bestindividual(1:7));
    newSpec=A0(:,ind);
    newtestSpec=B0(:,ind);
    num=num2str(Bestindividual(8:9));%次级模型编号
    temp1=bin2dec(num) %将编号转化成十进制数
    if temp1==0
        sigma0 = 0.2;
        kparams0 = [3.5, 6.2];
        stacking = fitrgp(newSpec,C,'KernelFunction','squaredexponential',...
            'KernelParameters',kparams0,'Sigma',sigma0);
        Ypredict5=predict(stacking,newtestSpec);
        YpredictC=predict(stacking,newSpec);
        %模型分析
        co=corr(ProbTest,Ypredict5,'type','Pearson');
        RMSE=sqrt(sum((ProbTest-Ypredict5).^2)/size(Ypredict5,1));
        R2 = corr(ProbTest,Ypredict5)^2;
        RMSEC=sqrt(sum((C-YpredictC).^2)/size(YpredictC,1));
        R2C = corr(C,YpredictC)^2;
        %         RMSE=num2str(roundn(RMSE,-4));
        result=[RMSEC;R2C;RMSE;R2];
        co=num2str(roundn(co,-4));
        RMSE=num2str(roundn(RMSE,-4));
        R2=num2str(roundn(R2,-4));
        RMSEC=num2str(roundn(RMSEC,-4));
        R2C=num2str(roundn(R2C,-4));
        msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
            ['R2C：',R2C,'R2：',R2]},...
            '次级模型高斯回归');
        %         次级模型运算结果写入文件
        GPRmetaName = {'RMSEC';'R²C';'RMSEP';'R²P'};
        % result=[RMSEP;R2P;RMSE;R2];
        Tabl=table(GPRmetaName,result);
        writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
        %         writetable(Tabl,'m.xls','Range','E34:F38','WriteVariableNames',true);
    end
    if temp1==1
        Stacking = fitrsvm(newSpec,C,'KernelFunction','gaussian','OptimizeHyperparameters','auto',...
            'HyperparameterOptimizationOptions',struct('Optimizer','gridsearch','NumGridDivisions',15,...
            'ShowPlots',false,'MaxObjectiveEvaluations',60));
        Ypredict5=predict(Stacking,newtestSpec);
        YpredictC=predict(Stacking,newSpec);
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
        %         次级模型运算结果写入文件
        SVMmetaName = {'RMSEC';'R²C';'RMSEP';'R²P'};
        % result=[RMSEP;R2P;RMSE;R2];
        Tabl=table(SVMmetaName,result);
        writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
        %         writetable(Tabl,'m.xls','Range','E34:F38','WriteVariableNames',true);
    end
    if temp1==2
        Stacking = fitlm(newSpec,C);
        Ypredict5=predict(Stacking,newtestSpec);
        YpredoctA=predict(Stacking,newSpec);
        %模型分析
        co=corr(ProbTest,Ypredict5,'type','Pearson');
        RMSE=sqrt(sum((ProbTest-Ypredict5).^2)/size(Ypredict5,1));
        R2 = corr(ProbTest,Ypredict5)^2;
        %     coP=corr(prob,YpredoctA,'type','Pearson');
        RMSEP=sqrt(sum((C-YpredoctA).^2)/size(YpredoctA,1));
        R2P = corr(C,YpredoctA)^2;
        result=[RMSEP;R2P;RMSE;R2];
        co=num2str(roundn(co,-4));
        RMSE=num2str(roundn(RMSE,-4));
        R2=num2str(roundn(R2,-4));
        RMSEP=num2str(roundn(RMSEP,-4));
        R2P=num2str(roundn(R2P,-4));
        msgbox({'您运算的结果为:',['RMSEC：',RMSEP,'RMSE：',RMSE],...
            ['R2C：',R2P,'R2：',R2]},...
            '次级模型PLS');
        %         次级模型运算结果写入文件
        SVMmetaName = {'RMSEC';'R²C';'RMSEP';'R²P'};
        % result=[RMSEP;R2P;RMSE;R2];
        Tabl=table(SVMmetaName,result);
        writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
        %         writetable(Tabl,'m.xls','Range','E34:F38','WriteVariableNames',true);
    end
    if temp1==3
        isCategorical = zeros(size(newSpec,2),1);
        treeNum=100;
        nodeNum=15;
        Stacking = TreeBagger(treeNum,newSpec,C,'Method','R','OOBPredictorImportance','On',...
            'CategoricalPredictors',find(isCategorical == 1),...
            'MinLeafSize',nodeNum);
        Ypredict5=predict(Stacking,newtestSpec);
        YpredictC=predict(Stacking,newSpec);
        %模型分析
        co=corr(ProbTest,Ypredict5,'type','Pearson');
        RMSE=sqrt(sum((ProbTest-Ypredict5).^2)/size(Ypredict5,1));
        R2 = corr(ProbTest,Ypredict5)^2;
        RMSEC=sqrt(sum((C-YpredictC).^2)/size(YpredictC,1));
        R2C = corr(C,YpredictC)^2;
        result=[RMSEC;R2C;RMSE;R2];
        RMSE=num2str(roundn(RMSE,-4));
        R2=num2str(roundn(R2,-4));
        RMSEC=num2str(roundn(RMSEC,-4));
        R2C=num2str(roundn(R2C,-4));
        msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
            ['R2C：',R2C,'R2：',R2]},...
            '次级随机森林');
        %         次级模型运算结果写入文件
        RFmetaName = {'RMSEC';'R²C';'RMSEP';'R²P'};
        % result=[RMSEP;R2P;RMSE;R2];
        Tabl=table(RFmetaName,result);
        writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
        %         writetable(Tabl,'m.xls','Range','E34:F38','WriteVariableNames',true);
    end
    
    %迭代轮次
    iteration=50;
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
    for ii=1:iteration                                    %20为迭代次数
        %     objvalue=R2;
        [objvalue]=calobjvalue3(pop,A0,C,B0,ProbTest);                      %计算目标函数
        fitvalue=objvalue;                   %计算群体中每个个体的适应度
        [newpop]=selection(pop,fitvalue);                 %复制
        [newpop]=crossover_multiv(newpop,pc);                       %交叉
        [newpop]=mutation(newpop,pc);                        %变异
        %     pop=newpop;
        [bestindividual,bestfit]=best(pop,fitvalue);%求出群体中适应值最大的个体及其适应值
        y(ii)=max(bestfit);
        bestindividuals(ii,:)=bestindividual;
        nn(ii)=ii;
        %     pop5=bestindividual;
        %     x(ii)=decodechrom(pop5,1,chromlength)*10/1023;
        pop=newpop;
        count(ii,:)=sum(pop,1);
    end
    % fplot('10*sin(5*x)+7*cos(4*x)',[0 10])
    % plot();
    % hold on
    
    % hold off
    [M ,I]=max(y);
    Bestindividual=bestindividuals(I,:);
    ind=logical(Bestindividual(1:7));
    newSpec=A0(:,ind);
    newtestSpec=B0(:,ind);
    num=num2str(Bestindividual(8:9));%次级模型编号
    temp1=bin2dec(num) %将编号转化成十进制数
    if temp1==0
        sigma0 = 0.2;
        kparams0 = [3.5, 6.2];
        stacking = fitrgp(newSpec,C,'KernelFunction','squaredexponential',...
            'KernelParameters',kparams0,'Sigma',sigma0);
        Ypredict5=predict(stacking,newtestSpec);
        YpredictC=predict(stacking,newSpec);
        %模型分析
        co=corr(ProbTest,Ypredict5,'type','Pearson');
        RMSE=sqrt(sum((ProbTest-Ypredict5).^2)/size(Ypredict5,1));
        R2 = corr(ProbTest,Ypredict5)^2;
        RMSEC=sqrt(sum((C-YpredictC).^2)/size(YpredictC,1));
        R2C = corr(C,YpredictC)^2;
        %         RMSE=num2str(roundn(RMSE,-4));
        result=[RMSEC;R2C;RMSE;R2];
        co=num2str(roundn(co,-4));
        RMSE=num2str(roundn(RMSE,-4));
        R2=num2str(roundn(R2,-4));
        RMSEC=num2str(roundn(RMSEC,-4));
        R2C=num2str(roundn(R2C,-4));
        msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
            ['R2C：',R2C,'R2：',R2]},...
            '次级模型高斯回归');
        %         次级模型运算结果写入文件
        GPRmetaName = {'RMSEC';'R²C';'RMSEP';'R²P'};
        % result=[RMSEP;R2P;RMSE;R2];
        Tabl=table(GPRmetaName,result);
        writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
    end
    if temp1==1
        Stacking = fitrsvm(newSpec,C,'KernelFunction','gaussian','OptimizeHyperparameters','auto',...
            'HyperparameterOptimizationOptions',struct('Optimizer','gridsearch','NumGridDivisions',15,...
            'ShowPlots',false,'MaxObjectiveEvaluations',60));
        Ypredict5=predict(Stacking,newtestSpec);
        YpredictC=predict(Stacking,newSpec);
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
        %         次级模型运算结果写入文件
        SVMmetaName = {'RMSEC';'R²C';'RMSEP';'R²P'};
        % result=[RMSEP;R2P;RMSE;R2];
        Tabl=table(SVMmetaName,result);
        writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
    end
    if temp1==2
        Stacking = fitlm(newSpec,C);
        Ypredict5=predict(Stacking,newtestSpec);
        YpredoctA=predict(Stacking,newSpec);
        %模型分析
        co=corr(ProbTest,Ypredict5,'type','Pearson');
        RMSE=sqrt(sum((ProbTest-Ypredict5).^2)/size(Ypredict5,1));
        R2 = corr(ProbTest,Ypredict5)^2;
        %     coP=corr(prob,YpredoctA,'type','Pearson');
        RMSEP=sqrt(sum((C-YpredoctA).^2)/size(YpredoctA,1));
        R2P = corr(C,YpredoctA)^2;
        result=[RMSEP;R2P;RMSE;R2];
        co=num2str(roundn(co,-4));
        RMSE=num2str(roundn(RMSE,-4));
        R2=num2str(roundn(R2,-4));
        RMSEP=num2str(roundn(RMSEP,-4));
        R2P=num2str(roundn(R2P,-4));
        msgbox({'您运算的结果为:',['RMSEC：',RMSEP,'RMSE：',RMSE],...
            ['R2C：',R2P,'R2：',R2]},...
            '次级模型PLS');
        %         次级模型运算结果写入文件
        SVMmetaName = {'RMSEC';'R²C';'RMSEP';'R²P'};
        % result=[RMSEP;R2P;RMSE;R2];
        Tabl=table(SVMmetaName,result);
        writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
    end
    if temp1==3
        isCategorical = zeros(size(newSpec,2),1);
        treeNum=100;
        nodeNum=15;
        Stacking = TreeBagger(treeNum,newSpec,C,'Method','R','OOBPredictorImportance','On',...
            'CategoricalPredictors',find(isCategorical == 1),...
            'MinLeafSize',nodeNum);
        Ypredict5=predict(Stacking,newtestSpec);
        YpredictC=predict(Stacking,newSpec);
        %模型分析
        co=corr(ProbTest,Ypredict5,'type','Pearson');
        RMSE=sqrt(sum((ProbTest-Ypredict5).^2)/size(Ypredict5,1));
        R2 = corr(ProbTest,Ypredict5)^2;
        RMSEC=sqrt(sum((C-YpredictC).^2)/size(YpredictC,1));
        R2C = corr(C,YpredictC)^2;
        result=[RMSEC;R2C;RMSE;R2];
        RMSE=num2str(roundn(RMSE,-4));
        R2=num2str(roundn(R2,-4));
        RMSEC=num2str(roundn(RMSEC,-4));
        R2C=num2str(roundn(R2C,-4));
        msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
            ['R2C：',R2C,'R2：',R2]},...
            '次级随机森林');
        %         次级模型运算结果写入文件
        RFmetaName = {'RMSEC';'R²C';'RMSEP';'R²P'};
        % result=[RMSEP;R2P;RMSE;R2];
        Tabl=table(RFmetaName,result);
        writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
    end
    
    
    %选三个最好的单学习器次级随机森林
    isCategorical = zeros(size(X,2),1);
    treeNum=100;
    nodeNum=15;
    Stacking = TreeBagger(treeNum,A1,C(:,1),'Method','R','OOBPredictorImportance','On',...
        'CategoricalPredictors',find(isCategorical == 1),...
        'MinLeafSize',nodeNum);
    Ypredict5=predict(Stacking,B1);
    YpredictC=predict(Stacking,A1);
    %模型分析
    co=corr(ProbTest,Ypredict5,'type','Pearson');
    RMSE=sqrt(sum((ProbTest-Ypredict5).^2)/size(Ypredict5,1));
    R2 = corr(ProbTest,Ypredict5)^2;
    RMSEC=sqrt(sum((C(:,1)-YpredictC).^2)/size(YpredictC,1));
    R2C = corr(C(:,1),YpredictC)^2;
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
    writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
    d11=Ypredict5;
    d21=YpredictC;
    
    
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
    writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
    % d1=Ypredict5;
    % d2=YpredictC;
    d11=d11+Ypredict5;
    d21=d21+YpredictC;
    
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
    writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
    d11=d11+Ypredict5;
    d21=d21+YpredoctA;
    
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
    d11=d11+Ypredict5;
    d21=d21+YpredictC;
    
    A=A0(:,logical(Bestindividual));
    B=B0(:,logical(Bestindividual));
    figure();
    plot(nn,y);
    figure();
    plot(1:7,sum(count,1));
    hold on;
    scatter(1:7,Bestindividual.*sum(count,1));
    
    % A=A0(:,logical(Bestindividual));
    % B=B0(:,logical(Bestindividual));
    %次级随机森林
    isCategorical = zeros(size(X,2),1);
    treeNum=100;
    nodeNum=15;
    Stacking = TreeBagger(treeNum,A,C,'Method','R','OOBPredictorImportance','On',...
        'CategoricalPredictors',find(isCategorical == 1),...
        'MinLeafSize',nodeNum);
    Ypredict5=predict(Stacking,B);
    YpredictC=predict(Stacking,A);
    %模型分析
    co=corr(ProbTest,Ypredict5,'type','Pearson');
    RMSE=sqrt(sum((ProbTest-Ypredict5).^2)/size(Ypredict5,1));
    R2 = corr(ProbTest,Ypredict5)^2;
    RMSEC=sqrt(sum((C-YpredictC).^2)/size(YpredictC,1));
    R2C = corr(C,YpredictC)^2;
    co=num2str(roundn(co,-4));
    RMSE=num2str(roundn(RMSE,-4));
    R2=num2str(roundn(R2,-4));
    RMSEC=num2str(roundn(RMSEC,-4));
    R2C=num2str(roundn(R2C,-4));
    msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
        ['R2C：',R2C,'R2：',R2]},...
        '次级随机森林');
    d10=Ypredict5;
    d20=YpredictC;
    
    
    
    %Name:genmain05.m
    % clear
    % clf
    popsize=10;                                       %群体大小
    % chromlength=7;                                   %字符串长度（个体长度）
    count = zeros(iteration,chromlength);
    pc=0.6;                                           %交叉概率
    pm=0.001;                                         %变异概率
    pop=initpop(popsize,chromlength);                 %随机产生初始群体
    bestindividuals=zeros(iteration,chromlength);     %每轮最好的特征选择
    for ii=1:iteration                                     %20为迭代次数
        [objvalue]=calobjvalueSVM(pop,A0,C,B0,ProbTest);                      %计算目标函数
        fitvalue=objvalue;                   %计算群体中每个个体的适应度
        [newpop]=selection(pop,fitvalue);                 %复制
        [newpop]=crossover_multiv(newpop,pc);                       %交叉
        [newpop]=mutation(newpop,pc);                        %变异
        %     pop=newpop;
        [bestindividual,bestfit]=best(pop,fitvalue);%求出群体中适应值最大的个体及其适应值
        y(ii)=max(bestfit);
        n(ii)=ii;
        bestindividuals(ii,:)=bestindividual;
        %     pop5=bestindividual;
        %     x(ii)=decodechrom(pop5,1,chromlength)*10/1023;
        pop=newpop;
        count(ii,:)=sum(pop,1);
    end
    % fplot('10*sin(5*x)+7*cos(4*x)',[0 10])
    % plot();
    % hold on
    [M ,I]=max(y);
    Bestindividual=bestindividuals(I,:);
    figure();
    plot(nn,y);
    figure();
    plot(1:7,sum(count,1));
    hold on;
    scatter(1:7,Bestindividual.*sum(count,1));
    % hold off
    A=A0(:,logical(Bestindividual));
    B=B0(:,logical(Bestindividual));
    
    %次级模型SVM
    Stacking = fitrsvm(A,C,'KernelFunction','gaussian','OptimizeHyperparameters','auto',...
        'HyperparameterOptimizationOptions',struct('Optimizer','gridsearch','NumGridDivisions',15,...
        'ShowPlots',false,'MaxObjectiveEvaluations',60));
    Ypredict5=predict(Stacking,B);
    YpredictC=predict(Stacking,A);
    %模型分析
    co=corr(ProbTest,Ypredict5,'type','Pearson');
    RMSE=sqrt(sum((ProbTest-Ypredict5).^2)/size(Ypredict5,1));
    R2 = corr(ProbTest,Ypredict5)^2;
    RMSEC=sqrt(sum((C-YpredictC).^2)/size(YpredictC,1));
    R2C = corr(C,YpredictC)^2;
    co=num2str(roundn(co,-4));
    RMSE=num2str(roundn(RMSE,-4));
    R2=num2str(roundn(R2,-4));
    RMSEC=num2str(roundn(RMSEC,-4));
    R2C=num2str(roundn(R2C,-4));
    msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
        ['R2C：',R2C,'R2：',R2]},...
        '次级模型SVM');
    % d1=Ypredict5;
    % d2=YpredictC;
    d10=d10+Ypredict5;
    d20=d20+YpredictC;
    
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
    co=num2str(roundn(co,-4));
    RMSE=num2str(roundn(RMSE,-4));
    R2=num2str(roundn(R2,-4));
    RMSEC=num2str(roundn(RMSEC,-4));
    R2C=num2str(roundn(R2C,-4));
    msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
        ['R2C：',R2C,'R2：',R2]},...
        '次级模型SVM');
    % d1=Ypredict5;
    % d2=YpredictC;
    d11=d11+Ypredict5;
    d21=d21+YpredictC;
    
    %Name:genmain05.m
    % clear
    % clf
    popsize=10;                                       %群体大小
    % chromlength=5;                                   %字符串长度（个体长度）
    count = zeros(iteration,chromlength);
    pc=0.6;                                           %交叉概率
    pm=0.001;                                         %变异概率
    pop=initpop(popsize,chromlength);                 %随机产生初始群体
    bestindividuals=zeros(iteration,chromlength);     %每轮最好的特征选择
    for ii=1:iteration                                     %20为迭代次数
        [objvalue]=calobjvaluePLS(pop,A0,C,B0,ProbTest);                      %计算目标函数
        fitvalue=objvalue;                   %计算群体中每个个体的适应度
        [newpop]=selection(pop,fitvalue);                 %复制
        [newpop]=crossover_multiv(newpop,pc);                       %交叉
        [newpop]=mutation(newpop,pc);                        %变异
        %     pop=newpop;
        [bestindividual,bestfit]=best(pop,fitvalue);%求出群体中适应值最大的个体及其适应值
        y(ii)=max(bestfit);
        n(ii)=ii;
        bestindividuals(ii,:)=bestindividual;
        %     pop5=bestindividual;
        %     x(ii)=decodechrom(pop5,1,chromlength)*10/1023;
        pop=newpop;
        count(ii,:)=sum(pop,1);
    end
    % fplot('10*sin(5*x)+7*cos(4*x)',[0 10])
    % plot();
    % hold on
    [M ,I]=max(y);
    Bestindividual=bestindividuals(I,:);
    figure();
    plot(nn,y);
    figure();
    plot(1:7,sum(count,1));
    hold on;
    scatter(1:7,Bestindividual.*sum(count,1));
    % hold off
    A=A0(:,logical(Bestindividual));
    B=B0(:,logical(Bestindividual));
    
    %次级模型PLS
    Stacking = fitlm(A,C);
    Ypredict5=predict(Stacking,B);
    YpredoctA=predict(Stacking,A);
    %模型分析
    co=corr(ProbTest,Ypredict5,'type','Pearson');
    RMSE=sqrt(sum((ProbTest-Ypredict5).^2)/size(Ypredict5,1));
    R2 = corr(ProbTest,Ypredict5)^2;
    coP=corr(C,YpredoctA,'type','Pearson');
    RMSEP=sqrt(sum((C-YpredoctA).^2)/size(YpredoctA,1));
    R2P = corr(C,YpredoctA)^2;
    co=num2str(roundn(co,-4));
    RMSE=num2str(roundn(RMSE,-4));
    R2=num2str(roundn(R2,-4));
    RMSEP=num2str(roundn(RMSEP,-4));
    R2P=num2str(roundn(R2P,-4));
    msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
        ['R2C：',R2C,'R2：',R2]},...
        '次级模型PLS');
    d10=d10+Ypredict5;
    d20=d20+YpredoctA;
    
    %选三个效果好的单模型次级模型PLS
    Stacking = fitlm(A1,C);
    Ypredict5=predict(Stacking,B1);
    YpredoctA=predict(Stacking,A1);
    %模型分析
    co=corr(ProbTest,Ypredict5,'type','Pearson');
    RMSE=sqrt(sum((ProbTest-Ypredict5).^2)/size(Ypredict5,1));
    R2 = corr(ProbTest,Ypredict5)^2;
    coP=corr(C,YpredoctA,'type','Pearson');
    RMSEP=sqrt(sum((C-YpredoctA).^2)/size(YpredoctA,1));
    R2P = corr(C,YpredoctA)^2;
    co=num2str(roundn(co,-4));
    RMSE=num2str(roundn(RMSE,-4));
    R2=num2str(roundn(R2,-4));
    RMSEP=num2str(roundn(RMSEP,-4));
    R2P=num2str(roundn(R2P,-4));
    msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
        ['R2C：',R2C,'R2：',R2]},...
        '次级模型PLS');
    d11=d11+Ypredict5;
    d21=d21+YpredoctA;
    
    %Name:genmain05.m
    % clear
    % clf
    popsize=10;                                       %群体大小
    % chromlength=5;                                   %字符串长度（个体长度）
    count = zeros(iteration,chromlength);
    pc=0.6;                                           %交叉概率
    pm=0.001;                                         %变异概率
    pop=initpop(popsize,chromlength);                 %随机产生初始群体
    bestindividuals=zeros(iteration,chromlength);     %每轮最好的特征选择
    for ii=1:iteration                                      %20为迭代次数
        [objvalue]=calobjvalueGPR(pop,A0,C,B0,ProbTest);                      %计算目标函数
        fitvalue=objvalue;                   %计算群体中每个个体的适应度
        [newpop]=selection(pop,fitvalue);                 %复制
        [newpop]=crossover_multiv(newpop,pc);                       %交叉
        [newpop]=mutation(newpop,pc);                        %变异
        %     pop=newpop;
        [bestindividual,bestfit]=best(pop,fitvalue);%求出群体中适应值最大的个体及其适应值
        y(ii)=max(bestfit);
        n(ii)=ii;
        bestindividuals(ii,:)=bestindividual;
        %     pop5=bestindividual;
        %     x(ii)=decodechrom(pop5,1,chromlength)*10/1023;
        pop=newpop;
        count(ii,:)=sum(pop,1);
    end
    % fplot('10*sin(5*x)+7*cos(4*x)',[0 10])
    % plot();
    % hold on
    [M ,I]=max(y);
    Bestindividual=bestindividuals(I,:);
    figure();
    plot(nn,y);
    figure();
    plot(1:7,sum(count,1));
    hold on;
    scatter(1:7,Bestindividual.*sum(count,1));
    % hold off
    A=A0(:,logical(Bestindividual));
    B=B0(:,logical(Bestindividual));
    
    %次级模型高斯回归
    sigma0 = 0.2;
    kparams0 = [3.5, 6.2];
    stacking = fitrgp(A,C,'KernelFunction','squaredexponential',...
        'KernelParameters',kparams0,'Sigma',sigma0);
    Ypredict5=predict(stacking,B);
    YpredictC=predict(stacking,A);
    %模型分析
    co=corr(ProbTest,Ypredict5,'type','Pearson');
    RMSE=sqrt(sum((ProbTest-Ypredict5).^2)/size(Ypredict5,1));
    R2 = corr(ProbTest,Ypredict5)^2;
    RMSEC=sqrt(sum((C-YpredictC).^2)/size(YpredictC,1));
    R2C = corr(C,YpredictC)^2;
    co=num2str(roundn(co,-4));
    RMSE=num2str(roundn(RMSE,-4));
    R2=num2str(roundn(R2,-4));
    RMSEC=num2str(roundn(RMSEC,-4));
    R2C=num2str(roundn(R2C,-4));
    msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
        ['R2C：',R2C,'R2：',R2]},...
        '次级模型高斯回归');
    d10=d10+Ypredict5;
    d20=d20+YpredictC;
    
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
    co=num2str(roundn(co,-4));
    RMSE=num2str(roundn(RMSE,-4));
    R2=num2str(roundn(R2,-4));
    RMSEC=num2str(roundn(RMSEC,-4));
    R2C=num2str(roundn(R2C,-4));
    msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
        ['R2C：',R2C,'R2：',R2]},...
        '次级模型高斯回归');
    d11=d11+Ypredict5;
    d21=d21+YpredictC;
    
    d10=d10./4;
    d20=d20./4;
    RMSE=sqrt(sum((ProbTest-d10).^2)/size(d10,1));
    R2 = corr(ProbTest,d10)^2;
    RMSEC=sqrt(sum((C-d20).^2)/size(d20,1));
    R2C = corr(C,d20)^2;
    % co=num2str(roundn(co,-4));
    RMSE=num2str(roundn(RMSE,-4));
    R2=num2str(roundn(R2,-4));
    RMSEC=num2str(roundn(RMSEC,-4));
    R2C=num2str(roundn(R2C,-4));
    msgbox({['您的第',i,'运算的结果为:'],['RMSEC：',RMSEC,'RMSE：',RMSE],...
        ['R2C：',R2C,'R2：',R2]},...
        '简单平均法');
    
    %选三个效果好的基模型
    d11=d11./4;
    d21=d21./4;
    RMSE=sqrt(sum((ProbTest-d11).^2)/size(d11,1));
    R2 = corr(ProbTest,d11)^2;
    RMSEC=sqrt(sum((C-d21).^2)/size(d21,1));
    R2C = corr(C,d21)^2;
    % co=num2str(roundn(co,-4));
    RMSE=num2str(roundn(RMSE,-4));
    R2=num2str(roundn(R2,-4));
    RMSEC=num2str(roundn(RMSEC,-4));
    R2C=num2str(roundn(R2C,-4));
    msgbox({['您的第',i,'运算的结果为:'],['RMSEC：',RMSEC,'RMSE：',RMSE],...
        ['R2C：',R2C,'R2：',R2]},...
        '简单平均法');
    
    % if (str2num(R2)>0.7 && str2num(R2C)>0.7)
    %     choos=1;
    % else
    %     i=i-1;
    % end
    % if(choos==1)
    %     if(i==1)
    %         D1=d10;
    %         D2=d20;
    %     else
    %         D1= D1+d10;
    %         D2=D2+d20;
    %     end
    %     choos=0;
    % end
    if(i==1)
        D1=d10;
        D2=d20;
    else
        D1= D1+d10;
        D2=D2+d20;
    end
    
    if(i==1)
        D11=d11;
        D21=d21;
    else
        D11= D11+d11;
        D21=D21+d21;
    end
end
D1=D1./i;
D2=D2./i;
RMSE=sqrt(sum((ProbTest-D1).^2)/size(D1,1));
R2 = corr(ProbTest,D1)^2;
RMSEC=sqrt(sum((C-D2).^2)/size(D2,1));
R2C = corr(C,D2)^2;
% co=num2str(roundn(co,-4));
RMSE=num2str(roundn(RMSE,-4));
R2=num2str(roundn(R2,-4));
RMSEC=num2str(roundn(RMSEC,-4));
R2C=num2str(roundn(R2C,-4));
msgbox({['您的最终运算的结果为:'],['RMSEC：',RMSEC,'RMSE：',RMSE],...
    ['R2C：',R2C,'R2：',R2]},...
    '最终结果');

D11=D11./i;
D21=D21./i;
RMSE=sqrt(sum((ProbTest-D11).^2)/size(D11,1));
R2 = corr(ProbTest,D11)^2;
RMSEC=sqrt(sum((C-D21).^2)/size(D21,1));
R2C = corr(C,D21)^2;
% co=num2str(roundn(co,-4));
RMSE=num2str(roundn(RMSE,-4));
R2=num2str(roundn(R2,-4));
RMSEC=num2str(roundn(RMSEC,-4));
R2C=num2str(roundn(R2C,-4));
msgbox({['您的最终运算的结果为:'],['RMSEC：',RMSEC,'RMSE：',RMSE],...
    ['R2C：',R2C,'R2：',R2]},...
    '最终结果');

