%Name: calobjvalue.m
%实现目标函数R2的计算
function [objvalue,Metaresult]=calobjvalue3(pop,spec,prob,testSpec,testProb)
% temp1=decodechrom(pop,1,10);    %将pop每行转化成十进制数
% x=temp1*10/1023;                %将二值域 中的数转化为变量域 的数
% objvalue=10*sin(5*x)+7*cos(4*x);%计算目标函数值
[a b] = size(pop);
objvalue=zeros(a,1);
t=zeros(1,b-2);
for i=1:a
    result=zeros(4,a);
    if pop(i,1:7)==t
        pop(i,1:7)=ones(1,b-2);
    end
    ind=logical(pop(i,1:7));
    newSpec=spec(:,ind);
    newtestSpec=testSpec(:,ind);
    num=num2str(pop(i,8:9));%次级模型编号
    temp1=bin2dec(num) %将编号转化成十进制数
    if temp1==0
        sigma0 = 0.2;
        kparams0 = [3.5, 6.2];
        stacking = fitrgp(newSpec,prob,'KernelFunction','squaredexponential',...
            'KernelParameters',kparams0,'Sigma',sigma0);
        Ypredict5=predict(stacking,newtestSpec);
        YpredictC=predict(stacking,newSpec);
        %模型分析
        co=corr(testProb,Ypredict5,'type','Pearson');
        RMSE=sqrt(sum((testProb-Ypredict5).^2)/size(Ypredict5,1));
        R2 = corr(testProb,Ypredict5)^2;
        RMSEC=sqrt(sum((prob-YpredictC).^2)/size(YpredictC,1));
        R2C = corr(prob,YpredictC)^2;
        objvalue(i,1)=R2;
%         temp=sqrt(R2*R2C);
%         objvalue(i,1)=R2*temp;
        %         objvalue(i,1)=(R2*R2+R2C*R2C) /(1-(R2*R2)/(R2C*R2C));%修改批判标准
%          objvalue(i,1)=(R2*R2) /(1-(R2*R2)/(R2C*R2C));%修改批判标准
    end
    if temp1==1
        Stacking = fitrsvm(newSpec,prob,'KernelFunction','gaussian','OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('Optimizer','gridsearch','NumGridDivisions',15,...
    'ShowPlots',false,'MaxObjectiveEvaluations',60));
    Ypredict5=predict(Stacking,newtestSpec);
    YpredictC=predict(Stacking,newSpec);
    %模型分析
    co=corr(testProb,Ypredict5,'type','Pearson');
    RMSE=sqrt(sum((testProb-Ypredict5).^2)/size(Ypredict5,1));
    R2 = corr(testProb,Ypredict5)^2;
    RMSEC=sqrt(sum((prob-YpredictC).^2)/size(YpredictC,1));
    R2C = corr(prob,YpredictC)^2;
            objvalue(i,1)=R2;
%         temp=sqrt(R2*R2C);
%         objvalue(i,1)=R2*temp;
 %         objvalue(i,1)=(R2*R2+R2C*R2C) /(1-(R2*R2)/(R2C*R2C));%修改批判标准
%          objvalue(i,1)=(R2*R2) /(1-(R2*R2)/(R2C*R2C));%修改批判标准
    end
    if temp1==2
         Stacking = fitlm(newSpec,prob);
    Ypredict5=predict(Stacking,newtestSpec);
    YpredoctA=predict(Stacking,newSpec);
    %模型分析
    co=corr(testProb,Ypredict5,'type','Pearson');
    RMSE=sqrt(sum((testProb-Ypredict5).^2)/size(Ypredict5,1));
    R2 = corr(testProb,Ypredict5)^2;
    coP=corr(prob,YpredoctA,'type','Pearson');
    RMSEC=sqrt(sum((prob-YpredoctA).^2)/size(YpredoctA,1));
    R2C = corr(prob,YpredoctA)^2;
            objvalue(i,1)=R2;
%         temp=sqrt(R2*R2C);
%         objvalue(i,1)=R2*temp;
%         objvalue(i,1)=(R2*R2+R2C*R2C) /(1-(R2*R2)/(R2C*R2C));%修改批判标准
%          objvalue(i,1)=(R2*R2) /(1-(R2*R2)/(R2C*R2C));%修改批判标准
    end
    if temp1==3
    isCategorical = zeros(size(spec,2),1);
    treeNum=100;
    nodeNum=15;
    Stacking = TreeBagger(treeNum,newSpec,prob,'Method','R','OOBPredictorImportance','On',...
    'CategoricalPredictors',find(isCategorical == 1),...
    'MinLeafSize',nodeNum);
    Ypredict5=predict(Stacking,newtestSpec);
    YpredictC=predict(Stacking,newSpec);
    %模型分析
    co=corr(testProb,Ypredict5,'type','Pearson');
    RMSE=sqrt(sum((testProb-Ypredict5).^2)/size(Ypredict5,1));
    R2 = corr(testProb,Ypredict5)^2;
    RMSEC=sqrt(sum((prob-YpredictC).^2)/size(YpredictC,1));
    R2C = corr(prob,YpredictC)^2;
         objvalue(i,1)=R2;
%         temp=sqrt(R2*R2C);
%         objvalue(i,1)=R2*temp;
     %         objvalue(i,1)=(R2*R2+R2C*R2C) /(1-(R2*R2)/(R2C*R2C));%修改批判标准
%          objvalue(i,1)=(R2*R2) /(1-(R2*R2)/(R2C*R2C));%修改批判标准
    end
   Metaresult(:,i) = [RMSE,R2,RMSEC,R2C];
end