%Name: MetaRF.m
function [objvalue] =MetaRF(SpecTrain, SpecTest, ProbTrain, ProbTest)
isCategorical = zeros(size(SpecTrain,2),1);% Categorical variable flag
treeNum=80;
nodeNum=32;
 b = TreeBagger(treeNum,SpecTrain,ProbTrain,'Method','R','OOBPredictorImportance','On',...
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
msgbox({'您运算的结果为:',['RMSEP：',RMSEP,'RMSE：',RMSE],...
    ['R2P：',R2P,'R2：',R2]},...
    '随机森林');
% % 将运算结果写入文件
% RFname = {'RMSEC';'R²C';'RMSEP';'R²P'};
% % result=[RMSEP;R2P;RMSE;R2];
% Tabl=table(RFname,result);
% writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
objvalue=result;

