%Name: MetaPLS.m
function [objvalue] =MetaPLS(SpecTrain, SpecTest, ProbTrain, ProbTest)
[XL,yl,XS,YS,beta] = plsregress(SpecTrain,ProbTrain,4);
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
msgbox({'您运算的结果为:',['RMSEP：',RMSEP,'RMSE：',RMSE],...
    ['R2P：',R2P,'R2：',R2]},...
    'PLS');
% % 将运算结果写入文件
% PLSname = {'RMSEC';'R²C';'RMSEP';'R²P'};
% % result=[RMSEP;R2P;RMSE;R2];
% Tabl=table(PLSname,result);
% writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
objvalue=result;

