%Name: MetaSVM.m
function [objvalue]=MetaSVM(SpecTrain, SpecTest, ProbTrain, ProbTest)
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
msgbox({'您运算的结果为:',['RMSEC：',RMSEC,'RMSE：',RMSE],...
    ['R2C：',R2C,'R2：',R2]},...
    'GS');
% % 将运算结果写入文件
% GPRname = {'RMSEC';'R²C';'RMSEP';'R²P'};
% % result=[RMSEP;R2P;RMSE;R2];
% Tabl=table(GPRname,result);
% writetable(Tabl,'m.xls','WriteVariableNames',true,'WriteMode',"append");
objvalue=result;

