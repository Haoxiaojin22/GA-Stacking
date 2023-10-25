%% a litte clean work
%close all;
% clear;
% clc;
format compact;
%% 导入数据
% load data
% load concentration_3
%load nirs_data; %光谱数据，变量名为nirs_data,数据为120*1557
% % 玉米数据选择567号样本
% data=csvread('D:\HXJ\研究生\数据\玉米发芽\玉米发芽率数据\玉米发芽率数据\laohuaYM_noAvg.csv');
% data=data(72:end,:);
% nirs_data=data(:,1:end-1);
% content_data=data(:,end);
% % 水稻数据样本
 data=load('corn.mat');
nirs_data=data.data_ys(:,1:end-1);
content_data=data.data_ys(:,end);
% %柴油十六烷值
% nirs_data=data(:,1:end-1);
%load label;    %波长值,变量名label，单位cm-1，数据为1*1557
% label=NIR_VarLabels0;
%lignocellulose_content_data=xlsread('lignocellulose_content_data.xls');  %纤维素、半纤维素、木质素含量，数据为120*3
%content_data=lignocellulose_content_data;

%% 光谱预处理
% %SG平滑：Savitzky Golay filter，平滑滤波
% nirs_sg_data = savgol(nirs_data,7,3,0);
% nirs_data=nirs_sg_data;
% 
% % %MSC多元散射校正（Multiplicative scatter correction）
%   nirs_msc_data=msc(nirs_data,nirs_data);
%   nirs_data=nirs_msc_data;
% % %SNV标准正则变换(Standard normal variate)
%   nirs_snv_data=snv(nirs_data);
%   nirs_data=nirs_snv_data;
% % %导数处理，参考diff(x,n)函数,导数计算后少一列
% %    nirs_ds_data=diff(nirs_data',1);
% %    nirs_data=nirs_ds_data';
% %   nirs_ds_data=diff(nirs_data',1);
% %   nirs_data=nirs_ds_data';
train_data=nirs_data;
train_label=content_data;
%% 
[m_size n_size]=size(train_data); %data为样本集合。每一行为一个观察样本
k=1000;%执行次数
% residual_C=zeros(k,m_size);%残差矩阵
% residual_H=zeros(k,m_size);%残差矩阵
residual_L=zeros(k,m_size);%残差矩阵
steps=k;
hwait=waitbar(0,'请等待>>>>>>>>');%进度条
step=steps/100;
for gen=1:k
    	%% 进度条
    if steps-gen<=1
        waitbar(gen/steps,hwait,'即将完成');
        pause(0.05);
    else
        PerStr=fix(gen/step);
        str=['正在进行中',num2str(PerStr),'%'];
        waitbar(gen/steps,hwait,str);
        pause(0.05);
    end
  
    
    [train, test] = crossvalind('HoldOut',m_size,1/3); %选出整体80%的个体作为训练集，其中train和test是m_size长的序列，对应位为1表示选中。
    data_train=train_data(train,:); %用于校正集样本数据的选取
    label_train=train_label(train,:); %用于校正集样本标签的选取
    data_test=train_data(test,:); %选取测试集的样本数据
    label_test=train_label(test,:);%选取测试集的标签
    %% PLS模型
    ab_train=[data_train label_train];  %校正集样本数据和标签
    mu=mean(ab_train);sig=std(ab_train); %求均值和标准差
    rr=corrcoef(ab_train);   %求相关系数矩阵
    ab=zscore(ab_train); %数据标准化
    a=ab(:,[1:n_size]);b=ab(:,[n_size+1:end]);  %提出标准化后的自变量和因变量数据
    ncomp=NCOMP_BestNumber_Search_RMSECV(data_train,label_train(:,1),5,15); %最佳主成分个数_纤维素
%      ncomp=NCOMP_BestNumber_Search_RMSECV(data_train,label_train(:,2),5,15); %最佳主成分个数_半纤维素
%      ncomp=NCOMP_BestNumber_Search_RMSECV(data_train,label_train(:,3),5,15); %最佳主成分个数_木质素
    [XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] =plsregress(a,b,ncomp);
    n=size(a,2); mm=size(b,2);%n是自变量的个数,m是因变量的个数
    beta3(1,:)=mu(n+1:end)-mu(1:n)./sig(1:n)*BETA([2:end],:).*sig(n+1:end); %原始数据回归方程的常数项
    beta3([2:n+1],:)=(1./sig(1:n))'*sig(n+1:end).*BETA([2:end],:); %计算原始变量x1,...,xn的系数，每一列是一个回归方程R
    ab_test=[data_test label_test];%测试集样本数据和标签
    a1=ab_test(:,[1:n_size]);b1=ab_test(:,[n_size+1:end]); 
    yhat_test=repmat(beta3(1,:),[size(a1,1),1])+ab_test(:,[1:n])*beta3([2:end],:); %求label_test的预测值
    temp_residual= abs(yhat_test-label_test); %求残差
    [test_m,test_n]=size(temp_residual);
    num=0;
    j=1;
    for j=1:m_size
        if test(j)==1
            num=num+1;
%             residual_C(gen,j)=temp_residual(num,1);
%             residual_H(gen,j)=temp_residual(num,1);
            residual_L(gen,j)=temp_residual(num,1);
        end
    end
end
close(hwait);%关闭进度条
%% 计算残差的均值和SD
for i=1:m_size
%     R_C=residual_C(:,i);
%     R_C(R_C==0)=[];%R_C(R_C==0)=[]表示去掉R_C中为0的元素
%     Mean_Residual_C(i)=mean(R_C);%残差平均值
%     Var_Residual_C(i)=var(R_C);%残差方差
%     R_H=residual_H(:,i);
%     R_H(R_H==0)=[];
%     Mean_Residual_H(i)=mean(R_H);
%     Var_Residual_H(i)=var(R_H);
    R_L=residual_L(:,i);
    R_L(R_L==0)=[];
    Mean_Residual_L(i)=mean(R_L);
    Var_Residual_L(i)=var(R_L);
    Number(i)=i;
end

% figure(1)
% % 绘制二维散点图并标出序号
% plot(Mean_Residual_C,Var_Residual_C,'r.');%绘制散点图
% hold on
% for i=1:max(size(Mean_Residual_C))
%     c = num2str(i);%数字转字符串
%     text(Mean_Residual_C(i),Var_Residual_C(i),c);%在图上显示文字
% end
% xlabel('均值');
% ylabel('方差');
% hold off
% 
% figure(2)
% 绘制二维散点图并标出序号
% plot(Mean_Residual_H,Var_Residual_H,'r.');%绘制散点图
% hold on
% for i=1:max(size(Mean_Residual_H))
%     c = num2str(i);%数字转字符串
%     text(Mean_Residual_H(i),Var_Residual_H(i),c);%在图上显示文字
% end
% xlabel('均值');
% ylabel('方差');
% hold off
figure(3)
%绘制二维散点图并标出序号
plot(Mean_Residual_L,Var_Residual_L,'r.');%绘制散点图
hold on
for i=1:max(size(Mean_Residual_L))
    c = num2str(i);%数字转字符串
    text(Mean_Residual_L(i),Var_Residual_L(i),c);%在图上显示文字
end

xlabel('均值');
ylabel('方差');
% xlabel('mean');
% ylabel('variance');
maxm=max(Mean_Residual_L);
maxv=max(Var_Residual_L);
plot([0,0.70*maxm],[0.70*maxv,0.70*maxv],'b' ,'linewidth',1) 
plot([0.70*maxm,0.70*maxm],[0,0.70*maxv],'b','linewidth',1)
hold off

M=mean(Mean_Residual_L);
Median=median(Mean_Residual_L);
Mv=mean(Var_Residual_L);
outliers=ones(m_size,1);
maxm=max(Mean_Residual_L);
maxv=max(Var_Residual_L);
% % find(Mean_Residual_L>M);
outliers(find(Mean_Residual_L>0.5),:)=0;
outliers(find(Var_Residual_L>0.012),:)=0;
% outliers(find(Mean_Residual_L>0.70*maxm),:)=0;
% outliers(find(Var_Residual_L>0.70*maxv),:)=0;
ind=logical(outliers);
data=data.fayalv(ind,:);
