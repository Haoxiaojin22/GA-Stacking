%% a litte clean work
%close all;
% clear;
% clc;
format compact;
%% ��������
% load data
% load concentration_3
%load nirs_data; %�������ݣ�������Ϊnirs_data,����Ϊ120*1557
% % ��������ѡ��567������
% data=csvread('D:\HXJ\�о���\����\���׷�ѿ\���׷�ѿ������\���׷�ѿ������\laohuaYM_noAvg.csv');
% data=data(72:end,:);
% nirs_data=data(:,1:end-1);
% content_data=data(:,end);
% % ˮ����������
 data=load('corn.mat');
nirs_data=data.data_ys(:,1:end-1);
content_data=data.data_ys(:,end);
% %����ʮ����ֵ
% nirs_data=data(:,1:end-1);
%load label;    %����ֵ,������label����λcm-1������Ϊ1*1557
% label=NIR_VarLabels0;
%lignocellulose_content_data=xlsread('lignocellulose_content_data.xls');  %��ά�ء�����ά�ء�ľ���غ���������Ϊ120*3
%content_data=lignocellulose_content_data;

%% ����Ԥ����
% %SGƽ����Savitzky Golay filter��ƽ���˲�
% nirs_sg_data = savgol(nirs_data,7,3,0);
% nirs_data=nirs_sg_data;
% 
% % %MSC��Ԫɢ��У����Multiplicative scatter correction��
%   nirs_msc_data=msc(nirs_data,nirs_data);
%   nirs_data=nirs_msc_data;
% % %SNV��׼����任(Standard normal variate)
%   nirs_snv_data=snv(nirs_data);
%   nirs_data=nirs_snv_data;
% % %���������ο�diff(x,n)����,�����������һ��
% %    nirs_ds_data=diff(nirs_data',1);
% %    nirs_data=nirs_ds_data';
% %   nirs_ds_data=diff(nirs_data',1);
% %   nirs_data=nirs_ds_data';
train_data=nirs_data;
train_label=content_data;
%% 
[m_size n_size]=size(train_data); %dataΪ�������ϡ�ÿһ��Ϊһ���۲�����
k=1000;%ִ�д���
% residual_C=zeros(k,m_size);%�в����
% residual_H=zeros(k,m_size);%�в����
residual_L=zeros(k,m_size);%�в����
steps=k;
hwait=waitbar(0,'��ȴ�>>>>>>>>');%������
step=steps/100;
for gen=1:k
    	%% ������
    if steps-gen<=1
        waitbar(gen/steps,hwait,'�������');
        pause(0.05);
    else
        PerStr=fix(gen/step);
        str=['���ڽ�����',num2str(PerStr),'%'];
        waitbar(gen/steps,hwait,str);
        pause(0.05);
    end
  
    
    [train, test] = crossvalind('HoldOut',m_size,1/3); %ѡ������80%�ĸ�����Ϊѵ����������train��test��m_size�������У���ӦλΪ1��ʾѡ�С�
    data_train=train_data(train,:); %����У�����������ݵ�ѡȡ
    label_train=train_label(train,:); %����У����������ǩ��ѡȡ
    data_test=train_data(test,:); %ѡȡ���Լ�����������
    label_test=train_label(test,:);%ѡȡ���Լ��ı�ǩ
    %% PLSģ��
    ab_train=[data_train label_train];  %У�����������ݺͱ�ǩ
    mu=mean(ab_train);sig=std(ab_train); %���ֵ�ͱ�׼��
    rr=corrcoef(ab_train);   %�����ϵ������
    ab=zscore(ab_train); %���ݱ�׼��
    a=ab(:,[1:n_size]);b=ab(:,[n_size+1:end]);  %�����׼������Ա��������������
    ncomp=NCOMP_BestNumber_Search_RMSECV(data_train,label_train(:,1),5,15); %������ɷָ���_��ά��
%      ncomp=NCOMP_BestNumber_Search_RMSECV(data_train,label_train(:,2),5,15); %������ɷָ���_����ά��
%      ncomp=NCOMP_BestNumber_Search_RMSECV(data_train,label_train(:,3),5,15); %������ɷָ���_ľ����
    [XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] =plsregress(a,b,ncomp);
    n=size(a,2); mm=size(b,2);%n���Ա����ĸ���,m��������ĸ���
    beta3(1,:)=mu(n+1:end)-mu(1:n)./sig(1:n)*BETA([2:end],:).*sig(n+1:end); %ԭʼ���ݻع鷽�̵ĳ�����
    beta3([2:n+1],:)=(1./sig(1:n))'*sig(n+1:end).*BETA([2:end],:); %����ԭʼ����x1,...,xn��ϵ����ÿһ����һ���ع鷽��R
    ab_test=[data_test label_test];%���Լ��������ݺͱ�ǩ
    a1=ab_test(:,[1:n_size]);b1=ab_test(:,[n_size+1:end]); 
    yhat_test=repmat(beta3(1,:),[size(a1,1),1])+ab_test(:,[1:n])*beta3([2:end],:); %��label_test��Ԥ��ֵ
    temp_residual= abs(yhat_test-label_test); %��в�
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
close(hwait);%�رս�����
%% ����в�ľ�ֵ��SD
for i=1:m_size
%     R_C=residual_C(:,i);
%     R_C(R_C==0)=[];%R_C(R_C==0)=[]��ʾȥ��R_C��Ϊ0��Ԫ��
%     Mean_Residual_C(i)=mean(R_C);%�в�ƽ��ֵ
%     Var_Residual_C(i)=var(R_C);%�в��
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
% % ���ƶ�άɢ��ͼ��������
% plot(Mean_Residual_C,Var_Residual_C,'r.');%����ɢ��ͼ
% hold on
% for i=1:max(size(Mean_Residual_C))
%     c = num2str(i);%����ת�ַ���
%     text(Mean_Residual_C(i),Var_Residual_C(i),c);%��ͼ����ʾ����
% end
% xlabel('��ֵ');
% ylabel('����');
% hold off
% 
% figure(2)
% ���ƶ�άɢ��ͼ��������
% plot(Mean_Residual_H,Var_Residual_H,'r.');%����ɢ��ͼ
% hold on
% for i=1:max(size(Mean_Residual_H))
%     c = num2str(i);%����ת�ַ���
%     text(Mean_Residual_H(i),Var_Residual_H(i),c);%��ͼ����ʾ����
% end
% xlabel('��ֵ');
% ylabel('����');
% hold off
figure(3)
%���ƶ�άɢ��ͼ��������
plot(Mean_Residual_L,Var_Residual_L,'r.');%����ɢ��ͼ
hold on
for i=1:max(size(Mean_Residual_L))
    c = num2str(i);%����ת�ַ���
    text(Mean_Residual_L(i),Var_Residual_L(i),c);%��ͼ����ʾ����
end

xlabel('��ֵ');
ylabel('����');
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
