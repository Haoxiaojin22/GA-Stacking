function [rmsecv] = pls_crossvalind(train_data,train_label,k,ncomp)

%% K-fold crossvalind
[m_size n_size]=size(train_data); 
indices = crossvalind('Kfold',m_size,k); 
rmse=[];
for i=1:k
    test=(indices==i); 
    train=~test;
    data_train=train_data(train,:);
    label_train=train_label(train,:); 
    data_test=train_data(test,:); 
    label_test=train_label(test,:);
    %% PLS
    ab_train=[data_train label_train];  
    mu=mean(ab_train);sig=std(ab_train); 
    rr=corrcoef(ab_train);   
    ab=zscore(ab_train); 
    a=ab(:,[1:n_size]);b=ab(:,[n_size+1:end]);  
    [XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] =plsregress(a,b,ncomp);
    n=size(a,2); mm=size(b,2);
    beta3(1,:)=mu(n+1:end)-mu(1:n)./sig(1:n)*BETA([2:end],:).*sig(n+1:end); 
    beta3([2:n+1],:)=(1./sig(1:n))'*sig(n+1:end).*BETA([2:end],:); 
    ab_test=[data_test label_test];
    a1=ab_test(:,[1:n_size]);b1=ab_test(:,[n_size+1:end]); 
    yhat_test=repmat(beta3(1,:),[size(a1,1),1])+ab_test(:,[1:n])*beta3([2:end],:); 
    rmse=[rmse sqrt(sum((yhat_test-label_test).^2)/m_size*k)]; 
end
   rmsecv=sum(rmse)/k;