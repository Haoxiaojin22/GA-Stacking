function A=SNV(X)
% A=SNV(X);
% plot(1:size(X,2),X);
% figure,plot(1:size(A,2),A);

mu=mean(X,2);
sigma=std(X,1,2);
siz=size(X);
MU=repmat(mu,1,siz(2));
SIGMA=repmat(sigma,1,siz(2));
A=(X-MU)./SIGMA;
