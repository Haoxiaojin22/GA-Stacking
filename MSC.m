function [MSC1,MSC2]=MSC(X,Y)
% 可以对一个光谱矩阵进行MSC变换，也可以同时对建模集和测试集的光谱矩阵同时进行MSC变换
% X 建模集光谱矩阵，必须参数
% Y 测试集光谱矩阵，可选参数

% 如果只有一个输入参数，则MSC1有效，如果有两个输入参数，MSC2也有效。

% Y=X(150:end,:);
% X=X(1:149,:);
% [A,B]=MSC(X,Y);
% n=size(A,2);
% 
% figure,plot(1:n,X);
% figure,plot(1:n,Y);
% 
% figure,plot(1:n,A);
% figure,plot(1:n,B);


if(nargin<1)
    msg = 'Provide at least one parameter.';
    error(msg)
elseif(nargin==1)
    XY=X;
elseif(nargin==2)
    if(size(X,2)~=size(Y,2))
        error('两个矩阵的第二维必须相等');
    end
    XY=[X;Y];
end

n=size(XY,2);
x_sta=mean(XY);
A=[ones(1,n)' x_sta'];
%对A矩阵SVD分解
[u, s, v]=svd(A);
%SVD求解最小二乘问题
A_plus=v*pinv(s)*u';
C=A_plus*XY';% C即为所要求解的系数矩阵

% %对原始光谱进行处理，减去偏移量与系数
P=C(1,:);
bais=P'*ones(1,n);
MSC=XY-bais;
P=C(2,:);
coeff=P'*ones(1,n);
MSC=MSC./coeff;

siz=size(X,1);

if(nargin==1)
    MSC1=MSC;
    MSC2=[];
elseif(nargin==2)
    MSC1=MSC(1:siz,:);
    MSC2=MSC(siz+1:end,:);
end