%Name: crossover.m
%交叉
function [newpop]=crossover_multiv(pop,pc)
% global Numv
[px,py]=size(pop);
% m=py/Numv;
% for j=1:Numv
%     pop1=ones(px,m);
newpop=ones(px,py);
%     pop2=pop(:,m*(j-1)+1:m*j);      %取出相应变量对应的二进制编码段
    for i=1:2:px-1
       if(rand<pc)
%           cpoint=round(rand*(m-1));   %cpoint为交叉点
           cpoint=round(rand*py);   %cpoint为交叉点
%           pop1(i,:)=[pop2(i,1:cpoint) pop2(i+1,cpoint+1:m)];
%           pop1(i+1,:)=[pop2(i+1,1:cpoint) pop2(i,cpoint+1:m)];
            newpop(i,:)=[pop(i,1:cpoint) pop(i+1,cpoint+1:py)];
            newpop(i+1,:)=[pop(i+1,1:cpoint) pop(i,cpoint+1:py)];
       else
%           pop1(i,:)=pop2(i,1:m);
%           pop1(i+1,:)=pop2(i+1,1:m);
            newpop(i,:)=pop(i,:);
            newpop(i+1,:)=pop(i+1,:);
       end
    end
%    newpop(:,m*(j-1)+1:m*j)=pop1;               %将交叉后的一个参数的编码放入新种群中
% end
