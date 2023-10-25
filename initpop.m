%Name: initpop.m
%初始化
function pop=initpop(popsize,chromlength)     
pop=round(rand(popsize,chromlength)); 
% t=zeros(1,chromlength);
% for i= 1:popsize
%     if pop(i,:)==t
%         pop(i,:)=ones(1,chromlength);
%     end
% end
% rand随机产生每个单元为 {0,1} 行数为popsize，列数为chromlength的矩阵，
% roud对矩阵的每个单元进行圆整。这样产生的初始种群。
