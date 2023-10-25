%Name: selection.m
%选择复制
function [newpop]=selection(pop,fitvalue)
totalfit=sum(fitvalue);%求适应值之和
fitvalue=fitvalue/totalfit;%单个个体被选择的概率
fitvalue=cumsum(fitvalue); %累积概率，如 fitvalue=[1 2 3 4]，则 cumsum(fitvalue)=[1 3 6 10]
[px,py]=size(pop);
ms=sort(rand(px,1)); %从小到大排列，将"rand(px,1)"产生的一列随机数变成轮盘赌形式的表示方法，由小到大排列
fitin=1;  %fivalue是一向量，fitin代表向量中元素位，即fitvalue(fitin)代表第fitin个个体的单个个体被选择的概率
newin=1;  %同理
newpop=pop;
while (newin<=px)&&(fitin<=px)
    if(ms(newin))<fitvalue(fitin)         %ms(newin)表示的是ms列向量中第"newin"位数值，同理fitvalue(fitin)
    newpop(newin,:)=pop(fitin,:);      %赋值 ,即将旧种群中 的第fitin个个体保留到下一代(newpop)
    newin=newin+1;
    else
    fitin=fitin+1;
    end
end
