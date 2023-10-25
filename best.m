%Name: best.m
%求出群体中适应值最大的值
function [bestindividual,bestfit,bestfitNum]=best(pop,fitvalue)
[px,py]=size(pop);
bestindividual=pop(1,:);
bestfit=fitvalue(1);
bestfitNum=1;
for i=2:px
      if fitvalue(i)>bestfit
          bestindividual=pop(i,:);
          bestfit=fitvalue(i);
          bestfitNum=i;
      end
end
