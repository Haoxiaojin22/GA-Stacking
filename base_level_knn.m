function [A,B,r2A,rmseA]=base_level_knn(XTrain,yTrain,XTest,yTest,c)
% %一阶导数适合

XTrain=diff(XTrain,1,2);
XTest=diff(XTest,1,2);

AA=cell(c.NumTestSets,1);%存储每一折的交叉验证结果
A=zeros(length(yTrain),1);
B=zeros(length(yTest),1);%存储5次XTest的预测结果的平均值
k=24; %通过实验，发现近邻数为24较好
for ModelIndex=1:c.NumTestSets
    trainidx5=training(c,ModelIndex);
    testidx5=test(c,ModelIndex);
    XTrain5=XTrain(trainidx5,:);
    yTrain5=yTrain(trainidx5,:);
    XTest5 =XTrain(testidx5,:);
        yTest5 =yTrain(testidx5,:);

    Mdl = KDTreeSearcher(XTrain5);

    idx = knnsearch(Mdl,XTest5,'k',k);
    a=zeros(c.TestSize(ModelIndex),1);
    for j = 1:size(idx,1)
        a(j) = mean(yTrain5(idx(j,:)));
    end
    A(testidx5)=a;
   [r2(ModelIndex),rmse(ModelIndex)] = rsquare(yTest5,a);
    idx = knnsearch(Mdl,XTest(:,:),'k',k);
    b=zeros(length(yTest),1);
    for j = 1:size(idx,1)
        b(j) = mean(yTrain5(idx(j,:)));
    end
    B=B+b;
end
r2A=mean(r2);
rmseA=mean(rmse);
B=B./c.NumTestSets;
% % aa=zeros(size(XTest5,1),c.NumTestSets);
% % for i=1:c.NumTestSets
% %     aa(:,i)=[AA{i,:}];
% % end
% % A=zeros(length(yTrain),1);
% % for i=1:c.NumTestSets
% %    A(testidx5)=aa(i);
% % end
% aa=[];
% for i=1:c.NumTestSets
%     aa=cat(1,aa,[AA{i,:}]);
% end
% A=zeros(length(yTrain),1);
% L=0;
% for i=1:c.NumTestSets
%     testidx5=test(c,i);
%     temp=sum(testidx5);
% %         try
%     A(testidx5)=aa(L+1:L+temp);
%     L=L+temp;
% %          catch ME
% %         if (strcmp(ME.identifier,'MATLAB:catenate:dimensionMismatch'))
% %           msg = ['Dimension mismatch occurred: First argument has ', ...
% %                 num2str(L),' columns while second has ', ...
% %                 num2str(size(XTest5,1)),' columns.'];
% %             causeException = MException('MATLAB:myCode:dimensions',msg);
% %             ME = addCause(ME,causeException);
% %         end
% %         rethrow(ME)
% %         end
% end

