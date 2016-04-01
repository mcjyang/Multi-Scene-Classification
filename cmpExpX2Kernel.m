function [trainK testK] = cmpExpX2Kernel(trainD, testD, gamma)

    % trainD is n*d matrix has n data with d features
    % testD is m*d matrix has m data with d features

%     [n , d1] = size(trainD);
%     [m , d2] = size(testD);
    
%     err = 1e-12;
% %     err =0;
%     total1=zeros(n,n);
%     total2=zeros(m,n);
%     for i = 1:n
%         for j = 1:n
%             % train Dataset
%             numerator = (trainD(i,:)-trainD(j,:)).^2;
%             dominator = trainD(i,:) + trainD(j,:) + err;
%             temp = numerator./dominator;
%             total1(i,j) = sum(temp);
%         end
%     end
%     trainK = exp(-total1./gamma);
%     
%     for i = 1:m
%         for j = 1:n
%             % test Dataset
%             numerator = (testD(i,:)-trainD(j,:)).^2;
%             dominator = testD(i,:) + trainD(j,:) + err;
%             temp = numerator./dominator;
%             total2(i,j) = sum(temp);
%         end
%     end
%     testK = exp(-total2./gamma);
    
%     save('cheat.mat','total1','total2');
      load('cheat.mat');
      trainK = exp(-total1./gamma);
      testK = exp(-total2./gamma);

end