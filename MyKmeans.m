function [result, center, totalWithin] = MyKmeans(data, k, isRandom)
    
    % Input:
    % data is n * d matrix, where n is number of data with d dimension
    % k is the number of center
    % isRandom is variable that decide whether centers are randomly decided
    % isRandom = 0: initial center is first k data point
    % otherwise decide randomly 
    %
    % Output:
    % result is a n*1 matrix of label
    % center are the clustering centers
    % totalWithin is the sum of k within group sum
    
    [n d] = size(data);
    result = zeros(n,1);
    
    % get initial centers
    if isRandom == 0
        center = data (1:k,:);
    else
        randidx=randperm(n,k)';
        center = data(randidx,:);
    end
    
    converged = 0;
    iter = 1;
    while ~converged
        totalWithin=0;
        
        % calculate the close point        
        D = bsxfun(@plus,dot(center',center',1)',dot(data',data',1))-2*(center*data');
        [~, idx] = min(D,[],1);
        result = idx';
        
        % recenter
        means = zeros(k,d);
        for j=1:k
            idx2 = find(result==j);
            means(j,:) = mean(data(idx2,:));
        end
        
        if ~any(center ~= means) | iter == 200
            converged = 1;
        end
        
        center=means;
        
%         calculate the total within group sum
        if converged ==1
            totalWithin=0;
            for j=1:k
                for i=1:n
                    if result(i)==j
                        v = center(j,:)-data(i,:);
                        totalWithin = totalWithin + norm(v);
                    end
                end
            end
        totalWithin
        iter 
        end
        
        iter = iter + 1;
        
    end
    
end