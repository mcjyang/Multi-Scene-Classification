clc; clear all; close all;

X = load('../digits/digit.txt');
Y = load('../digits/labels.txt');

[n d] = size(X);
truth = unique(Y);
num = size(truth,1);

% 3.5.1 / 3.5.2
k=4;
random=0;
[label,center,~] = MyKmeans(X, k, random);


sameY = 0;
sameL = 0;
diffL = 0;

for i=1:n
    for j=1:n
        if Y(i) == Y(j)
            sameY = sameY+1;
            if label(i) == label(j)
                sameL = sameL+1;
            end
        else
            if label(i) ~= label(j)
                diffL = diffL+1;
            end
        end
        
    end
end

p1 = sameL/sameY
p2 = diffL/(n*n-sameY)
p3 = (p1+p2)/2

% 3.5.3

random = 1;
within=zeros(1,10);
average = zeros(1,10);
for iter=1:10
    for k=1:10
        [~,~,totalSum] = MyKmeans(X, k, random);
        within(k)=totalSum;
    end
    average = average+within;
end
average = average/10;

fig1=figure(1);
plot([1:10],average,'-ro');
legend('Total SS versus k');
saveas(fig1,'totalSS_versus_k.png');
close();

% 3.5.4

random = 1;
p=zeros(3,10);
average_p=zeros(3,10);

for iter=1:10
    for k=1:10
        [label, center, ~] = MyKmeans(X,k,random);
        sameY = 0;
        sameL = 0;
        diffL = 0;

        for i=1:n
            for j=1:n
                if Y(i) == Y(j)
                    sameY = sameY+1;
                    if label(i) == label(j)
                        sameL = sameL+1;
                    end
                else
                    if label(i) ~= label(j)
                        diffL = diffL+1;
                    end
                end
            end
        end

        p(1,k) = sameL/sameY;
        p(2,k) = diffL/(n*n-sameY);
        p(3,k) = (p(1,k)+p(2,k))/2;
    end
    average_p = average_p + p;
end
average_p = average_p/10;
fig2=figure(2);
plot([1:10],average_p(1,:),'-ro',[1:10],average_p(2,:),'-bx',[1:10],average_p(3,:),'-g+');
legend('p1','p2','p3');
saveas(fig2,'p_versus_k.png');
close();

