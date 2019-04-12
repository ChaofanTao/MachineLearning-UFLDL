function [centroid, result] = Kmeanspp(data, k, iteration)
% choose initial cluster centroids appropriately
% choose an initial cluster centroid randomly from data
centroid = data(randperm(size(data,1),1),:);

% choose other centroids (a total number of k-1) through roulette method
for i = 2:k
    distance_matrix = zeros(size(data,1),i-1);
    for j = 1:size(distance_matrix,1)
        for k = 1:size(distance_matrix,2)
            distance_matrix(j,k) = sum((data(j,:)-centroid(k,:)) .^ 2);
        end
    end
    % choose next centroid according to distances between samples and
    % previous cluster centroids.
    index = Roulettemethod(distance_matrix);
    %利用前k-1个质心推算出第k个质心
    centroid(i,:) = data(index,:);
    clear distance_matrix;
end

% following steps are same to original k-means
result = zeros(size(data,1),1);
distance_matrix = zeros(size(data,1), k);
SSE = [];
flag = 0;

for i = 1:iteration
    
    previous_result = result;
    
    % calculate distance between each sample and each centroid
    for j = 1:size(distance_matrix,1)
        for k = 1:size(centroid,1)
            distance_matrix(j,k) = sqrt(sum((data(j,:)-centroid(k,:)) .^ 2));
        end
    end
    
    % assign each sample to the nearest controid
    [~,result] = min(distance_matrix,[],2);
    SSE(i) = 0;
    % recalculate centroid locations after assignment
    for j = 1:k
        centroid(j,:) = mean(data(result(:,1) == j,:));
        [m, ~] = size(data(result(:,1)==j,:));
        SSE(i) = SSE(i) + sum(sqrt(sum( (repmat(centroid(j,:),m,1) - data(result(:,1)==j,:)).^2, 2 )));
    end
    % if classified results on all samples do not change after an iteration, 
    % clustering process will quit immediately
    if(result == previous_result)
        flag =flag +1;
        if flag == 5
            fprintf('Clustering over after %i iterations...\n',i);
            break;
        end
    end
end
colors = ['b','g','r','y','m','c','k'];
for i=1:k
    hold on
    plot3(centroid(i,1),centroid(i,2),centroid(i,3),'Marker','o','MarkerFaceColor',colors(i),'MarkerSize',8)
    axis([-50 350 -50 350 -50 350]);
    hold on
    subdata = data(result(:,1)==i,:); %第i类有n个点
    [n,~] = size(subdata);
    for j=1:n
        %'Color',[rand(),rand(),rand()]
        plot3(subdata(j,1),subdata(j,2),subdata(j,3),colors(i),'Marker','*','MarkerSize',5)
        axis([-50 350 -50 350 -50 350]);
    end
end
figure;
plot(SSE,'-o')
title('SSE');

end
