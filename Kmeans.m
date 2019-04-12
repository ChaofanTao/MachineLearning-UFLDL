function [centroid, result] = Kmeans(data, k, loop_max)

% choose initial k cluster centroids randomly
centroid = data(randperm(size(data,1),k)', :);

% pre-allocate a variable to record distances between samples and centroids
distance_matrix = zeros(size(data,1), k);

% pre-allocate result
result = zeros(size(data,1),1);
loop=0;

for i = 1:loop_max
    loop=loop+1;
    % used to monitor changes on result
    previous_result = result;
    
    % calculate distance between each sample and each centroid
    for j = 1:size(distance_matrix,1)
        for k = 1:size(distance_matrix,2)
            distance_matrix(j,k) = sqrt(sum((data(j,:)-centroid(k,:)) .^ 2));
        end
    end
    
    % assign each sample to the nearest cluster controid
    [~,result] = min(distance_matrix,[],2);
    
    % recalculate cluster centroid locations after assignment
    for j = 1:k
        centroid(j,:) = mean(data(result(:,1) == j,:));
    end
    
    % if classified results on all samples do not change after an iteration, 
    % clustering process will quit immediately
    if(result == previous_result)
        
        break;
    end
end


for i=1:k
    hold on
    plot(centroid(i,1),centroid(i,2),'k+' )
end

result=result';
fprintf('Clustering over after %i iterations...\n',loop);
title('Kmeans');
hold on

end


