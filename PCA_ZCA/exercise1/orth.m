function [ output] = orth( raw )
%raw是输入列向量组成的矩阵
%对输入的列向量标准正交化
%output是输出列向量组成的矩阵
[m,n ]=size(raw);
output=zeros(m,n);
temp=zeros((n-1) ,n);
for i=1:n-1
    for j=i:n
        temp(i,j) = dot(raw(:,i),raw(:,j));
    end
end

if all(temp(:))==0
    output=raw;
 %判断是否已经正交，是则跳过正交化
else
    output(:,1)=raw(:,1);
    for i=2:n
        for j=1:n-1
            output(:,i)=raw(:,i) - output(:,j)*dot(raw(:,i),output(:,j))/ dot(output(:,j),output(:,j));
        end
    end
end
%开始标准化
for i=1:n
    output(:,i)=output(:,i)/sqrt(sum( output(:,i).^2 ));
end


end

