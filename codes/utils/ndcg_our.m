function res=ndcg2_k(I, Lbase, Lquery, n_k)
%% 
% I is the predicted oeder, an n by m matrix
% y is the ground truth relevance, an n by m matrix
% n is the size of query set
% m is the size of retrieval set
% n_k is the top number

% I = full(I);
% y = full(y);

if ~exist('n_k','var') || n_k==0
    n_k = size(Lbase,1); % Configurable
end

%n_k = 1000;

y = Lquery*Lbase';
clear Lbase Lquery

% return the averaged ndcg for retrieving items for the users
[n,m]=size(I);

%% compute the ranks of the items for each user

[~, ideal_I] = sort(y, 2, 'descend');

% res = zeros(1, n_k);
res = 0;
cnt = 0;
for i=1:n
    
    
    nominator = (2.^y(i, I(i,1:n_k))-1)./log2([1:n_k]+1);
    denominator = (2.^y(i, ideal_I(i,1:n_k))-1) ./ log2([1:n_k]+1);
    

    tmp = sum(nominator)./ sum(denominator);
    cnt = cnt + 1;
    %tmp = full(tmp);
    %tmp = padarray(tmp, [0, length(k_vals) - size(tmp, 2)], 0, 'post');
	res = res + tmp;
end
res = res / cnt;
