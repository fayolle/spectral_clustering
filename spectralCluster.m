function Y = spectralCluster(X, k, method)
% An implementation of spectral clustering. For details, refer to:
% "A tutorial on spectral clustering" by U. von Luxburg.
% https://arxiv.org/abs/0711.0189
%
% INPUTS:
% X: Input affinity matrix
% k: Number of clusters
% method: Method to use for spectral clustering.
% 1: L_{rw} by computing generalized eigenvalue (of L,D) [default]
% 2: use L_{rw}
% 3: use L_{sym}
%
% OUTPUT:
% Y: Cluster indices
%

if (nargin < 3)
    method = 1;
end

% degree matrix
D = diag(sum(X, 1));

% graph Laplacian
L = D - X;

if method == 1
    % solve the generalized eigen decomposition
    % L V = lambda D V
    % i.e. D^(-1) L V = lambda V
    % L_{rw} = D^(-1) L
    [V, S] = eig(L, D);
    
elseif method == 2
    % not sure that this is a good idea since L is not symmetric
    L = D^(-1) * L;
    [V, S] = eig(L);
    
elseif method == 3
    % L_{sym}
    L = D^(-1/2) * L * D^(-1/2);
    [V,S] = eig(L);
    
end

% sort eigenvalues
abs_d = abs(diag(S));
[~, idx] = sort(abs_d);

Vk = V(:, idx(1:k));

% k-means clustering
[idx, ~] = kmeans(Vk, k);

% assignments
Y = zeros(size(X, 1), k);
for i=1:k
    k_idx = idx == i;
    Y(k_idx, i) = 1;
end

end
