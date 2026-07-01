%==========================================================================
% Compute the tensor contraction of two tensors using explicitly supplied
% logical numbers of dimensions.
%
% Input:
%   X        - the first tensor
%   Y        - the second tensor
%   n        - modes of X used for contraction
%   m        - modes of Y used for contraction
%   numDimsX - logical number of dimensions of X
%   numDimsY - logical number of dimensions of Y
%
% Output:
%   Out      - the contraction result
%
% This function performs the same contraction as tensor_contraction.m.
% The only essential difference is that it does not infer the tensor orders
% with ndims(X) and ndims(Y). This preserves logical trailing singleton
% dimensions, for example when an intended four-dimensional tensor has size
% [R, I, L, 1].
%==========================================================================
function Out = tensor_contraction_explicit( ...
    X, Y, n, m, numDimsX, numDimsY)

% Logical size vectors, including trailing singleton dimensions.
Lx = ones(1, numDimsX);
Ly = ones(1, numDimsY);

for k = 1:numDimsX
    Lx(k) = size(X, k);
end

for k = 1:numDimsY
    Ly(k) = size(Y, k);
end

% Uncontracted modes.
indexx = 1:numDimsX;
indexy = 1:numDimsY;

indexx(n) = [];
indexy(m) = [];

% Put the contracted modes of X last and those of Y first.
tempX = permute(X, [indexx, n]);
tempY = permute(Y, [m, indexy]);

% Convert the tensor contraction into a matrix product.
tempXX = reshape(tempX, prod(Lx(indexx)), prod(Lx(n)));
tempYY = reshape(tempY, prod(Ly(m)), prod(Ly(indexy)));

tempOut = tempXX * tempYY;

% Restore the logical uncontracted tensor dimensions.
outputSize = [Lx(indexx), Ly(indexy)];

% MATLAB requires at least two dimensions in a reshape size vector.
% This does not change the mathematical result; omitted dimensions are
% singleton dimensions.
if isempty(outputSize)
    outputSize = [1, 1];
elseif isscalar(outputSize)
    outputSize = [outputSize, 1];
end

Out = reshape(tempOut, outputSize);
end
