function X = tucker_full_local(Core, U)
%TUCKER_FULL_LOCAL Reconstructs a full tensor from a Tucker representation.
%
%   X = tucker_full_local(Core, U)
%
%   Core is an R_1 x R_2 x ... x R_N core tensor.
%   U{n} is an I_n x R_n factor matrix.
%
%   The result is equivalent to:
%
%       X = double(ttm(tensor(Core), U));
%
%   but does not require Tensor Toolbox.

    N = numel(U);

    X = Core;

    for n = 1:N
        X = mode_product_local(X, U{n}, n);
    end
end


function Y = mode_product_local(X, A, n)
%MODE_PRODUCT_LOCAL Computes the mode-n tensor-matrix product.
%
%   Y = mode_product_local(X, A, n)
%
%   If X has size:
%
%       I_1 x ... x I_n x ... x I_N
%
%   and A has size:
%
%       J x I_n
%
%   then Y has size:
%
%       I_1 x ... x J x ... x I_N

    sz = size(X);

    if numel(sz) < n
        sz(end+1:n) = 1;
    end

    N = numel(sz);

    In = sz(n);

    if size(A, 2) ~= In
        error(['Dimension mismatch in mode %d: ' ...
               'size(A,2) = %d, but size(X,%d) = %d.'], ...
               n, size(A, 2), n, In);
    end

    J = size(A, 1);

    % Move mode n to the first dimension.
    perm = [n, 1:n-1, n+1:N];
    Xp = permute(X, perm);

    % Matricize tensor along mode n.
    Xmat = reshape(Xp, In, []);

    % Matrix multiplication along mode n.
    Ymat = A * Xmat;

    % Restore tensor shape with updated dimension n.
    newSz = sz;
    newSz(n) = J;

    Yp = reshape(Ymat, [J, sz(perm(2:end))]);

    % Inverse permutation.
    invPerm = zeros(1, N);
    invPerm(perm) = 1:N;

    Y = permute(Yp, invPerm);

    % Enforce expected shape explicitly.
    Y = reshape(Y, newSz);
end