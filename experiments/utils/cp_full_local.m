function X = cp_full_local(U)
%CP_FULL_LOCAL Reconstructs a full tensor from CP factor matrices.
%
%   X = cp_full_local(U)
%
%   U{n} is an I_n x R factor matrix.
%   The result is equivalent to:
%
%       X = double(ktensor(U));
%
%   but does not require Tensor Toolbox.

    N = numel(U);
    R = size(U{1}, 2);

    dims = zeros(1, N);
    for n = 1:N
        dims(n) = size(U{n}, 1);

        if size(U{n}, 2) ~= R
            error('All factor matrices must have the same number of columns R.');
        end
    end

    X = zeros(dims);

    for r = 1:R
        T = 1;

        for n = 1:N
            shape = ones(1, N);
            shape(n) = dims(n);

            T = T .* reshape(U{n}(:, r), shape);
        end

        X = X + T;
    end
end