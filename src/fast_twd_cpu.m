function [X_hat, G, Core, Out] = fast_twd_cpu(X, Omega, opts)

% FAST_TWD_CPU Matrix-free CPU solver for tensor-wheel decomposition.
%
% Syntax
%   [X_hat, G, Core, Out] = fast_twd_cpu(X, Omega, opts)
%   [X_hat, G, Core, Out] = fast_twd_cpu(X, opts)
%
% Description
%   Computes a tensor-wheel (TW) approximation of an N-way tensor X using
%   proximal alternating minimization (PAM) and matrix-free tensor
%   contractions.
%
%   The TW model consists of factors
%
%       G{n} in R^{R_n x I_n x L_n x R_{n+1}},
%
%   with R_{N+1} = R_1, and a core tensor
%
%       Core in R^{L_1 x ... x L_N}.
%
% Inputs
%   X
%       Input tensor of size I_1 x ... x I_N.
%
%   Omega
%       Optional observed-entry set, supplied as linear indices or as a
%       logical mask of the same size as X. If Omega is omitted, the call
%
%           fast_twd_cpu(X, opts)
%
%       is used.
%
%   opts
%       Structure containing algorithm parameters.
%
% Options
%   opts.R
%       Required 2-by-N matrix of maximum TW ranks:
%
%           opts.R(1,:) = [R_1, ..., R_N]   outer ranks,
%           opts.R(2,:) = [L_1, ..., L_N]   inner ranks.
%
%   opts.tol
%       Stopping tolerance for the relative change between consecutive
%       tensor estimates. Default: 1e-6.
%
%   opts.maxit
%       Maximum number of PAM iterations. Default: 500.
%
%   opts.rho
%       Proximal regularization parameter. Default: 1e-3.
%
%   opts.seed
%       Nonnegative integer seed used for reproducible initialization of
%       the TW factors and core with the Mersenne Twister generator.
%       Default: 0.
%
%   opts.num_padarray
%       Rank-reduction parameter used at initialization. The initial ranks
%       are computed elementwise as
%
%           R_init = max(opts.R - opts.num_padarray, min(2, opts.R)).
%
%       Default: 0.
%
%   opts.core_update_after
%       Iteration threshold for scheduled core updates. Default: 3.
%
%   opts.core_update_every
%       Core-update period after opts.core_update_after. Default: 2.
%
%   opts.enforceOmega
%       If true and Omega is nonempty, the observed entries are restored
%       after every tensor update:
%
%           X_hat(Omega) = X(Omega).
%
%       Default: ~isempty(Omega).
%
% Outputs
%   X_hat
%       Reconstructed tensor with the same size as X.
%
%   G
%       N-element cell array containing the TW factors.
%
%   Core
%       Current TW core tensor.
%
%   Out
%       Diagnostic structure containing:
%
%           Out.RSE
%               Relative changes recorded during the PAM iterations.
%
%           Out.RES_init
%               Initial relative residual
%
%                   norm(X(:) - Y0(:)) / norm(Y0(:)).
%
%           Out.iterations
%               Number of PAM iterations executed before termination.
%
%           Out.converged
%               Logical flag indicating whether the relative-change
%               stopping criterion was satisfied.
%
%           Out.final_R
%               Final 2-by-N matrix of TW ranks:
%
%                   Out.final_R(1,:) = [R_1, ..., R_N]   outer ranks,
%                   Out.final_R(2,:) = [L_1, ..., L_N]   inner ranks.
%
%               These ranks may differ from the initial ranks if adaptive
%               rank growth was activated during the iterations.
%
% Notes
%   Factor and core updates are computed from tensor-network contractions
%   without explicitly forming the large least-squares design matrices used
%   by the reference TW implementation.
%
%   The core is updated in the first iteration, after adaptive rank growth,
%   and according to opts.core_update_after and opts.core_update_every.
% =========================================================================
% -------------------- argument handling --------------------
% Valid calls:
%   fast_twd_cpu(X, opts)
%   fast_twd_cpu(X, Omega, opts)
% =========================================================================
if nargin == 2
    opts = Omega;
    Omega = [];
elseif nargin == 3
    % X, Omega, opts supplied explicitly
elseif nargin < 2
    error('fast_twd_cpu requires at least X and opts.');
else
    error('fast_twd_cpu accepts either (X, opts) or (X, Omega, opts).');
end

% -------------------- basic argument validation --------------------

if ~isstruct(opts) || ~isscalar(opts)
    error('fast_twd_cpu:InvalidOptions', ...
        'opts must be a scalar structure.');
end

if isempty(X) || ~isa(X, 'double') || ~isreal(X)
    error('fast_twd_cpu:InvalidInputTensor', ...
        'X must be a nonempty real tensor of class double.');
end

% -------------------- options --------------------

tol   = 1e-6;
maxit = 500;
rho   = 1e-3;
max_R = [];
seed  = 0;

% Core-update schedule.
core_update_after = 3;
core_update_every = 2;

% Initialization padding.
num_padarray = 0;

if isfield(opts, 'tol')
    tol = opts.tol;
end

if isfield(opts, 'maxit')
    maxit = opts.maxit;
end

if isfield(opts, 'rho')
    rho = opts.rho;
end

if isfield(opts, 'R')
    max_R = opts.R;
end

if isfield(opts, 'seed') && ~isempty(opts.seed)
    seed = opts.seed;
end

if isfield(opts, 'core_update_after')
    core_update_after = opts.core_update_after;
end

if isfield(opts, 'core_update_every')
    core_update_every = opts.core_update_every;
end

if isfield(opts, 'num_padarray')
    num_padarray = opts.num_padarray;
end

% -------------------- rank validation --------------------

if isempty(max_R)
    error('fast_twd_cpu:MissingRanks', ...
        'opts.R must be provided as a 2-by-N matrix [R; L].');
end

if ~isnumeric(max_R) || ~isreal(max_R) || ~ismatrix(max_R) || ...
        size(max_R, 1) ~= 2 || size(max_R, 2) < 2
    error('fast_twd_cpu:InvalidRankSize', ...
        ['opts.R must be a real 2-by-N matrix of outer and inner ', ...
         'TW ranks, with N >= 2.']);
end

if any(~isfinite(max_R(:))) || ...
        any(max_R(:) < 1) || ...
        any(max_R(:) ~= fix(max_R(:)))
    error('fast_twd_cpu:InvalidRanks', ...
        'All entries of opts.R must be finite positive integers.');
end

% The logical tensor order is determined by opts.R. This also preserves
% possible trailing physical dimensions of size one.
Ndim = size(max_R, 2);

Nway = ones(1, Ndim);
for n = 1:Ndim
    Nway(n) = size(X, n);
end

% Reject additional non-singleton physical dimensions not represented
% by opts.R.
sizeX = size(X);
if numel(sizeX) > Ndim && any(sizeX(Ndim+1:end) ~= 1)
    error('fast_twd_cpu:InconsistentTensorOrder', ...
        ['The number of physical modes of X is inconsistent with ', ...
         'the number of columns in opts.R.']);
end

% -------------------- scalar option validation --------------------

validateattributes(tol, {'numeric'}, ...
    {'scalar', 'real', 'finite', 'positive'}, ...
    mfilename, 'opts.tol');

validateattributes(maxit, {'numeric'}, ...
    {'scalar', 'real', 'finite', 'integer', 'positive'}, ...
    mfilename, 'opts.maxit');

validateattributes(rho, {'numeric'}, ...
    {'scalar', 'real', 'finite', 'positive'}, ...
    mfilename, 'opts.rho');

validateattributes(core_update_after, {'numeric'}, ...
    {'scalar', 'real', 'finite', 'integer', 'nonnegative'}, ...
    mfilename, 'opts.core_update_after');

validateattributes(core_update_every, {'numeric'}, ...
    {'scalar', 'real', 'finite', 'integer', 'positive'}, ...
    mfilename, 'opts.core_update_every');

validateattributes(num_padarray, {'numeric'}, ...
    {'scalar', 'real', 'finite', 'integer', 'nonnegative'}, ...
    mfilename, 'opts.num_padarray');

validateattributes(seed, {'numeric'}, ...
    {'scalar', 'real', 'finite', 'integer', 'nonnegative'}, ...
    mfilename, 'opts.seed');

if seed > 2^32 - 1
    error('fast_twd_cpu:InvalidSeed', ...
        'opts.seed must not exceed 2^32-1.');
end

% -------------------- Omega validation --------------------

if ~isempty(Omega)

    if islogical(Omega)

        omegaSize = ones(1, Ndim);
        for n = 1:Ndim
            omegaSize(n) = size(Omega, n);
        end

        sizeOmega = size(Omega);

        hasExtraNonSingletonModes = ...
            numel(sizeOmega) > Ndim && ...
            any(sizeOmega(Ndim+1:end) ~= 1);

        if numel(Omega) ~= numel(X) || ...
                ~isequal(omegaSize, Nway) || ...
                hasExtraNonSingletonModes
            error('fast_twd_cpu:InvalidLogicalOmega', ...
                ['A logical Omega must have the same logical size ', ...
                 'as the input tensor X.']);
        end

    elseif isnumeric(Omega)

        if ~isreal(Omega) || ~isvector(Omega) || ...
                any(~isfinite(Omega(:))) || ...
                any(Omega(:) ~= fix(Omega(:))) || ...
                any(Omega(:) < 1) || ...
                any(Omega(:) > numel(X))
            error('fast_twd_cpu:InvalidIndexOmega', ...
                ['A numeric Omega must be a vector of valid positive ', ...
                 'integer linear indices into X.']);
        end

    else
        error('fast_twd_cpu:InvalidOmegaType', ...
            ['Omega must be empty, a logical mask, or a numeric ', ...
             'vector of linear indices.']);
    end
end

% -------------------- observed-entry projection option --------------------

if isfield(opts, 'enforceOmega')

    if ~islogical(opts.enforceOmega) || ...
            ~isscalar(opts.enforceOmega)
        error('fast_twd_cpu:InvalidEnforceOmega', ...
            'opts.enforceOmega must be a logical scalar.');
    end

    enforceOmega = opts.enforceOmega;

else
    enforceOmega = ~isempty(Omega);
end

% -------------------- init --------------------
R = max(max_R - num_padarray, min(2, max_R));

X_hat = X;

Factors_dims      = factor_dims(Nway, R);
Max_Factors_dims  = factor_dims(Nway, max_R);

rng(seed, 'twister');
G = cell(Ndim,1);
for i = 1:Ndim
    G{i} = rand(Factors_dims(i,:));
end
Core  = rand(R(2,:));
C_old = Core;

% Initial residuals
Y0 = cores_prod_single_tw(G,Core);
RES_init = norm(X(:)-Y0(:))/norm(Y0(:));

Out.RSE = zeros(1, maxit);
Out.RES_init = RES_init;
converged = false;
r_change = 0.0005;


% -------------------- main loop --------------------
for k = 1:maxit

    X_old = X_hat;

    % ---- Update G_k, k=1,2,...,N ----
    for num = 1:Ndim
        dimsG = size(G{num});
        G2    = unfold(G{num}, dimsG, 2);

        % Compute the exact matrix-free factor right-hand side B_n and
        % Gram matrix H_n without explicitly forming the design matrix Q_n.
        B_ten = tw_factor_rhs_dense_exact(X_hat, G, Core, num);
        TempA = unfold(B_ten, dimsG, 2) + rho * G2;

        TempB = tw_factor_gram_dense_exact(G, Core, num);
        J = size(TempB,1);
        TempB(1:J+1:end) = TempB(1:J+1:end) + rho;

        % Solve TempA * inv(TempB) via Cholesky (fast) with safe fallback.
        [Lchol,p] = chol(TempB,'lower');
        if p==0
            Sol = (TempA / Lchol') / Lchol;
        else
            % Rare: numerical issues. Add a tiny jitter and retry once.
            TempB(1:J+1:end) = TempB(1:J+1:end) + 1e-12;
            [Lchol,p] = chol(TempB,'lower');
            if p==0
                Sol = (TempA / Lchol') / Lchol;
            else
                Sol = TempA / TempB;
            end
        end
        G{num} = fold(Sol, dimsG, 2);
    end

    % ---- Update the core tensor C (fast contraction-based) ----
    do_core = (k==1) || (numel(Core) > numel(C_old)) || ...
              (k > core_update_after && mod(k, core_update_every) == 0);

    if do_core
        % b = A*vec(X_hat) as a tensor of size(L1,...,LN)
        b_ten = tw_core_rhs_dense(X_hat, G);
        b_vec = b_ten(:);

        % H = A*A' as a (prod(L) x prod(L)) SPD matrix
        H = tw_core_gram(G);

        % Proximal solve: (H + rho I) c = b + rho c_prev       
        nC = numel(Core);
        % Add rho to diagonal in-place (avoid speye allocation)
        H(1:nC+1:end) = H(1:nC+1:end) + rho;
        rhs = b_vec + rho*Core(:);

        % Cholesky solve with p-flag (avoid try/catch overhead)
        [Lchol,p] = chol(H,'lower');
        if p==0
            c_vec = Lchol' \ (Lchol \ rhs);
        else
            % Rare: add jitter and retry once
            H(1:nC+1:end) = H(1:nC+1:end) + 1e-12;
            [Lchol,p] = chol(H,'lower');
            if p==0
                c_vec = Lchol' \ (Lchol \ rhs);
            else
                c_vec = H \ rhs;
            end
        end
        Core = reshape(c_vec, size(Core));
    end

    % ---- Update X_hat ----
    Xhat_tmp = cores_prod_single_tw(G, Core);
    X_hat = (Xhat_tmp + rho*X_old) / (1 + rho);

    if enforceOmega && ~isempty(Omega)
       X_hat(Omega) = X(Omega);
    end

    % ---- Convergence ----
    rse = norm(X_hat(:) - X_old(:)) / max(1e-12, norm(X_old(:)));
    Out.RSE(k) = rse;

    if k == 1 || mod(k, 20) == 0
        fprintf('fast TWD (CPU): iter = %d   RSE=%.10f\n', k, rse);
    end
    if rse < tol
       converged = true;
       break;
    end

    % ---- Adaptive rank increment (kept compatible with baseline) ----
    C_old = Core;
    rank_inc = double(Factors_dims < Max_Factors_dims);
    if rse < r_change && sum(rank_inc(:)) ~= 0
        [G, Core] = rank_inc_adaptive(G, Core, rank_inc, Ndim);
        Factors_dims = Factors_dims + rank_inc;
        r_change = r_change * 0.1;
    end
end

% Trim preallocated history
Out.RSE = Out.RSE(1:k);

% Final diagnostics
Out.iterations = k;
Out.converged  = converged;

% Final rank matrix:
%   first row  - outer ranks R_n,
%   second row - inner ranks L_n.
Out.final_R = [Factors_dims(:,1).'; Factors_dims(:,3).'];

end

% ======================== helpers ========================

function B_ten = tw_factor_rhs_dense_exact(X_hat, G, Core, n)
% Compute the factor right-hand side
%
%     B_n = X_hat_(n) * Q_n^T
%
% exactly without explicitly forming the design matrix Q_n.
% B_ten has logical size [R_n, I_n, L_n, R_{n+1}], and its
% mode-2 unfolding equals B_n.

N = numel(G);
seq = [n+1:N, 1:n-1];

% Bring mode n first: [I_n, I_{n+1}, ..., I_{n-1}]
Xp = permute(X_hat, [n, seq]);

% First factor contraction: contract I_{n+1}
% T layout after step 1:
%   [I_n, I_{n+2}, ..., I_{n-1}, R_{n+1}, L_{n+1}, R_{n+2}]
T = tensor_contraction_explicit(Xp, G{seq(1)}, 2, 2, N, 4);
logicalNdimsT = N + 2;

% Process the remaining factors in circular order.
for t = 2:numel(seq)
    j = seq(t);
    % Contract next physical mode and the carried outer rank.
    T = tensor_contraction_explicit(T, G{j},[2, logicalNdimsT], [2, 1], ...
    logicalNdimsT, 4);
end

% Contract all L_j, j ~= n, with the core; leave L_n open.
Cperm = permute(Core, [seq, n]);
T = tensor_contraction_explicit(T, Cperm, 3:(N+1), 1:(N-1), ...
    logicalNdimsT, N);
% T layout is now [I_n, R_{n+1}, R_n, L_n]

B_ten = permute(T, [3, 1, 4, 2]);  % [R_n, I_n, L_n, R_{n+1}]
end


function H_n = tw_factor_gram_dense_exact(G, Core, n)
% Compute the factor Gram matrix
%
%     H_n = Q_n * Q_n^T
%
% exactly without explicitly forming Q_n.
% The output size is J_n-by-J_n, where
%
%     J_n = R_n * L_n * R_{n+1}.

N = numel(G);
seq = [n+1:N, 1:n-1];

Mk = cell(N,1);
for j = seq
    % Contract physical mode i_j between two copies of G{j}.
    Tj = tensor_contraction_explicit(G{j}, G{j}, 2, 2, 4, 4);
    % Tj dims: [R_j, L_j, R_{j+1}, R'_j, L'_j, R'_{j+1}]
    Mk{j} = permute(Tj, [1, 4, 2, 5, 3, 6]);
    % Mk{j}: [R_j, R'_j, L_j, L'_j, R_{j+1}, R'_{j+1}]
end

U = Mk{seq(1)};
logicalNdimsU = 6;
for t = 2:numel(seq)
    j = seq(t);
    U = tensor_contraction_explicit(U, Mk{j}, ...
    [logicalNdimsU-1, logicalNdimsU], [1, 2], ...
    logicalNdimsU, 6);
    logicalNdimsU = logicalNdimsU + 2;
end
% U: [R_{n+1},R'_{n+1},L_seq(1),L'_seq(1),...,L_seq(end),L'_seq(end),R_n,R'_n]

Cperm = permute(Core, [seq, n]);

% Contract the unprimed L_seq modes with Core.
V = tensor_contraction_explicit(U, Cperm, 3:2:(2*N-1), 1:(N-1), ...
    logicalNdimsU, N);
logicalNdimsV = N + 4;
% V: [R_{n+1},R'_{n+1},L'_seq...,R_n,R'_n,L_n]

% Contract the primed L'_seq modes with a second copy of Core.
W = tensor_contraction_explicit(V, Cperm, 3:(N+1), 1:(N-1), ...
    logicalNdimsV, N);
% W: [R_{n+1},R'_{n+1},R_n,R'_n,L_n,L'_n]

W = permute(W, [3, 5, 1, 4, 6, 2]);
% W: [R_n,L_n,R_{n+1},R'_n,L'_n,R'_{n+1}]

J = size(G{n},1) * size(G{n},3) * size(G{n},4);
H_n = reshape(W, J, J);
H_n = (H_n + H_n.') * 0.5;   % Remove roundoff-induced asymmetry before the Cholesky factorization.
end


function b_ten = tw_core_rhs_dense(X_hat, G)
% Compute the core right-hand side
%
%     b_C = A_C * vec(X_hat),
%
% where
%
%     vec(X_hat) = A_C^T * vec(Core).
%
% The result has logical size [L_1, ..., L_N].

N = numel(G);

% First contraction: contract the physical mode I_1.
T = tensor_contraction_explicit(G{1}, X_hat, 2, 1, 4, N); % [R1, L1, R2, I2..IN]
logicalNdimsT = N + 2;

% Enforce canonical ordering: [R1, L1..L_{k-1}, Rk, Ik, I_{k+1}..IN]
for k = 2:N-1
    % current T dims: [R1, L1..L_{k-1}, Rk, Ik, I_{k+1}..IN]
    % contract (Rk,Ik) with G{k}(Rk,Ik,Lk,R{k+1})
    mode_Rk = 1 + (k-1) + 1; % 1 (R1) + (k-1) L-modes + Rk
    mode_Ik = mode_Rk + 1;
    T = tensor_contraction_explicit(T, G{k}, [mode_Rk, mode_Ik], [1, 2], ...
    logicalNdimsT, 4);

    % After contraction, dims are:
    % [R1, L1..L_{k-1}, I_{k+1}..IN, Lk, R{k+1}]
    nI = N - k;
    prefix_len = 1 + (k-1); % R1 + previous L's
    % current dims: [prefix, I-block (nI dims), Lk, R{k+1}]
    perm = [1:prefix_len, prefix_len+nI+1, prefix_len+nI+2, prefix_len+1:prefix_len+nI];
    T = permute(T, perm);
end

% Now T dims: [R1, L1..L_{N-1}, RN, IN]
% Final contraction with G{N} closes ring by contracting (R1,RN,IN).
mode_R1 = 1;
mode_RN = 1 + (N-1) + 1; % R1 + (N-1) L-modes + RN
mode_IN = mode_RN + 1;

T = tensor_contraction_explicit(T, G{N}, [mode_R1, mode_RN, mode_IN], [4, 1, 2], ...
    logicalNdimsT, 4);

% Output dims: [L1..L_{N-1}, LN]
b_ten = T;
end

function H = tw_core_gram(G)
% Compute H = A*A' (size prod(L) x prod(L)) without forming A.
% Uses double-layer transfer tensors and ring contraction.
N = numel(G);

L = zeros(1,N);
Mk = cell(N,1);
for k = 1:N
    L(k) = size(G{k}, 3);
    % Contract physical index i_k (mode-2) between two copies of G{k}
    T = tensor_contraction_explicit(G{k}, G{k}, 2, 2, 4, 4);

    % T dims: [Rk, Lk, R{k+1}, Rk', Lk', R{k+1}']
    % Reorder to [Rk, Rk', Lk, Lk', R{k+1}, R{k+1}']
    Mk{k} = permute(T, [1, 4, 2, 5, 3, 6]);
end

% Contract the ring over outer ranks (and their primed copies)
T = Mk{1}; % [R1,R1',L1,L1',R2,R2']
logicalNdimsT = 6;
for k = 2:N
    % Contract last two modes (Rk,Rk') with first two modes of Mk{k}
    T = tensor_contraction_explicit(T, Mk{k}, ...
    [logicalNdimsT-1, logicalNdimsT], [1, 2], ...
    logicalNdimsT, 6);

    logicalNdimsT = logicalNdimsT + 2;
    % Result ends with [R{k+1}, R{k+1}']
end

% Close the ring: sum over (R1,R1') with (last R1,R1')
sz = ones(1, logicalNdimsT);
for d = 1:logicalNdimsT
    sz(d) = size(T, d);
end
r1 = sz(1);
r1p = sz(2);

Ldims = sz(3:logicalNdimsT-2); % [L1,L1',...,LN,LN']
Hten = zeros(Ldims);
idx = repmat({':'}, 1, logicalNdimsT);
for a = 1:r1
    idx{1} = a;
    idx{end-1} = a;
    for b = 1:r1p
        idx{2} = b;
        idx{end} = b;
	        % T(idx{:}) keeps the indexed outer-rank modes as singleton
	        % dimensions (size 1). 
	        slice = T(idx{:});
	        slice = reshape(slice, Ldims);
	        Hten = Hten + slice;
    end
end

% Permute from interleaved [L1,L1',L2,L2',...] to [L1..LN, L1'..LN']
permL = [1:2:(2*N), 2:2:(2*N)];
Hten = permute(Hten, permL);

H = reshape(Hten, prod(L), prod(L));
% Symmetrize for numerical stability
H = (H + H.') * 0.5;
end

function [G, Core] = rank_inc_adaptive(G, Core, rank_inc, N)
%RANK_INC_ADAPTIVE Increase the current TW ranks by post-padding.

for j = 1:N
    fillValue = rand(1);
    G{j} = pad_post_constant(G{j}, rank_inc(j,:), fillValue);
end

fillValue = rand(1);
Core = pad_post_constant(Core, rank_inc(:,3), fillValue);
end

function B = pad_post_constant(A, padSize, fillValue)
%PAD_POST_CONSTANT Pad an array at the end of each dimension.
%   B = PAD_POST_CONSTANT(A, PADSIZE, FILLVALUE) enlarges A by PADSIZE
%   elements along the corresponding dimensions. 

padSize = double(padSize(:).');

if any(~isfinite(padSize)) || any(padSize < 0) || ...
        any(padSize ~= round(padSize))
    error('padSize must contain finite nonnegative integers.');
end

oldSize = size(A);
nDims = max(numel(oldSize), numel(padSize));

oldSize(end+1:nDims) = 1;
padSize(end+1:nDims) = 0;

newSize = oldSize + padSize;

% Preserve the numeric type of A.
B = repmat(cast(fillValue, 'like', A), newSize);

subs = arrayfun(@(s) 1:s, oldSize, 'UniformOutput', false);
B(subs{:}) = A;
end


function W = fold(W, dim, i)
nd = numel(dim);

if nd <= 1
    W = reshape(W, dim);
    return
end

ord = [i, 1:i-1, i+1:nd];
W   = reshape(W, dim(ord));

if i ~= 1
    W = permute(W, [2:i, 1, i+1:nd]);
end
end

function W = unfold(W, dim, i)
nd = numel(dim);

if nd <= 1
    W = reshape(W, dim(i), []);
    return
end

if i ~= 1
    W = permute(W, [i, 1:i-1, i+1:nd]);
end

W = reshape(W, dim(i), []);
end

function X_hat = cores_prod_single_tw(G,C)

    Nc = length(G);

    % The intermediate tensor has Nc+2 logical dimensions.
    % This value must not be determined with ndims(X_hat), because MATLAB
    % omits trailing singleton dimensions.
    logicalNdimsX = Nc + 2;

    % Contract L_1 between G{1} and the core C.
    % G{1} has 4 logical dimensions and C has Nc dimensions.
    X_hat = tensor_contraction_explicit(G{1}, C, 3, 1, 4, Nc);

    % Change the layout from
    % [R_1, I_1, R_2, L_2, ..., L_N]
    % to
    % [L_2, ..., L_N, R_1, I_1, R_2].
    X_hat = permute(X_hat, [4:logicalNdimsX, 1:3]);

    for n = 1:Nc-2
        X_hat = tensor_contraction_explicit(X_hat, G{n+1}, ...
            [1, logicalNdimsX], [3, 1], ...
            logicalNdimsX, 4);
    end
    X_hat = tensor_contraction_explicit(X_hat, G{Nc}, ...
        [1, 2, logicalNdimsX], [3, 4, 1], ...
        logicalNdimsX, 4);
end

function Z = factor_dims(Nway,r)

%FACTOR_DIMS Construct the dimensions of the fourth-order TW factors.
%
N = numel(Nway);
Z = zeros(N,4);  % 4th-order tensor
for i=1:N-1
    Z(i,:) = [r(1,i),Nway(i),r(2,i),r(1,i+1)];
end
    Z(N,:) = [r(1,N),Nway(N),r(2,N),r(1,1)];
end