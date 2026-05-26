function [X, G, Core, Out] = fast_twd_cpu(F, Omega, opts)

% fast_twd_cpu: PAM-based solver for tensor-wheel (TW) decomposition
% with matrix-free contraction-based factor and core updates.
%
% Syntax
%   [X, G, Core, Out] = fast_twd_cpu(F, Omega, opts)
%   [X, G, Core, Out] = fast_twd_cpu(F, opts)
%
% Description
%   fast_twd_cpu computes a TW approximation of the input tensor F 
%   using a proximal alternating minimization (PAM) scheme. The tensor is
%   represented by N fourth-order tensor-wheel factors G{n} and an N-way
%   core tensor Core.
%
%   For an input tensor
%
%       F in R^{I_1 x I_2 x ... x I_N},
%
%   the factor tensors have sizes
%
%       G{n} in R^{R_n x I_n x L_n x R_{n+1}},      n = 1,...,N-1,
%       G{N} in R^{R_N x I_N x L_N x R_1},
%
%   and the core tensor has size
%
%       Core in R^{L_1 x L_2 x ... x L_N}.
%
%   Here R_n are the outer tensor-wheel ranks, and L_n are the inner/core
%   ranks. These ranks are specified through opts.R.
%
% ========================================================================
% Input arguments
%   F
%       Input tensor of size I_1 x I_2 x ... x I_N.
%
%   Omega
%       Index set or logical mask of observed entries. Omega may be either
%       a vector of linear indices or a logical array of the same size as F.
%
%       If Omega is nonempty and opts.enforceOmega is true, the algorithm
%       enforces the observed-entry constraint after each X-update:
%
%           X(Omega) = F(Omega).
%
%       Thus, the observed entries remain fixed to the data tensor F, while
%       the unobserved entries are updated from the current tensor-wheel
%       approximation.
%
%       If Omega is empty, or if opts.enforceOmega is false, the algorithm
%       operates on the full tensor F without enforcing the observed-entry
%       projection.
%
%       If Omega is omitted, the function may be called as
%
%           fast_twd_cpu(F, opts).
%
%   opts
%       Structure containing algorithmic parameters. The required field is
%       opts.R. Other fields are optional and use default values if omitted.
%
% Options in opts
%   opts.R
%       Required. A 2 x N matrix specifying the maximum tensor-wheel ranks:
%
%           opts.R(1,:) = [R_1, R_2, ..., R_N]    outer ranks,
%           opts.R(2,:) = [L_1, L_2, ..., L_N]    inner/core ranks.
%
%       The final factor dimensions are bounded by these ranks. The initial
%       ranks are computed as
%
%           R_init = max(opts.R - opts.num_padarray, 2).
%
%   opts.tol
%       Stopping tolerance for the relative change of X between consecutive
%       PAM iterations.
%       Default:
%
%           opts.tol = 1e-6.
%
%   opts.maxit
%       Maximum number of PAM iterations.
%       Default:
%
%           opts.maxit = 500.
%
%   opts.rho
%       Proximal regularization parameter used in the factor, core, and X
%       updates.
%       Default:
%
%           opts.rho = 1e-3.
%
%   opts.core_update_after
%       Iteration threshold controlling scheduled core updates. In addition
%       to the mandatory first-iteration update and updates after rank
%       growth, the core is updated only when
%
%           k > opts.core_update_after.
%
%       Default:
%
%           opts.core_update_after = 3.
%
%   opts.core_update_every
%       Period of scheduled core updates after opts.core_update_after.
%       The scheduled core update condition is
%
%           k > opts.core_update_after
%           and
%           mod(k, opts.core_update_every) == 0.
%
%       Default:
%
%           opts.core_update_every = 2.
%
%   opts.num_padarray
%       Rank padding parameter used to initialize the algorithm with ranks
%       smaller than opts.R. The initial ranks are
%
%           max(opts.R - opts.num_padarray, 2).
%
%       Set opts.num_padarray = 0 to initialize directly at opts.R.
%       Default:
%
%           opts.num_padarray = 0.
%
%   opts.enforceOmega
%       Logical flag controlling whether the observed-entry projection is
%       applied after each X-update:
%
%           X(Omega) = F(Omega).
%
%       If omitted, the default value is
%
%           opts.enforceOmega = ~isempty(Omega).
%
%       Hence, the projection is enabled automatically whenever Omega is
%       supplied, and disabled when Omega is empty.
%
% ========================================================================
% Output arguments
%   X
%       Reconstructed tensor of the same size as F:
%
%           size(X) = size(F).
%
%   G
%       Cell array of tensor-wheel factors:
%
%           G = cell(N,1).
%
%       The n-th factor has size
%
%           size(G{n}) = [R_n, I_n, L_n, R_{n+1}],    n = 1,...,N-1,
%           size(G{N}) = [R_N, I_N, L_N, R_1].
%
%       The ranks may increase adaptively during the iterations, up to the
%       maximum ranks specified by opts.R.
%
%   Core
%       Core tensor of size
%
%           size(Core) = [L_1, L_2, ..., L_N].
%
%       Its dimensions correspond to the current inner/core ranks.
%
%   Out
%       Structure with diagnostic information:
%
%           Out.RSE
%               Relative step errors
%
%                   norm(X_k(:) - X_{k-1}(:)) / norm(X_{k-1}(:))
%
%               recorded over the executed iterations.
%
%           Out.RES_init
%               Initial relative residual computed from the randomly
%               initialized tensor-wheel representation.
%
% =========================================================================
% Implementation notes
%   This function keeps the external calling convention of inc_TW_TC.m, but
%   replaces the explicit construction of large least-squares design matrices
%   by matrix-free tensor-network contractions.
%
%   Factor updates are computed without explicitly constructing the full
%   GCrest matrix. Instead, the normal equations
%
%       H = A' * A,
%       b = A' * y
%
%   are assembled by contractions over the current tensor-wheel environment.
%
%   Core updates are likewise computed without explicitly forming the large
%   core subwheel/unfolding matrix A = Girest. The code contracts the
%   surrounding tensor-wheel network to assemble
%
%       H = A * A',
%       b = A * vec(X),
%
%   and solves the proximal regularized system
%
%       (H + rho * I) c = b + rho * c_prev,
%
%   where
%
%       c      = vec(Core),
%       c_prev = vec(Core)
%
%   immediately before the current core update.
%
%   Core updates are performed at the first iteration, after adaptive rank
%   growth, and according to the schedule determined by
%   opts.core_update_after and opts.core_update_every.
%
%   If opts.enforceOmega is true and Omega is nonempty, the observed entries
%   are projected back to the data tensor after each X-update:
%
%       X(Omega) = F(Omega).
%
%   This gives the usual tensor-completion behavior: observed entries are
%   fixed, while missing entries are estimated by the tensor-wheel model.
%   If opts.enforceOmega is false, the method instead behaves as a full-data
%   tensor reconstruction algorithm.
%
% =========================================================================
% -------------------- argument handling --------------------
% Valid calls:
%   fast_twd_cpu(F, opts)
%   fast_twd_cpu(F, Omega, opts)

if nargin == 2
    opts = Omega;
    Omega = [];
elseif nargin == 3
    % F, Omega, opts supplied explicitly
elseif nargin < 2
    error('fast_twd_cpu requires at least F and opts.');
else
    error('fast_twd_cpu accepts either (F, opts) or (F, Omega, opts).');
end

enforceOmega = ~isempty(Omega);

if isfield(opts,'enforceOmega')
    enforceOmega = opts.enforceOmega;
end

% -------------------- options --------------------
tol   = 1e-6;
maxit = 500;
rho   = 1e-3;
max_R = [];

% Core-update schedule (baseline Algorithm 2):
%   update at k==1, on rank growth, and every core_update_every after core_update_after.
core_update_after = 3;
core_update_every = 2;

% Initialization padding (baseline uses 2). Set opts.num_padarray=0 to disable.
num_padarray = 0;

if isfield(opts,'tol');   tol   = opts.tol;   end
if isfield(opts,'maxit'); maxit = opts.maxit; end
if isfield(opts,'rho');   rho   = opts.rho;   end
if isfield(opts,'R');     max_R = opts.R;     end
if isfield(opts,'core_update_after'); core_update_after = opts.core_update_after; end
if isfield(opts,'core_update_every'); core_update_every = opts.core_update_every; end
if isfield(opts,'num_padarray'); num_padarray = opts.num_padarray; end

if isempty(max_R)
    error('opts.R must be provided as [R; L] (outer; inner ranks).');
end

% -------------------- init --------------------
R = max(max_R - num_padarray, 2);

Ndim = ndims(F);
Nway = size(F);
X = F;

Factors_dims      = factor_dims(Nway, R);
Max_Factors_dims  = factor_dims(Nway, max_R);

rng('default');
G = cell(Ndim,1);
for i = 1:Ndim
    G{i} = rand(Factors_dims(i,:));
end
Core  = rand(R(2,:));
C_old = Core;

% Initial residuals
Y0 = cores_prod_single_tw(G,Core);
RES_init = norm(F(:)-Y0(:))/norm(Y0(:));

Out.RSE = zeros(1, maxit);
Out.RES_init = RES_init;
r_change = 0.0005;


% -------------------- main loop --------------------
for k = 1:maxit

    X_old = X;

    % ---- Update G_k, k=1,2,...,N ----
    for num = 1:Ndim
        dimsG = size(G{num});
        G2    = unfold(G{num}, dimsG, 2);

        % Exact matrix-free factor RHS and Gram (avoid forming GCrest explicitly).
        A_ten = tw_factor_rhs_dense_exact(X, G, Core, num);
        TempA = unfold(A_ten, dimsG, 2) + rho * G2;

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
        % b = A*vec(X) as a tensor of size(L1,...,LN)
        b_ten = tw_core_rhs_dense(X, G);
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

    % ---- Update X ----
    Xhat = cores_prod_single_tw(G, Core);
    X = (Xhat + rho*X_old) / (1 + rho);

    if enforceOmega && ~isempty(Omega)
       X(Omega) = F(Omega);
    end

    % ---- Convergence ----
    rse = norm(X(:) - X_old(:)) / max(1e-12, norm(X_old(:)));
    Out.RSE(k) = rse;

    if k == 1 || mod(k, 20) == 0
        fprintf('fast TWD (CPU): iter = %d   RSE=%.10f\n', k, rse);
    end
    if rse < tol
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

end

% ======================== helpers ========================

function A_ten = tw_factor_rhs_dense_exact(X, G, Core, n)
% Compute A_n = X_(n) * Q_n' exactly without forming Q_n = GCrest.
% Returns A_ten of size [R_n, I_n, L_n, R_{n+1}] so that
%   unfold(A_ten, size(G{n}), 2) == tenmat_sb(X,n) * GCrest'.
%
% Index/order is matched to the existing unfold(...,2) convention.

N = numel(G);
seq = [n+1:N, 1:n-1];

% Bring mode n first: [I_n, I_{n+1}, ..., I_{n-1}]
Xp = permute(X, [n, seq]);

% First factor contraction: contract I_{n+1}
% T layout after step 1:
%   [I_n, I_{n+2}, ..., I_{n-1}, R_{n+1}, L_{n+1}, R_{n+2}]
T = tensor_contraction(Xp, G{seq(1)}, 2, 2);

% Process the remaining factors in circular order.
for t = 2:numel(seq)
    j = seq(t);
    % Contract next physical mode and the carried outer rank.
    T = tensor_contraction(T, G{j}, [2, ndims(T)], [2, 1]);
end

% Contract all L_j, j ~= n, with the core; leave L_n open.
Cperm = permute(Core, [seq, n]);
T = tensor_contraction(T, Cperm, 3:(N+1), 1:(N-1));
% T layout is now [I_n, R_{n+1}, R_n, L_n]

A_ten = permute(T, [3, 1, 4, 2]);  % [R_n, I_n, L_n, R_{n+1}]
end


function B = tw_factor_gram_dense_exact(G, Core, n)
% Compute B_n = Q_n * Q_n' exactly without forming Q_n = GCrest.
% Output size is J_n x J_n where J_n = R_n * L_n * R_{n+1}.

N = numel(G);
seq = [n+1:N, 1:n-1];

Mk = cell(N,1);
for j = seq
    % Contract physical mode i_j between two copies of G{j}.
    Tj = tensor_contraction(G{j}, G{j}, 2, 2);
    % Tj dims: [R_j, L_j, R_{j+1}, R'_j, L'_j, R'_{j+1}]
    Mk{j} = permute(Tj, [1, 4, 2, 5, 3, 6]);
    % Mk{j}: [R_j, R'_j, L_j, L'_j, R_{j+1}, R'_{j+1}]
end

U = Mk{seq(1)};
for t = 2:numel(seq)
    j = seq(t);
    U = tensor_contraction(U, Mk{j}, [ndims(U)-1, ndims(U)], [1, 2]);
end
% U: [R_{n+1},R'_{n+1},L_seq(1),L'_seq(1),...,L_seq(end),L'_seq(end),R_n,R'_n]

Cperm = permute(Core, [seq, n]);

% Contract the unprimed L_seq modes with Core.
V = tensor_contraction(U, Cperm, 3:2:(2*N-1), 1:(N-1));
% V: [R_{n+1},R'_{n+1},L'_seq...,R_n,R'_n,L_n]

% Contract the primed L'_seq modes with a second copy of Core.
W = tensor_contraction(V, Cperm, 3:(N+1), 1:(N-1));
% W: [R_{n+1},R'_{n+1},R_n,R'_n,L_n,L'_n]

W = permute(W, [3, 5, 1, 4, 6, 2]);
% W: [R_n,L_n,R_{n+1},R'_n,L'_n,R'_{n+1}]

J = size(G{n},1) * size(G{n},3) * size(G{n},4);
B = reshape(W, J, J);
B = (B + B.') * 0.5;   % stabilize for chol
end


function b_ten = tw_core_rhs_dense(X, G)
% Compute b = A*vec(X) where vec(X) = A'*vec(C).
% Returns b as a tensor of size [L1 ... LN].
N = numel(G);

% First contraction: contract i1
T = tensor_contraction(G{1}, X, 2, 1); % [R1, L1, R2, I2..IN]

% Enforce canonical ordering: [R1, L1..L_{k-1}, Rk, Ik, I_{k+1}..IN]
for k = 2:N-1
    % current T dims: [R1, L1..L_{k-1}, Rk, Ik, I_{k+1}..IN]
    % contract (Rk,Ik) with G{k}(Rk,Ik,Lk,R{k+1})
    mode_Rk = 1 + (k-1) + 1; % 1 (R1) + (k-1) L-modes + Rk
    mode_Ik = mode_Rk + 1;
    T = tensor_contraction(T, G{k}, [mode_Rk, mode_Ik], [1, 2]);

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

T = tensor_contraction(T, G{N}, [mode_R1, mode_RN, mode_IN], [4, 1, 2]);
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
    T = tensor_contraction(G{k}, G{k}, 2, 2);
    % T dims: [Rk, Lk, R{k+1}, Rk', Lk', R{k+1}']
    % Reorder to [Rk, Rk', Lk, Lk', R{k+1}, R{k+1}']
    Mk{k} = permute(T, [1, 4, 2, 5, 3, 6]);
end

% Contract the ring over outer ranks (and their primed copies)
T = Mk{1}; % [R1,R1',L1,L1',R2,R2']
for k = 2:N
    % Contract last two modes (Rk,Rk') with first two modes of Mk{k}
    T = tensor_contraction(T, Mk{k}, [ndims(T)-1, ndims(T)], [1, 2]);
    % Result ends with [R{k+1}, R{k+1}']
end

% Close the ring: sum over (R1,R1') with (last R1,R1')
sz = size(T);
r1 = sz(1);
r1p = sz(2);

Ldims = sz(3:end-2); % [L1,L1',...,LN,LN']
Hten = zeros(Ldims);
idx = repmat({':'}, 1, ndims(T));
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
% increase the estimated rank until max_R
for j = 1:N
    G{j} = padarray(G{j}, rank_inc(j,:), rand(1), 'post');
end
Core = padarray(Core, rank_inc(:,3), rand(1), 'post');
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

function X = cores_prod_single_tw(G,C)

    Nc = length(G);
    X = tensorprod(G{1},C,3,1);
    X = permute(X,[4:Nc+2,1:3]);

    for n = 1:Nc-2
        X = tensorprod(X,G{n+1},[1,ndims(X)],[3,1]);
    end
    X = tensorprod(X,G{Nc},[1, 2, ndims(X)],[3, 4, 1]);

end

% this is for simulation data, randomly generate tensor dims 
function [Z] = factor_dims(Nway,r)
%
N = numel(Nway);
Z = zeros(N,4);  % 4th-order tensor
for i=1:N-1
    Z(i,:) = [r(1,i),Nway(i),r(2,i),r(1,i+1)];
end
    Z(N,:) = [r(1,N),Nway(N),r(2,N),r(1,1)];
end