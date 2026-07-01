function [X_hat, G, Core, Out] = fast_twd_gpu(X, Omega, opts)

% fast_twd_gpu: GPU/CPU fast implementation of Tensor Wheel (TW) decomposition.
%
% This routine approximates an Nth-order tensor X by the TW representation:
%
%       X_hat = TW({G_k}_{k=1}^N; C),
%
% where the TW factors consist of:
%
%   - N fourth-order ring factors
%
%         G_k in R^{R_k x I_k x L_k x R_{k+1}},  k = 1,...,N,
%
%   - one Nth-order core factor
%
%         C in R^{L_1 x L_2 x ... x L_N}.
%
% The cyclic convention is R_{N+1}=R_1. The quantities R_k and L_k are the
% outer and inner ranks, respectively.
%
% -------------------------------------------------------------------------
% MODEL
% -------------------------------------------------------------------------
%
% This function implements the full iterative update scheme for updating
% the TW factors:
%
%   - updates the ring factors G_k, k = 1,...,N,
%   - periodically updates the core factor C,
%   - reconstructs the dense tensor approximation X_hat,
%   - updates the working tensor Xwork using a proximal/relaxed step,
%   - monitors convergence by an internal relative-change criterion.
%
% The implementation supports both GPU and CPU execution:
%
%   - ring-factors {G_k} and core-factor C updates:
%         GPU when opts.use_gpu=true, otherwise CPU;
%
%   - core-factor linear solve:
%         GPU by default when opts.core_solve_on_gpu=true,
%         with optional CPU fallback;
%
%   - dense reconstruction X_hat:
%         CPU reconstruction using a local helper based on
%         tensor_contraction_explicit, followed by transfer to the GPU
%         when required.
%
% -------------------------------------------------------------------------
% CALLING SYNTAX
% -------------------------------------------------------------------------
%
%   [X_hat,G,Core,Out] = fast_twd_gpu(X, opts)
%   [X_hat,G,Core,Out] = fast_twd_gpu(X, Omega, opts)
%
% Omega is accepted for compatibility with tensor-completion interfaces.
% In the current dense implementation, Omega is ignored.
%
% -------------------------------------------------------------------------
% INPUT ARGUMENTS
% -------------------------------------------------------------------------
%
% X
%   Dense input tensor to be approximated.
%
%   Size:
%       X is an Nth-order tensor:
%
%           X in R^{I_1 x I_2 x ... x I_N}.
%
%       Here:
%
%           N   = size(opts.R, 2),
%           I_k = size(X, k).
%
%       The tensor order is determined from opts.R so that trailing
%       physical dimensions of size one are preserved.
%
%   Type:
%       Numeric MATLAB array, typically double or single.
%       If opts.use_gpu=true, X is internally transferred to gpuArray.
%
% Omega
%   Optional compatibility argument for tensor-completion-style interfaces.
%
%   In this dense implementation:
%
%       Omega is ignored.
%
% opts
%   Structure containing TW-ranks and algorithmic options.
%
%   Required field:
%
%       opts.R
%           2 x N matrix of TW-ranks:
%
%               opts.R = [R_1 R_2 ... R_N
%                         L_1 L_2 ... L_N].
%
%           First row:
%               outer ranks R_k.
%
%           Second row:
%               inner ranks L_k.
%
%           Required size:
%               size(opts.R) = [2, N].
%
%           Cyclic convention:
%               R_{N+1} = R_1.
%
% -------------------------------------------------------------------------
% OUTPUT ARGUMENTS
% -------------------------------------------------------------------------
%
% X_hat
%   Dense reconstructed tensor.
%
%   Size:
%       same as X:
%
%           I_1 x I_2 x ... x I_N.
%
%   Type:
%       CPU array if opts.gather_output=true.
%       gpuArray if opts.gather_output=false and opts.use_gpu=true.
%
% G
%   Cell array of TW ring factors.
%
%   Size:
%       G is an N x 1 cell array.
%
%   Each entry:
%
%       G{k} has size
%
%           R_k x I_k x L_k x R_{k+1}.
%
%   Type:
%       CPU arrays if opts.gather_output=true.
%       gpuArray entries if opts.gather_output=false and opts.use_gpu=true.
%
% Core
%   TW core factor C.
%
%   Size:
%
%       L_1 x L_2 x ... x L_N.
%
%   Type:
%       CPU array if opts.gather_output=true.
%       gpuArray if opts.gather_output=false and opts.use_gpu=true.
%
% Out
%   Structure with convergence and timing diagnostics.
%
%   Main fields:
%
%       Out.RES_init
%           Initial relative residual
%
%               norm(X(:) - Y0(:)) / norm(Y0(:)),
%
%           where Y0 is the TW tensor reconstructed from the initial
%           factors and core.
%
%       Out.RSE
%           Internal relative-change history, typically of the form
%
%               norm(Xwork^{t+1} - Xwork^t)_F / norm(Xwork^t)_F.
%
%           This is the stopping criterion controlled by opts.tol.
%
%       Out.iterations
%           Number of PAM iterations executed before termination.
%
%       Out.converged
%           Logical flag indicating whether the relative-change stopping
%           criterion was satisfied.
%
%       Out.final_R
%           Final 2-by-N matrix of TW ranks:
%
%               Out.final_R(1,:) = [R_1, ..., R_N]   outer ranks,
%               Out.final_R(2,:) = [L_1, ..., L_N]   inner ranks.
%
%           The current GPU implementation uses the fixed ranks opts.R.
%
%       Out.did_core
%           Logical vector indicating whether the core factor C was updated
%           at a given iteration.
%
%       Out.time_factor
%           Time spent in ring-factor updates.
%
%       Out.time_core
%           Time spent in core-factor update.
%
%       Out.time_recon
%           Time spent in dense reconstruction.
%
%       Out.time_total
%           Total time per iteration.
%
%       Out.stop_iter
%           Final iteration index.
%
%       Out.settings
%           Parsed solver settings.
%
%   Important:
%       Out.RSE is an internal convergence measure. It is not necessarily the
%       same quantity as the reconstruction residual:
%
%           RES = norm(X(:) - X_hat(:)) / norm(X(:)).
%
% -------------------------------------------------------------------------
% OPTIONS
% -------------------------------------------------------------------------
%
% Required:
%
%   R                  - 2 x N matrix of TW-ranks [R; L]
%
% Optional:
%
%   rho                - proximal parameter, default 1e-3
%   tol                - stopping tolerance for internal relative change,
%                        default 1e-6
%   maxit              - maximum number of iterations, default 500
%   seed               - nonnegative integer seed used for reproducible
%                        initialization of the TW factors and core with
%                        the Mersenne Twister generator, default 0
%   core_update_after  - first iteration after which scheduled core updates
%                        are allowed, default 3
%   core_update_every  - period of core updates after core_update_after,
%                        default 2
%   relax              - relaxation parameter for ring-factor and core-factor
%                        updates, default 1
%   precision          - 'double' (default), 'single', or 'mixed'
%   use_gpu            - true (default); if false, CPU execution is used
%   gather_output      - true (default); gather X_hat, G, and Core to CPU before
%                        returning
%   enable_core        - true (default); enable scheduled core-factor updates
%   enable_symmetrize  - true (default); symmetrize Gram matrices before
%                        linear solves
%   core_solve_on_gpu  - true (default); if false, the core linear system is
%                        built on GPU but solved on CPU
%   check_every        - convergence-check period, default 1
%   verbose_every      - console logging period, default 20
%
% Precision modes:
%
%   'double'
%       Store and compute working arrays in double precision.
%
%   'single'
%       Store and compute working arrays in single precision.
%
%   'mixed'
%       Store selected GPU working arrays in single precision while using
%       double precision for selected linear solves or CPU fallback paths.
%
% -------------------------------------------------------------------------
% REQUIRED / OPTIONAL PROJECT HELPERS
% -------------------------------------------------------------------------
%
% Required helper on the MATLAB path:
%
%   tensor_contraction_explicit
%
% Dense TW reconstruction is performed by a local helper contained in this
% file. No external reconstruction helper is required.
%
% -------------------------------------------------------------------------
% NOTES
% -------------------------------------------------------------------------
%
%   - This routine targets TW decomposition of a dense tensor.
%   - Omega is currently ignored. 
%   - The implementation is designed to reproduce the mathematical update
%     logic of the CPU fast TWD solver while accelerating ring-factor and
%     core-factor subproblems on GPU.
%   - If opts.use_gpu=false, the same algorithmic logic is executed on CPU.   
% =========================================================================
%
% -------------------------------------------------------------------------
% Parse input arguments
% Supported call forms:
%   fast_twd_gpu(X, opts)
%   fast_twd_gpu(X, Omega, opts)
% -------------------------------------------------------------------------

if nargin < 2
    error('fast_twd_gpu:MissingOptions', ...
        ['fast_twd_gpu requires at least X and opts. ', ...
         'Supported calls are:\n', ...
         '  [X_hat,G,Core,Out] = fast_twd_gpu(X, opts)\n', ...
         '  [X_hat,G,Core,Out] = fast_twd_gpu(X, Omega, opts)']);

elseif nargin == 2
    % Two-argument call: the second input is opts.
    opts = Omega;

elseif nargin == 3
    % Three-argument call: Omega is accepted for interface compatibility
    % and ignored in the current dense implementation.

else
    error('fast_twd_gpu:TooManyInputs', ...
        ['Supported calls are fast_twd_gpu(X, opts) and ', ...
         'fast_twd_gpu(X, Omega, opts).']);
end

if ~isstruct(opts) || ~isscalar(opts)
    error('fast_twd_gpu:InvalidOptions', 'opts must be a scalar structure.');
end

if isa(X, 'gpuArray')
    error('fast_twd_gpu:GpuArrayInput', ...
        ['X must be supplied as a CPU array. The solver transfers X ', ...
         'to the GPU internally when opts.use_gpu is true.']);
end

if isempty(X) || ~isnumeric(X) || ~isreal(X) || ...
        ~(isa(X, 'double') || isa(X, 'single'))
    error('fast_twd_gpu:InvalidInputTensor', ...
        'X must be a nonempty real tensor of class single or double.');
end

if ~isfield(opts, 'R') || isempty(opts.R)
    error('fast_twd_gpu:MissingRanks', ...
        'Missing required field opts.R. Expected opts.R to be a 2 x N matrix of TW-ranks [R; L].');
end

%% ------------------------- options & defaults ---------------------------
p = parse_opts_gpu(opts);

tol                = p.tol;
maxit              = p.maxit;
rho                = p.rho;
seed               = p.seed;
core_update_after  = p.core_update_after;
core_update_every  = p.core_update_every;
relax              = p.relax;
check_every        = p.check_every;
verbose_every      = p.verbose_every;
precision_mode     = p.precision;
use_gpu            = p.use_gpu;
gather_output      = p.gather_output;
enable_core        = p.enable_core;
enable_symmetrize  = p.enable_symmetrize;
core_solve_on_gpu   = p.core_solve_on_gpu;

%% ---------------------- tensor-wheel dimensions -------------------------
RL = opts.R;

if ~isnumeric(RL) || ~isreal(RL) || ~ismatrix(RL) || ...
        size(RL,1) ~= 2 || size(RL,2) < 2
    error('fast_twd_gpu:InvalidRankSize', ...
        ['opts.R must be a real 2-by-N matrix of outer and inner ', ...
         'TW ranks, with N >= 2.']);
end

if any(~isfinite(RL(:))) || ...
        any(RL(:) < 1) || ...
        any(RL(:) ~= fix(RL(:)))
    error('fast_twd_gpu:InvalidRanks', ...
        'All entries of opts.R must be finite positive integers.');
end

% Determine the logical tensor order from opts.R. This preserves possible
% trailing physical dimensions of size one.
Ndim = size(RL, 2);

Nway = ones(1, Ndim);
for n = 1:Ndim
    Nway(n) = size(X, n);
end

% Reject additional non-singleton physical dimensions not represented
% by opts.R.
sizeX = size(X);
if numel(sizeX) > Ndim && any(sizeX(Ndim+1:end) ~= 1)
    error('fast_twd_gpu:InconsistentTensorOrder', ...
        ['The number of physical modes of X is inconsistent with ', ...
         'the number of columns in opts.R.']);
end

L = RL(2,:);
Factors_dims = factor_dims(Nway, RL);

%% ------------------------- option validation ----------------------------

validateattributes(tol, {'numeric'}, ...
    {'scalar', 'real', 'finite', 'positive'}, ...
    mfilename, 'opts.tol');

validateattributes(maxit, {'numeric'}, ...
    {'scalar', 'real', 'finite', 'integer', 'positive'}, ...
    mfilename, 'opts.maxit');

validateattributes(rho, {'numeric'}, ...
    {'scalar', 'real', 'finite', 'positive'}, ...
    mfilename, 'opts.rho');

validateattributes(seed, {'numeric'}, ...
    {'scalar', 'real', 'finite', 'integer', 'nonnegative'}, ...
    mfilename, 'opts.seed');

if seed > 2^32 - 1
    error('fast_twd_gpu:InvalidSeed', ...
        'opts.seed must not exceed 2^32-1.');
end

validateattributes(core_update_after, {'numeric'}, ...
    {'scalar', 'real', 'finite', 'integer', 'nonnegative'}, ...
    mfilename, 'opts.core_update_after');

validateattributes(core_update_every, {'numeric'}, ...
    {'scalar', 'real', 'finite', 'integer', 'positive'}, ...
    mfilename, 'opts.core_update_every');

validateattributes(relax, {'numeric'}, ...
    {'scalar', 'real', 'finite', 'positive'}, ...
    mfilename, 'opts.relax');

validateattributes(check_every, {'numeric'}, ...
    {'scalar', 'real', 'finite', 'integer', 'positive'}, ...
    mfilename, 'opts.check_every');

validateattributes(verbose_every, {'numeric'}, ...
    {'scalar', 'real', 'finite', 'integer', 'positive'}, ...
    mfilename, 'opts.verbose_every');

% Accept either a character vector or a scalar string.
if isstring(precision_mode)
    if ~isscalar(precision_mode)
        error('fast_twd_gpu:InvalidPrecision', ...
            'opts.precision must be a scalar string or character vector.');
    end
    precision_mode = char(precision_mode);
end

if ~ischar(precision_mode) || ...
        ~ismember(lower(precision_mode), {'double', 'single', 'mixed'})
    error('fast_twd_gpu:InvalidPrecision', ...
        ['opts.precision must be ''double'', ''single'', ', ...
         'or ''mixed''.']);
end

precision_mode = lower(precision_mode);

logicalNames = { ...
    'use_gpu', ...
    'gather_output', ...
    'enable_core', ...
    'enable_symmetrize', ...
    'core_solve_on_gpu'};

logicalValues = { ...
    use_gpu, ...
    gather_output, ...
    enable_core, ...
    enable_symmetrize, ...
    core_solve_on_gpu};

for q = 1:numel(logicalNames)
    if ~islogical(logicalValues{q}) || ~isscalar(logicalValues{q})
        error('fast_twd_gpu:InvalidLogicalOption', ...
            'opts.%s must be a logical scalar.', logicalNames{q});
    end
end

%% -------------------------- device / dtype ------------------------------
[dtype, core_cpu_dtype] = get_dtypes_from_precision(precision_mode);

if use_gpu
    if exist('gpuDeviceCount','file') ~= 2 || gpuDeviceCount == 0
        error('No compatible GPU / Parallel Computing Toolbox not available. Set opts.use_gpu=false to run on CPU.');
    end
end

Xwork = cast(X, dtype);
if use_gpu
    Xwork = gpuArray(Xwork);
end

%% ----------------------------- init -------------------------------------
rng(seed, 'twister');
Gwork = cell(Ndim,1);
for n = 1:Ndim
    Gwork{n} = rand_gpu_or_cpu(Factors_dims(n,:), dtype, use_gpu);
end
Cwork = rand_gpu_or_cpu(L, dtype, use_gpu);

Out = struct();
Out.RSE            = nan(maxit,1);
Out.did_core       = false(maxit,1);
Out.time_factor    = zeros(maxit,1);
Out.time_core      = zeros(maxit,1);
Out.time_recon     = zeros(maxit,1);
Out.time_total     = zeros(maxit,1);
Out.stop_iter      = maxit;
Out.settings       = p;

% Maintain a CPU copy of the core for dense reconstruction, scheduling,
% and the optional CPU core-solve path.
Core_cpu = gather_if_needed(Cwork, use_gpu);
C_old_cpu = Core_cpu;

% Initial reconstruction and relative residual.
G0_cpu = cell(Ndim,1);
for n = 1:Ndim
    G0_cpu{n} = gather_if_needed(Gwork{n}, use_gpu);
end

X0_cpu = gather_if_needed(Xwork, use_gpu);
Y0_cpu = reconstruct_cpu_from_factors(G0_cpu, Core_cpu, Nway);

Out.RES_init = norm(X0_cpu(:) - Y0_cpu(:)) / norm(Y0_cpu(:));
converged = false;

%% --------------------------- main loop ----------------------------------
for k = 1:maxit
    t_total = tic;
    Xwork_old = Xwork;
    G_cpu_cache = [];

    %% ---------------------- factor updates -----------------------------
    t_factor = tic;
    for num = 1:Ndim
        dimsG = Factors_dims(num,:);
        G2 = unfold_fast(Gwork{num}, dimsG, 2); % [I_n x J]

        % Compute the exact matrix-free factor right-hand side B_n and
        % Gram matrix H_n without explicitly forming the design matrix Q_n.
        B_ten = tw_factor_rhs_dense_exact_generic(Xwork, Gwork, Cwork, num);

        TempA = unfold_fast(B_ten, dimsG, 2) + rho * G2;

        H_n = tw_factor_gram_dense_exact_generic(Gwork, Cwork, num);
        H_n = add_diag_inplace(H_n, rho);

        if enable_symmetrize
            H_n = 0.5 * (H_n + H_n.');
        end

        Sol = solve_right_spd(TempA, H_n);

        if relax ~= 1
            Sol = (1-relax) * G2 + relax * Sol;
        end

        Gwork{num} = fold_fast(Sol, dimsG, 2);
    end
    Out.time_factor(k) = toc(t_factor);

    %% ----------------------- core update -------------------------------
    do_core = false;
    if enable_core
        do_core = (k == 1) || (numel(Core_cpu) > numel(C_old_cpu)) || ...
                  (k > core_update_after && mod(k, core_update_every) == 0);
    end
    Out.did_core(k) = do_core;

    if do_core
        t_core = tic;

        % Exact Core update on the active device (GPU when use_gpu=true).
        b_ten = tw_core_rhs_dense_exact_generic(Xwork, Gwork);
        b_vec = b_ten(:);
        H = tw_core_gram_dense_exact_generic(Gwork);

        nC = numel(Cwork);
        H(1:nC+1:end) = H(1:nC+1:end) + rho;
        rhs = b_vec + rho * Cwork(:);

        if enable_symmetrize
            H = 0.5 * (H + H.');
        end

        if core_solve_on_gpu || ~use_gpu
            c_vec = solve_left_spd(H, rhs);
            Cnew_work = reshape(c_vec, size(Cwork));
        else
            % Optional fallback: build exact system on GPU, solve on CPU.
            H_cpu = cast(gather_if_needed(H, use_gpu), core_cpu_dtype);
            rhs_cpu = cast(gather_if_needed(rhs, use_gpu), core_cpu_dtype);
            c_vec_cpu = solve_left_spd(H_cpu, rhs_cpu);
            Cnew_work = cast(reshape(c_vec_cpu, size(Core_cpu)), dtype);
            if use_gpu
                Cnew_work = gpuArray(Cnew_work);
            end
        end

        if relax ~= 1
            Cnew_work = (1-relax) * Cwork + relax * Cnew_work;
        end

        Cwork = Cnew_work;
        Core_cpu = cast(gather_if_needed(Cwork, use_gpu), core_cpu_dtype);
        C_old_cpu = Core_cpu;

        Out.time_core(k) = toc(t_core);
    else
        Out.time_core(k) = 0;
    end

    %% --------------------- reconstruction / X update -------------------
    t_recon = tic;

    % Reconstruct the dense tensor on the CPU using the local helper based on
    % tensor_contraction_explicit. Gather the current factors once per iteration.
    if isempty(G_cpu_cache)
        G_cpu_cache = cell(Ndim,1);
        for n = 1:Ndim
            G_cpu_cache{n} = gather_if_needed(Gwork{n}, use_gpu);
        end
    end

    Xhat_cpu = reconstruct_cpu_from_factors(G_cpu_cache, Core_cpu, Nway);
    Xhat = cast(Xhat_cpu, dtype);
    if use_gpu
        Xhat = gpuArray(Xhat);
    end

    Xwork = (Xhat + rho * Xwork_old) / (1 + rho);
    Out.time_recon(k) = toc(t_recon);

    %% ---------------------- convergence / logging ----------------------
    Out.time_total(k) = toc(t_total);

    do_check = (k == 1) || (mod(k, check_every) == 0) || (k == maxit);
    if do_check
        tiny = zeros(1, 'like', Xwork) + 1e-12;
        denom = max(tiny, norm(Xwork_old(:)));
        rse = gather_if_needed(norm(Xwork(:) - Xwork_old(:)) / denom, use_gpu);
        Out.RSE(k) = double(rse);

        if (k == 1) || (mod(k, verbose_every) == 0)
            fprintf('fast TWD (GPU): iter = %d   RSE=%.10f\n', k, double(rse));
        end

        if rse < tol
            converged = true;
            Out.stop_iter = k;
            break;
        end
        
    end
end

%% ---------------------------- finalize ----------------------------------
last_valid = find(~isnan(Out.RSE), 1, 'last');
if isempty(last_valid)
    last_valid = Out.stop_iter;
end
Out.RSE         = Out.RSE(1:last_valid);
Out.did_core    = Out.did_core(1:last_valid);
Out.time_factor = Out.time_factor(1:last_valid);
Out.time_core   = Out.time_core(1:last_valid);
Out.time_recon  = Out.time_recon(1:last_valid);
Out.time_total  = Out.time_total(1:last_valid);

% Final diagnostics.
Out.iterations = k;
Out.converged  = converged;

% The current GPU implementation uses fixed ranks opts.R.
Out.final_R = RL;

if gather_output
    X_hat    = gather_if_needed(Xwork, use_gpu);
    G    = cellfun(@(x) gather_if_needed(x, use_gpu), Gwork, 'UniformOutput', false);
    Core = gather_if_needed(Cwork, use_gpu);
else
    X_hat = Xwork;
    G = Gwork;
    Core = Cwork;
end

end

%% =========================================================================
%%                              OPTIONS
%% =========================================================================
function p = parse_opts_gpu(opts)
    p = struct();
    p.tol                = get_opt(opts, 'tol', 1e-6);
    p.maxit              = get_opt(opts, 'maxit', 500);
    p.rho                = get_opt(opts, 'rho', 1e-3);
    p.seed               = get_opt(opts, 'seed', 0);
    p.core_update_after  = get_opt(opts, 'core_update_after', 3);
    p.core_update_every  = get_opt(opts, 'core_update_every', 2);
    p.relax              = get_opt(opts, 'relax', 1.0);
    p.check_every        = get_opt(opts, 'check_every', 1);
    p.verbose_every      = get_opt(opts, 'verbose_every', 20);
    p.precision          = get_opt(opts, 'precision', 'double');
    p.use_gpu            = get_opt(opts, 'use_gpu', true);
    p.gather_output      = get_opt(opts, 'gather_output', true);
    p.enable_core        = get_opt(opts, 'enable_core', true);
    p.enable_symmetrize  = get_opt(opts, 'enable_symmetrize', true);
    p.core_solve_on_gpu  = get_opt(opts, 'core_solve_on_gpu', true);
end

function v = get_opt(s, name, defaultv)
    if isfield(s, name)
        v = s.(name);
    else
        v = defaultv;
    end
end

function [dtype, core_cpu_dtype] = get_dtypes_from_precision(mode)
    switch lower(mode)
        case 'double'
            dtype = 'double';
            core_cpu_dtype = 'double';
        case 'single'
            dtype = 'single';
            core_cpu_dtype = 'single';
        case 'mixed'
            dtype = 'single';
            core_cpu_dtype = 'double';
        otherwise
            error('Unknown precision mode: %s', mode);
    end
end

%% =========================================================================
%%                         DEVICE / NUMERICS HELPERS
%% =========================================================================
function A = rand_gpu_or_cpu(sz, dtype, use_gpu)
    A = cast(rand(sz), dtype);
    if use_gpu
        A = gpuArray(A);
    end
end

function y = gather_if_needed(x, use_gpu)
    if use_gpu
        y = gather(x);
    else
        y = x;
    end
end

function M = add_diag_inplace(M, alpha)
    J = size(M,1);
    M(1:J+1:end) = M(1:J+1:end) + alpha;
end

function X_hat = solve_right_spd(A, B)
% Solve X_hat = A / B where B is SPD.
    J = size(B,1);
    if J ~= size(B,2)
        error('solve_right_spd: B must be square.');
    end

    [L,p] = chol(B, 'lower');
    if p == 0
        X_hat = (L' \ (L \ A'))';
    else
        tiny = zeros(1, 'like', B) + 1e-12;
        B(1:J+1:end) = B(1:J+1:end) + tiny;
        [L,p] = chol(B, 'lower');
        if p == 0
            X_hat = (L' \ (L \ A'))';
        else
            X_hat = A / B;
        end
    end
end

function x = solve_left_spd(B, rhs)
% Solve B*x = rhs where B is SPD.
    J = size(B,1);
    if J ~= size(B,2)
        error('solve_left_spd: B must be square.');
    end

    [L,p] = chol(B, 'lower');
    if p == 0
        x = L' \ (L \ rhs);
    else
        tiny = zeros(1, 'like', B) + 1e-12;
        B(1:J+1:end) = B(1:J+1:end) + tiny;
        [L,p] = chol(B, 'lower');
        if p == 0
            x = L' \ (L \ rhs);
        else
            x = B \ rhs;
        end
    end
end

%% =========================================================================
%%                          FAST FOLD / UNFOLD
%% =========================================================================
function W = unfold_fast(W, dim, i)
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

function W = fold_fast(W, dim, i)
    nd = numel(dim);
    if nd <= 1
        W = reshape(W, dim);
        return
    end
    ord = [i, 1:i-1, i+1:nd];
    W = reshape(W, dim(ord));
    if i ~= 1
        W = permute(W, [2:i, 1, i+1:nd]);
    end
end

%% =========================================================================
%%                        FACTOR UPDATE HELPERS
%% =========================================================================
function B_ten = tw_factor_rhs_dense_exact_generic(X_hat, G, Core, n)
% Compute the exact factor right-hand side
%
%     B_n = X_hat_(n) * Q_n^T
%
% without explicitly forming Q_n.

    N = numel(G);
    seq = [n+1:N, 1:n-1];

    T = X_hat;
    labT = labels_I(N);

    % Contract all physical modes except I_n and all ring ranks except the
    % two ranks adjacent to G_n.
    for t = 1:numel(seq)
        j = seq(t);
        [T, labT] = labelled_contract_common(T, labT, G{j}, labels_G(j,N));
    end

    % Contract all inner L modes except L_n against Core.
    [T, labT] = labelled_contract_common(T, labT, Core, labels_L(N));

    wanted = {label_R(n), label_I(n), label_L(n), label_R(next_idx(n,N))};
    B_ten = labelled_permute(T, labT, wanted);
end

function H_n = tw_factor_gram_dense_exact_generic(G, Core, n)
% Compute the exact factor Gram matrix
%
%     H_n = Q_n * Q_n^T
%
% without explicitly forming Q_n.
    N = numel(G);
    seq = [n+1:N, 1:n-1];

    Mk = cell(N,1);
    labMk = cell(N,1);
    for j = seq
        % Contract only the physical index I_j between two copies of G_j.
        [Mk{j}, labMk{j}] = labelled_contract( ...
            G{j}, labels_G(j,N), G{j}, labels_Gp(j,N), ...
            {label_I(j)}, {label_Ip(j)});
    end

    T = Mk{seq(1)};
    labT = labMk{seq(1)};
    for t = 2:numel(seq)
        j = seq(t);
        [T, labT] = labelled_contract_common(T, labT, Mk{j}, labMk{j});
    end

    % Contract unprimed and primed inner modes of the remaining network with
    % two copies of the Core. This leaves exactly the variables of G_n and
    % G_n' in the order selected below.
    [T, labT] = labelled_contract_common(T, labT, Core, labels_L(N));
    [T, labT] = labelled_contract_common(T, labT, Core, labels_Lp(N));

    wanted = {label_R(n), label_L(n), label_R(next_idx(n,N)), ...
              label_Rp(n), label_Lp(n), label_Rp(next_idx(n,N))};
    W = labelled_permute(T, labT, wanted);

    J = size(G{n},1) * size(G{n},3) * size(G{n},4);
    H_n = reshape(W, J, J);
end

%% =========================================================================
%%                     CORE UPDATE HELPERS (GPU/CPU-EXACT)
%% =========================================================================
function b_ten = tw_core_rhs_dense_exact_generic(X_hat, G)
% Exact Core RHS for arbitrary tensor order N.
    N = numel(G);
    T = X_hat;
    labT = labels_I(N);

    % Contract the whole tensor-wheel ring with X_hat over all physical indices
    % and all circular R ranks. The remaining labels are L_1,...,L_N.
    for k = 1:N
        [T, labT] = labelled_contract_common(T, labT, G{k}, labels_G(k,N));
    end

    b_ten = labelled_permute(T, labT, labels_L(N));
end

function H = tw_core_gram_dense_exact_generic(G)
% Exact Core Gram for arbitrary tensor order N.
    N = numel(G);
    L = zeros(1,N);

    Mk = cell(N,1);
    labMk = cell(N,1);
    for k = 1:N
        L(k) = size(G{k},3);
        [Mk{k}, labMk{k}] = labelled_contract( ...
            G{k}, labels_G(k,N), G{k}, labels_Gp(k,N), ...
            {label_I(k)}, {label_Ip(k)});
    end

    T = Mk{1};
    labT = labMk{1};
    for k = 2:N
        [T, labT] = labelled_contract_common(T, labT, Mk{k}, labMk{k});
    end

    Hten = labelled_permute(T, labT, [labels_L(N), labels_Lp(N)]);
    H = reshape(Hten, prod(L), prod(L));   
end

%% =========================================================================
%%                    LABELLED TENSOR CONTRACTION HELPERS
%% =========================================================================
function [Z, labZ] = labelled_contract_common(X_hat, labX, Y, labY)
% Contract all symbolic labels occurring in both tensors.
%
% The logical tensor orders are passed explicitly to
% tensor_contraction_explicit so that trailing singleton dimensions
% associated with unit TW ranks are preserved. The result is subsequently
% reshaped to the exact logical shape implied by the surviving labels.
    X_hat = normalize_tensor_to_labels(X_hat, labX);
    Y = normalize_tensor_to_labels(Y, labY);
    sx = size_with_labels(X_hat, labX);
    sy = size_with_labels(Y, labY);

    [~, ix, iy] = intersect_stable_labels(labX, labY);
    if isempty(ix)
    Z = tensor_contraction_explicit(X_hat, Y, [], [], numel(labX), numel(labY));

    labZ = [labX, labY];
    else
    Z = tensor_contraction_explicit( ...
        X_hat, Y, ix, iy, ...
        numel(labX), numel(labY));

    labZ = [labX(setdiff_stable(1:numel(labX), ix)), ...
            labY(setdiff_stable(1:numel(labY), iy))];
    end
    
    expected = expected_sizes_for_labels(labZ, labX, sx, labY, sy);
    Z = reshape_to_label_shape(Z, expected, labZ);
end

function [Z, labZ] = labelled_contract(X_hat, labX, Y, labY, contractX, contractY)
% Contract explicit label lists contractX and contractY.
% Logical tensor orders are supplied explicitly so that unit-rank
% dimensions are preserved.
    X_hat = normalize_tensor_to_labels(X_hat, labX);
    Y = normalize_tensor_to_labels(Y, labY);
    sx = size_with_labels(X_hat, labX);
    sy = size_with_labels(Y, labY);

    ix = label_positions(labX, contractX);
    iy = label_positions(labY, contractY);

    Z = tensor_contraction_explicit(X_hat, Y, ix, iy, numel(labX), numel(labY));

    labZ = [labX(setdiff_stable(1:numel(labX), ix)), ...
            labY(setdiff_stable(1:numel(labY), iy))];

    expected = expected_sizes_for_labels(labZ, labX, sx, labY, sy);
    Z = reshape_to_label_shape(Z, expected, labZ);
end

function X_hat = normalize_tensor_to_labels(X_hat, lab)
% Normalize an array to the logical tensor order represented by lab.
% Internal singleton dimensions are preserved. Only omitted trailing
% singleton dimensions are restored explicitly.

k = numel(lab);

if k == 0
    if numel(X_hat) ~= 1
        error('normalize_tensor_to_labels:NonScalar', ...
            ['Expected a scalar tensor for an empty label list, ', ...
             'but received %d elements.'], numel(X_hat));
    end

    X_hat = reshape(X_hat, 1, 1);
    return;
end

% MATLAB represents every vector using at least two physical dimensions.
% For one symbolic label, all elements form a single logical mode.
if k == 1
    X_hat = reshape(X_hat, numel(X_hat), 1);
    return;
end

actualSize = size(X_hat);

% Additional dimensions are permitted only if they are trailing singleton
% dimensions omitted or retained by MATLAB's physical representation.
if numel(actualSize) > k && any(actualSize(k+1:end) ~= 1)
    error('normalize_tensor_to_labels:ExtraNonSingletonDims', ...
        ['Tensor has %d logical labels but additional non-singleton ', ...
         'dimensions: size=[%s], labels={%s}.'], ...
        k, num2str(actualSize), strjoin(lab, ','));
end

logicalSize = ones(1, k);
for d = 1:k
    logicalSize(d) = size(X_hat, d);
end

if prod(logicalSize) ~= numel(X_hat)
    error('normalize_tensor_to_labels:ElementMismatch', ...
        ['Cannot reshape tensor of size [%s] to the logical shape ', ...
         'associated with labels {%s}.'], ...
        num2str(actualSize), strjoin(lab, ','));
end

X_hat = reshape(X_hat, logicalSize);
end

function Y = labelled_permute(X_hat, labX, wanted)
% Permute tensor X_hat from labX order into wanted order.
% Normalize first because permute() requires ORDER to cover every actual dimension.
    X_hat = normalize_tensor_to_labels(X_hat, labX);
    if numel(labX) ~= numel(wanted)
        error('labelled_permute:LabelCountMismatch', ...
            'Expected %d output labels, but tensor currently has %d labels. Current={%s}, wanted={%s}.', ...
            numel(wanted), numel(labX), strjoin(labX, ','), strjoin(wanted, ','));
    end
    ord = label_positions(labX, wanted);
    if isequal(ord, 1:numel(ord))
        Y = X_hat;
    else
        Y = permute(X_hat, ord);
    end
end

function pos = label_positions(lab, wanted)
    pos = zeros(1,numel(wanted));
    for q = 1:numel(wanted)
        hit = find(strcmp(lab, wanted{q}), 1, 'first');
        if isempty(hit)
            error('label_positions:MissingLabel', ...
                'Missing label "%s". Available labels: {%s}.', wanted{q}, strjoin(lab, ','));
        end
        pos(q) = hit;
    end
end

function sx = size_with_labels(X_hat, lab)
% Return exactly one dimension per symbolic label.
    X_hat = normalize_tensor_to_labels(X_hat, lab);
    sx = size(X_hat);
    k = numel(lab);
    if k == 0
        sx = [];
    elseif numel(sx) < k
        sx = [sx, ones(1, k-numel(sx))];
    else
        sx = sx(1:k);
    end
end

function expected = expected_sizes_for_labels(labZ, labX, sx, labY, sy)
% Infer expected dimensions of a contraction result from surviving labels.
    expected = zeros(1, numel(labZ));
    for q = 1:numel(labZ)
        ix = find(strcmp(labX, labZ{q}), 1, 'first');
        if ~isempty(ix)
            expected(q) = sx(ix);
            continue;
        end
        iy = find(strcmp(labY, labZ{q}), 1, 'first');
        if ~isempty(iy)
            expected(q) = sy(iy);
            continue;
        end
        error('expected_sizes_for_labels:MissingLabel', ...
            'Cannot infer size for label "%s".', labZ{q});
    end
end

function Z = reshape_to_label_shape(Z, expected, labZ)
% Reshape the contraction result to the exact logical shape implied by
% the surviving labels. This restores trailing singleton dimensions that
% MATLAB may omit from the physical size representation while preserving
% the output order: remaining X labels followed by remaining Y labels.
    if isempty(expected)
        if numel(Z) ~= 1
            error('reshape_to_label_shape:ElementMismatch', ...
                'Expected scalar result, got %d elements.', numel(Z));
        end
        Z = reshape(Z, 1, 1);
        return;
    end

    if numel(Z) ~= prod(expected)
        error('reshape_to_label_shape:ElementMismatch', ...
            'Cannot reshape contraction result of size [%s] to expected labelled size [%s], labels={%s}.', ...
            num2str(size(Z)), num2str(expected), strjoin(labZ, ','));
    end

   if isscalar(expected)
      Z = reshape(Z, expected, 1);
   else
      Z = reshape(Z, expected);
   end
end


function [common, ix, iy] = intersect_stable_labels(a, b)
% Return common labels and their positions while preserving the order of a.

maxMatches = numel(a);

common = cell(1, maxMatches);
ix = zeros(1, maxMatches);
iy = zeros(1, maxMatches);

numMatches = 0;

for i = 1:numel(a)
    j = find(strcmp(b, a{i}), 1, 'first');

    if ~isempty(j)
        numMatches = numMatches + 1;

        common{numMatches} = a{i};
        ix(numMatches) = i;
        iy(numMatches) = j;
    end
end

% Remove unused preallocated entries.
common = common(1:numMatches);
ix = ix(1:numMatches);
iy = iy(1:numMatches);
end

function out = setdiff_stable(a, b)
    out = a(~ismember(a,b));
end

function c = labels_I(N)
    c = cell(1,N);
    for k = 1:N, c{k} = label_I(k); end
end

function c = labels_L(N)
    c = cell(1,N);
    for k = 1:N, c{k} = label_L(k); end
end

function c = labels_Lp(N)
    c = cell(1,N);
    for k = 1:N, c{k} = label_Lp(k); end
end

function c = labels_G(k,N)
    c = {label_R(k), label_I(k), label_L(k), label_R(next_idx(k,N))};
end

function c = labels_Gp(k,N)
    c = {label_Rp(k), label_Ip(k), label_Lp(k), label_Rp(next_idx(k,N))};
end

function k2 = next_idx(k,N)
    k2 = k + 1;
    if k2 > N, k2 = 1; end
end

function s = label_I(k),  s = sprintf('I%d', k);  end
function s = label_Ip(k), s = sprintf('Ip%d', k); end
function s = label_L(k),  s = sprintf('L%d', k);  end
function s = label_Lp(k), s = sprintf('Lp%d', k); end
function s = label_R(k),  s = sprintf('R%d', k);  end
function s = label_Rp(k), s = sprintf('Rp%d', k); end

%% =========================================================================
%%                         CPU RECONSTRUCTION HELPER
%% =========================================================================
function Xhat_cpu = reconstruct_cpu_from_factors(G_cpu, Core_cpu, ~)
%RECONSTRUCT_CPU_FROM_FACTORS Reconstruct the dense TW tensor on the CPU.
%
% The reconstruction uses explicitly supplied logical tensor orders so
% that trailing singleton dimensions associated with unit TW ranks are
% preserved.
    Xhat_cpu = cores_prod_single_tw_explicit(G_cpu, Core_cpu);
end

function X_hat = cores_prod_single_tw_explicit(G, C)
%CORES_PROD_SINGLE_TW_EXPLICIT Reconstruct a dense TW tensor.
%
% The logical tensor orders are passed explicitly to
% tensor_contraction_explicit. This preserves trailing singleton
% dimensions when some outer or inner TW ranks are equal to one.

Nc = numel(G);

% After the first contraction, the intermediate tensor has Nc+2 logical
% dimensions. This number must not be inferred with ndims because MATLAB
% omits trailing singleton dimensions.
logicalNdimsX = Nc + 2;

% Contract L_1 between G{1} and the core C.
%
% G{1}: [R_1, I_1, L_1, R_2]
% C:    [L_1, L_2, ..., L_N]
%
% Result:
%       [R_1, I_1, R_2, L_2, ..., L_N]
X_hat = tensor_contraction_explicit( ...
    G{1}, C, 3, 1, 4, Nc);

% Reorder the intermediate tensor to:
%
%       [L_2, ..., L_N, R_1, I_1, R_2]
X_hat = permute(X_hat, [4:logicalNdimsX, 1:3]);

% Successively contract L_{n+1} and the connecting outer rank R_{n+1}.
for n = 1:Nc-2
    X_hat = tensor_contraction_explicit( ...
        X_hat, G{n+1}, ...
        [1, logicalNdimsX], [3, 1], ...
        logicalNdimsX, 4);
end

% Close the TW ring by contracting:
%
%   L_N with mode 3 of G{N},
%   R_1 with mode 4 of G{N},
%   R_N with mode 1 of G{N}.
X_hat = tensor_contraction_explicit( ...
    X_hat, G{Nc}, ...
    [1, 2, logicalNdimsX], [3, 4, 1], ...
    logicalNdimsX, 4);
end

%% =========================================================================
%%                            FACTOR DIMENSIONS
%% =========================================================================
function Z = factor_dims(Nway, r)
%FACTOR_DIMS  Project-compatible local implementation.
% r is 2xN with rows [R; L]
% Z(n,:) = [R_n, I_n, L_n, R_{n+1}]
    N = numel(Nway);
    if size(r,1) ~= 2 || size(r,2) ~= N
        error('factor_dims: r must be 2xN, with r = [R; L].');
    end
    Z = zeros(N,4);
    for i = 1:N-1
        Z(i,:) = [r(1,i), Nway(i), r(2,i), r(1,i+1)];
    end
    Z(N,:) = [r(1,N), Nway(N), r(2,N), r(1,1)];
end
