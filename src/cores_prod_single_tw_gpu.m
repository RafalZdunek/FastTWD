function Xhatg = cores_prod_single_tw_gpu(Gg, Cg, Nway, opts)
%CORES_PROD_SINGLE_TW_GPU GPU-native skeleton for Tensor Wheel reconstruction.
%
%   Xhatg = cores_prod_single_tw_gpu(Gg, Cg)
%   Xhatg = cores_prod_single_tw_gpu(Gg, Cg, Nway)
%   Xhatg = cores_prod_single_tw_gpu(Gg, Cg, Nway, opts)
%
% Input:
%   Gg   - cell array, Gg{n} of size [R_n, I_n, L_n, R_{n+1}]
%   Cg   - core tensor of size [L_1, L_2, ..., L_N]
%   Nway - optional output size [I_1, ..., I_N]
%   opts - optional struct with fields:
%          .method        = 'auto' | 'generic' | 'serial-gemm' (default 'auto')
%          .check_sizes   = true/false (default true)
%          .force_gpu     = true/false (default true)
%          .gather_output = true/false (default false)
%
% Output:
%   Xhatg - reconstructed tensor of size Nway. If input is on GPU and
%           gather_output=false, output stays on GPU.
%
% Notes:
%   1) This is a GPU-native skeleton intended to keep the full reconstruction
%      on device.
%   2) The 'generic' path is exact and simple; it relies on a local tensor
%      contraction helper implemented via permute + reshape + mtimes.
%   3) The 'serial-gemm' path is a concrete placeholder for a future faster
%      specialized implementation. It currently falls back to the generic path.
%   4) The output order is [I_1, I_2, ..., I_N].
%
% Typical use inside the solver:
%   Xhatg = cores_prod_single_tw_gpu(Gg, Cg, size(Xg));
%
% -------------------------------------------------------------------------

    if nargin < 3 || isempty(Nway)
        Nway = infer_nway_from_factors(Gg);
    end
    if nargin < 4
        opts = struct();
    end

    method        = get_opt_local(opts, 'method', 'auto');
    check_sizes   = get_opt_local(opts, 'check_sizes', true);
    force_gpu     = get_opt_local(opts, 'force_gpu', true);
    gather_output = get_opt_local(opts, 'gather_output', false);

    N = numel(Gg);
    if N == 0
        error('cores_prod_single_tw_gpu: Gg must be a non-empty cell array.');
    end

    if check_sizes
        validate_tw_inputs_local(Gg, Cg, Nway);
    end

    use_gpu = isa(Cg, 'gpuArray') || any(cellfun(@(x) isa(x, 'gpuArray'), Gg));
    if force_gpu && ~use_gpu
        Cg = gpuArray(Cg);
        for n = 1:N
            Gg{n} = gpuArray(Gg{n});
        end
        use_gpu = true;
    end

    switch lower(method)
        case 'auto'
            % For now, auto -> generic exact path.
            Xhatg = reconstruct_generic_exact_gpu_local(Gg, Cg, Nway);

        case 'generic'
            Xhatg = reconstruct_generic_exact_gpu_local(Gg, Cg, Nway);

        case 'serial-gemm'
            % Concrete placeholder. Keep exactness by falling back to the
            % generic path until the specialized contraction chain is filled in.
            Xhatg = reconstruct_serial_gemm_placeholder_local(Gg, Cg, Nway);

        otherwise
            error('cores_prod_single_tw_gpu: unknown method "%s".', method);
    end

    if gather_output && use_gpu
        Xhatg = gather(Xhatg);
    end
end

%% =========================================================================
%% Generic exact GPU reconstruction
%% =========================================================================
function Xhatg = reconstruct_generic_exact_gpu_local(Gg, Cg, Nway)
% Exact GPU/CPU reconstruction that mirrors the original CPU helper exactly.
%
% Original reference implementation:
%   X = tensorprod(G{1},C,3,1);
%   X = permute(X,[4:Nc+2,1:3]);
%   for n = 1:Nc-2
%       X = tensorprod(X,G{n+1},[1,ndims(X)],[3,1]);
%   end
%   X = tensorprod(X,G{Nc},[1,2,ndims(X)],[3,4,1]);
%
% Here tensor_contract_local reproduces the same output-order convention
% as tensorprod: [A_uncontracted_modes, B_uncontracted_modes].

    Nc = numel(Gg);

    if Nc == 1
        % Degenerate 1-way case: contract the core mode with factor mode-3,
        % then trace the ring ranks [R1, R1].
        T = tensor_contract_local(Gg{1}, 3, Cg, 1);   % [R1, I1, R1]
        sz = size(T);
        if numel(sz) ~= 3 || sz(1) ~= sz(3)
            error('cores_prod_single_tw_gpu: invalid 1-way layout during ring closure.');
        end
        Xv = zeros(sz(2), 1, 'like', T);
        for rr = 1:sz(1)
            Xv = Xv + reshape(T(rr,:,rr), [], 1);
        end
        Xhatg = reshape(Xv, Nway);
        return
    end

    % Step 1: identical to CPU helper
    Xhatg = tensor_contract_local(Gg{1}, 3, Cg, 1);
    Xhatg = permute(Xhatg, [4:Nc+2, 1:3]);

    % Middle factors: identical contraction pattern to CPU helper
    for n = 1:Nc-2
        Xhatg = tensor_contract_local(Xhatg, [1, ndims(Xhatg)], Gg{n+1}, [3, 1]);
    end

    % Final factor closes the ring exactly as in CPU helper
    Xhatg = tensor_contract_local(Xhatg, [1, 2, ndims(Xhatg)], Gg{Nc}, [3, 4, 1]);

    % Ensure exact output shape
    Xhatg = reshape(Xhatg, Nway);
end

%% =========================================================================
%% Serial GEMM placeholder (future fast path)
%% =========================================================================
function Xhatg = reconstruct_serial_gemm_placeholder_local(Gg, Cg, Nway)
% Concrete placeholder for a future faster path.
%
% Intention for future implementation:
%   - maintain a rolling partial state,
%   - reshape to 2D / 3D pages,
%   - use mtimes / pagemtimes where possible,
%   - minimize generic permute-heavy contractions.
%
% For now, preserve exactness by calling the generic path.

    Xhatg = reconstruct_generic_exact_gpu_local(Gg, Cg, Nway);
end

%% =========================================================================
%% Local tensor contraction helper (GPU-safe)
%% =========================================================================
function C = tensor_contract_local(A, aModes, B, bModes)
%TENSOR_CONTRACT_LOCAL Contract tensor A with tensor B over specified modes.
%
% Supports a single mode or lists of modes. Works with cpuArray/gpuArray.
%
% Result layout:
%   [A_uncontracted_modes, B_uncontracted_modes]

    if isscalar(aModes), aModes = aModes(:).'; end
    if isscalar(bModes), bModes = bModes(:).'; end

    aModes = double(aModes);
    bModes = double(bModes);

    aSize = size(A);
    bSize = size(B);
    aNd   = ndims(A);
    bNd   = ndims(B);

    if numel(aModes) ~= numel(bModes)
        error('tensor_contract_local: number of contracted modes must match.');
    end

    aKeep = setdiff(1:aNd, aModes, 'stable');
    bKeep = setdiff(1:bNd, bModes, 'stable');

    aCtrSize = aSize(aModes);
    bCtrSize = bSize(bModes);
    if numel(aCtrSize) ~= numel(bCtrSize) || any(aCtrSize ~= bCtrSize)
        error('tensor_contract_local: contracted dimensions must match.');
    end

    Aperm = permute(A, [aKeep, aModes]);
    Bperm = permute(B, [bModes, bKeep]);

    aKeepSize = aSize(aKeep);
    bKeepSize = bSize(bKeep);

    if isempty(aKeepSize), aKeepSize = 1; end
    if isempty(bKeepSize), bKeepSize = 1; end

    Ka = prod(double(aKeepSize));
    Kc = prod(double(aCtrSize));
    Kb = prod(double(bKeepSize));

    Amat = reshape(Aperm, [Ka, Kc]);
    Bmat = reshape(Bperm, [Kc, Kb]);

    Cmat = Amat * Bmat;
    C = reshape(Cmat, [aKeepSize, bKeepSize]);
end

%% =========================================================================
%% Validation helpers
%% =========================================================================
function validate_tw_inputs_local(Gg, Cg, Nway)
    N = numel(Gg);
    if numel(Nway) ~= N
        error('cores_prod_single_tw_gpu: numel(Nway) must equal numel(Gg).');
    end

    cSize = size(Cg);
    if numel(cSize) ~= N
        % MATLAB may omit trailing singleton dimensions in size output.
        cSize = [cSize, ones(1, N - numel(cSize))];
    end

    for n = 1:N
        sz = size(Gg{n});
        if numel(sz) ~= 4
            error('cores_prod_single_tw_gpu: Gg{%d} must be 4-D [R_n, I_n, L_n, R_{n+1}].', n);
        end

        if sz(2) ~= Nway(n)
            error('cores_prod_single_tw_gpu: Gg{%d} physical mode mismatch. Expected %d, got %d.', ...
                n, Nway(n), sz(2));
        end

        if sz(3) ~= cSize(n)
            error('cores_prod_single_tw_gpu: Gg{%d} core mode mismatch. Expected L_%d=%d, got %d.', ...
                n, n, cSize(n), sz(3));
        end

        np1 = mod(n, N) + 1;
        szNext = size(Gg{np1});
        if sz(4) ~= szNext(1)
            error('cores_prod_single_tw_gpu: ring rank mismatch between Gg{%d} and Gg{%d}.', n, np1);
        end
    end
end

function Nway = infer_nway_from_factors(Gg)
    N = numel(Gg);
    Nway = zeros(1, N);
    for n = 1:N
        sz = size(Gg{n});
        if numel(sz) ~= 4
            error('cores_prod_single_tw_gpu: cannot infer Nway because Gg{%d} is not 4-D.', n);
        end
        Nway(n) = sz(2);
    end
end

function v = get_opt_local(s, name, defaultv)
    if isfield(s, name)
        v = s.(name);
    else
        v = defaultv;
    end
end
