clear; clc;

% -------------------------------------------------------------------------
% Repository path setup.
% This script can be run either from the repository root by
%   run(fullfile('experiments','run_sweep_outer_rank.m'))
% or directly from the experiments directory.
% -------------------------------------------------------------------------
scriptPath = mfilename('fullpath');
if isempty(scriptPath)
    scriptPath = which('run_sweep_outer_rank');
end
expDir  = fileparts(scriptPath);
rootDir = fileparts(expDir);

addpath(fullfile(rootDir, 'src'));
addpath(fullfile(rootDir, 'experiments'));
addpath(fullfile(rootDir, 'experiments', 'utils'));

baselineDir = fullfile(rootDir, 'third_party', 'Baseline_TW_TC');
if exist(baselineDir, 'dir')
    addpath(baselineDir);
else
    baselineDir = fullfile(rootDir, 'third_party', 'code_TWDec', 'Baseline_TW_TC');
    if exist(baselineDir, 'dir')
        addpath(baselineDir);
    else
        warning('Baseline TW directory was not found. Baseline runs may fail.');
    end
end


% =========================================================================
% Sweep over outer ranks R with two measurement modes:
%   MODE = 'runtime' -> runtime only (no peak-memory sampling)
%   MODE = 'memory'  -> peak memory only (runs algorithms, no timing)
%
% Fixed:
%   - N = 4
%   - I = [30 30 30 30]
%   - L = 2 (all modes)
%
% Sweep:
%   - R in [2 3 4 5 6 7 8 9 10 11 12] (uniform outer ranks)
%
% Methods:
%   1) baseline CPU:  inc_TW_TC
%   2) fast CPU: fast_twd_cpu
%   3) fast GPU: fast_twd_gpu
%
% Outputs:
%   - resultsTable (raw per-run)
%   - summaryTable (mean/std grouped by R and method)
%   - plot: runtime vs R (mean ± std) or peak memory vs R (mean ± std)
%   - MAT-file saved automatically
% =========================================================================

fprintf('Working folder: %s\n', pwd);
fprintf('MATLAB version: %s\n', version);

% ----------------------- fixed settings -----------------------
dims = [30 30 30 30];            % Mode sizes: I_1=I_2=I_3=I_4=30 (dense tensor 30^4 entries)
Lval = 2;                        % Inner rank value (uniform): L_n = 3 for all modes
R_list = 2:12;                    % Sweep range for outer ranks (uniform): R_n = R, R in {2,3,...,12}

N = length(dims);                % Tensor order: N 
L = Lval * ones(1,N);            % Inner ranks vector [L_1 ... L_N]

opts.maxit = 50;                 % Maximum number of PAM iterations
opts.rho   = 1;                  % Proximal/regularization parameter (stabilizes updates; larger => stronger damping)
opts.tol   = 1e-8;               % Stopping tolerance on relative change (RSE) between iterates

Omega = [];                      % Mask tensor for completion only; empty => dense decomposition (no missing entries)

% ----------------------- user controls -----------------------
MODE = 'runtime';        % 'runtime' | 'memory'
MC_runtime = 10;         % Monte Carlo runs for MODE='runtime'
MC_memory  = 10;         % Monte Carlo runs for MODE='memory' (memory-only tends to be long)

% Memory polling periods (seconds), used only in MODE='memory'.
% The same conservative settings as in run_sweep_inner_rank.m are used here.
period_base = 0.05;
period_fast = 0.03;
period_gpu  = 0.02;

% Memory measurement method:
%   'sampled' : timer-based sampling of MATLAB/GPU memory
%   'proxy'   : deterministic analytical proxy estimates
%   'hybrid'  : proxy for small tensors, sampled for larger tensors
MEM_METHOD = 'sampled';  % 'sampled' | 'proxy' | 'hybrid'
fprintf('Execution MODE: %s | MEM_METHOD: %s\n', MODE, MEM_METHOD);

% ----------------------- GPU availability -----------------------
hasGPU = (exist('gpuDeviceCount','file')==2) && gpuDeviceCount>0;
if hasGPU
    g = gpuDevice;
    fprintf('GPU detected: %s\n', g.Name);
    % Warm-up GPU (outside measured regions)
    a = gpuArray.ones(1024,1024,'double');
    wait(gpuDevice);
else
    fprintf('GPU not detected (or no Parallel Computing Toolbox). GPU method will be skipped.\n');
end

% ----------------------- results storage -----------------------
rows = {};
rowi = 0;
colnames = {'R','mc','method','time_s','RES','iters','peakMatlab_bytes','peakGpu_bytes','ok','err'};
t0 = datetime('now');

% ----------------------- sweep -----------------------
for rr = 1:numel(R_list)
    Rval = R_list(rr);
    R = Rval * ones(1,N);
    opts.R = [R; L];  

    fprintf('\n=============================================================\n');
    fprintf('Sweep R=%d, dims=[%s], N=%d, L=%d, MODE=%s\n', Rval, num2str(dims), N, Lval, MODE);
    fprintf('=============================================================\n');

    if strcmpi(MODE,'runtime')
        MC = MC_runtime;
    else
        MC = MC_memory;
    end

    for mc = 1:MC
        rng(100000 + 1000*Rval + mc, 'twister');

        % Generate synthetic dense TW tensor for this (R,L)
        Y = make_tw_tensor(dims, R, L);
        Ynorm = norm(Y(:));

        fprintf('\n-------------------------------------------------------------\n');
        fprintf('R = %d | Monte Carlo trial = %d / %d\n', Rval, mc, MC);
        fprintf('-------------------------------------------------------------\n');

        % baseline CPU
        [rowi, rows] = run_one('baseline', @()call_baseline(Y, Omega, opts), ...
            Y, Ynorm, false, period_base, MODE, MEM_METHOD, rowi, rows, Rval, mc, dims, Lval);
        
        % fast CPU
       [rowi, rows] = run_one('fast CPU', @()call_fast_cpu(Y, Omega, opts), ...
            Y, Ynorm, false, period_fast, MODE, MEM_METHOD, rowi, rows, Rval, mc, dims, Lval);

        % fast GPU
        if hasGPU
            [rowi, rows] = run_one('fast GPU', @()call_fast_gpu_full(Y, Omega, opts), ...
                Y, Ynorm, true, period_gpu, MODE, MEM_METHOD, rowi, rows, Rval, mc, dims, Lval);
        else
            rowi = rowi + 1;
            rows(rowi,:) = {Rval, mc, 'fast GPU', NaN, NaN, NaN, NaN, NaN, false, 'GPU not available'};
        end
    end
end

% ----------------------- build tables -----------------------
resultsTable = cell2table(rows, 'VariableNames', colnames);
resultsTable.method = categorical(resultsTable.method);
okMask = resultsTable.ok == true;

% Exclude the first Monte Carlo trial from summary statistics and plots.
% The first trial often includes MATLAB/GPU warm-up effects such as JIT
% compilation, memory-pool initialization, cache effects, and first gpuArray
% allocations. The raw resultsTable is kept unchanged.
excludeFirstMCForPlots = true;

if excludeFirstMCForPlots
    summaryInput = resultsTable(okMask & resultsTable.mc > 1, :);
else
    summaryInput = resultsTable(okMask, :);
end

summaryTable = groupsummary(summaryInput, ...
    {'R','method'}, ...
    {'mean','std','median',@iqr}, ...
    {'time_s','RES','iters','peakMatlab_bytes','peakGpu_bytes'});

summaryTable.Properties.VariableNames = strrep( ...
    summaryTable.Properties.VariableNames, ...
    'fun1_', ...
    'iqr_');

requiredMedianCols = {'median_peakMatlab_bytes','median_peakGpu_bytes'};
for k = 1:numel(requiredMedianCols)
    if ~ismember(requiredMedianCols{k}, summaryTable.Properties.VariableNames)
        error('Missing expected column "%s" in summaryTable.', requiredMedianCols{k});
    end
end

% Show a compact numerical summary in the Command Window.
fprintf('\nSummary table for outer-rank sweep:\n');
disp(summaryTable);

timestamp = datestr(t0,'yyyymmdd_HHMMSS');

scriptPath = mfilename('fullpath');
if isempty(scriptPath)
    scriptPath = which('run_sweep_outer_rank');
end
expDir  = fileparts(scriptPath);
rootDir = fileparts(expDir);

resultsRoot = fullfile(rootDir, 'results');
if ~exist(resultsRoot, 'dir')
    mkdir(resultsRoot);
end

runDir = fullfile(resultsRoot, sprintf('sweep_outer_rank_%s_%s', MODE, timestamp));
if ~exist(runDir, 'dir')
    mkdir(runDir);
end

figDir = fullfile(runDir, 'figures');
if ~exist(figDir, 'dir')
    mkdir(figDir);
end

outname = fullfile(runDir, sprintf('TW_final_sweep_R_%s_%s.mat', MODE, timestamp));

save(outname, 'resultsTable', 'summaryTable', 'MODE', 'MEM_METHOD', 'MC_runtime', 'MC_memory', ...
    'period_base','period_fast','period_gpu', 'R_list', 'dims', 'N', 'Lval', 't0', ...
    'excludeFirstMCForPlots', 'timestamp', 'runDir', 'figDir');

fprintf('\nSaved results to %s\n', outname);
fprintf('Saved figures to %s\n', figDir);

% ----------------------- plots -----------------------
if strcmpi(MODE,'runtime')
    plot_runtime_vs_R(summaryTable, R_list, figDir);
else
    plot_memory_vs_R(summaryTable, R_list, figDir);
end

if any(~isnan(summaryTable.mean_RES) & summaryTable.mean_RES > 0)
    plot_residual_vs_R(summaryTable, R_list, figDir);
end

if any(~isnan(summaryTable.mean_iters) & summaryTable.mean_iters > 0)
    plot_iters_vs_R(summaryTable, R_list, figDir);
end

% =========================================================================
% Local functions
% =========================================================================
function [rowi, rows] = run_one(methodName, fhandle, Y, Ynorm, trackGPU, period_s, MODE, MEM_METHOD, rowi, rows, Rval, ...
    mc, dims, Lval)

% Run a single method with runtime and/or memory measurement.
%
% MODE:
%   'runtime' : measure runtime only
%   'memory'  : measure memory only
%   'both'    : measure both runtime and memory
%
% MEM_METHOD:
%   'sampled' : sampled peak memory using memmon_start/memmon_stop
%   'proxy'   : deterministic proxy memory estimate
%   'hybrid'  : use proxy for selected small cases, sampled otherwise

ok = true;
err = '';

t = NaN;
RES = NaN;
iters = NaN;

peakMat = NaN;
peakGpu = NaN;

doTime = strcmpi(MODE,'runtime') || strcmpi(MODE,'both');
doMem  = strcmpi(MODE,'memory')  || strcmpi(MODE,'both');

useSampledMem = false;
useProxyMem   = false;

if doMem
    switch lower(MEM_METHOD)
        case 'sampled'
            useSampledMem = true;

        case 'proxy'
            useProxyMem = true;

        case 'hybrid'
            % Example hybrid rule.
            % Modify this condition if you want a different threshold.
            if Rval <= 3
                useProxyMem = true;
            else
                useSampledMem = true;
            end

        otherwise
            error('Unknown MEM_METHOD: %s. Use ''sampled'', ''proxy'', or ''hybrid''.', MEM_METHOD);
    end
end

mon = [];

if useSampledMem
    % Make dynamic field names safe, because methodName may contain spaces.
    key = matlab.lang.makeValidName(sprintf('%s_%d_%d', methodName, Rval, mc));

    if trackGPU
        g = gpuDevice;
        wait(g);
    end

    mon = memmon_start(trackGPU, key, period_s);
end

try
    if doTime
        if trackGPU
            g = gpuDevice;
            wait(g);
        end

        tic;
        [Yhat, Out] = fhandle();

        if trackGPU
            wait(g);
        end

        t = toc;
    else
        % memory-only: no timing
        [Yhat, Out] = fhandle();

        if trackGPU
            g = gpuDevice;
            wait(g);
        end
    end

    RES = norm(Y(:)-Yhat(:)) / Ynorm;
    iters = numel(Out.RSE);

catch ME
    ok = false;
    err = ME.message;
    Yhat = [];
    Out = struct('RSE',[]);
end

if useSampledMem
    stats = memmon_stop(mon);
    peakMat = stats.peakMatlabDelta;
    peakGpu = stats.peakGpuDelta;
end

if useProxyMem
    % Proxy placeholders.
    % These functions must exist elsewhere in the script if you use MEM_METHOD='proxy' or 'hybrid'.
    peakMat = proxy_peak_matlab_bytes(methodName, dims, Rval, Lval);
    peakGpu = proxy_peak_gpu_bytes(methodName, dims, Rval, Lval, trackGPU);
end

rowi = rowi + 1;
rows(rowi,:) = {Rval, mc, methodName, t, RES, iters, peakMat, peakGpu, ok, err};

end


% ========================= Deterministic memory proxy helpers =========================
function bytes = proxy_peak_matlab_bytes(methodName, dims, Rval, Lval)
% Deterministic proxy of peak MATLAB host memory in bytes.
%
% This is not a profiler. It gives a stable scale estimate for the memory
% footprint when timer-based sampled deltas are zero, NaN, or dominated by
% MATLAB/GPU memory-pool reuse. The formulas below use the dominant dense
% work-array sizes of the baseline and fast TW solvers, assuming double
% precision and uniform outer/inner ranks for the current sweep point.

N = numel(dims);
I_tot = prod(double(dims));
I = double(dims(1));

core_tot = double(Lval)^double(N);        % number of core entries, L^N
J = double(Rval)^2 * double(Lval);        % one TW factor slice/block scale, R*L*R

switch lower(char(methodName))
    case 'baseline'
        % Baseline inc_TW_TC can implicitly create large dense unfolded
        % subwheel/design-like arrays. Dominant terms: baseline work array,
        % core Gram term, and input/output dense tensors.
        bytes = 8 * (core_tot*I_tot + core_tot^2 + 2*I_tot);

    case 'fast cpu'
        % Matrix-free fast CPU solver: dominant host terms are compact
        % contraction RHS/Gram work arrays plus dense input/output tensors.
        bytes = 8 * (J*(I_tot/I) + core_tot^2 + 2*I_tot);

    case 'fast gpu'
        % Host side of the GPU solver is mainly dense host tensors plus the
        % small core/Gram-related terms. Device memory is estimated below.
        bytes = 8 * (2*I_tot + core_tot^2);

    otherwise
        bytes = NaN;
end

end


function bytes = proxy_peak_gpu_bytes(methodName, dims, Rval, Lval, trackGPU)
% Deterministic proxy of peak GPU device memory in bytes.
%
% Returns zero for CPU-only methods. For the GPU solver, the proxy includes
% dense device tensors and representative matrix-free contraction workspaces.

if ~trackGPU || ~strcmpi(char(methodName), 'fast GPU')
    bytes = 0;
    return;
end

N = numel(dims);
I_tot = prod(double(dims));
I = double(dims(1));

core_tot = double(Lval)^double(N);
J = double(Rval)^2 * double(Lval);

% Dominant device residents: input/output/reconstruction-sized tensors,
% compact contraction work arrays, core Gram-like terms, and small J-by-J
% workspaces used by factor updates.
bytes = 8 * (3*I_tot + J*(I_tot/I) + core_tot^2 + J^2);

end

function [Yhat, Out] = call_baseline(Y, Omega, opts)
% Wrapper for the baseline TW-TC implementation.
% The output format of inc_TW_TC may differ between package versions, so the
% wrapper first tries the dense/decomposition call with Omega and then falls
% back to the shorter syntax.

try
    [Yhat, ~, ~, Out] = inc_TW_TC(Y, Omega, opts);
catch
    [Yhat, ~, ~, Out] = inc_TW_TC(Y, opts);
end

end

function [Yhat, Out] = call_fast_cpu(Y, Omega, opts)
try
    [Yhat, ~, ~, Out] = fast_twd_cpu(Y, Omega, opts);
catch
    [Yhat, ~, ~, Out] = fast_twd_cpu(Y, opts);
end
end

function [Yhat, Out] = call_fast_gpu_full(Y, Omega, opts)
try
    [Yhat, ~, ~, Out] = fast_twd_gpu(Y, Omega, opts);
catch
    [Yhat, ~, ~, Out] = fast_twd_gpu(Y, opts);
end
end

function X = make_tw_tensor(dims, R, L)
N = numel(dims);
Core = max(0, randn(L));
G = cell(N,1);
for n = 1:N
    rn  = R(n);
    in  = dims(n);
    ln  = L(n);
    rnp = R(mod(n,N)+1);
    G{n} = max(0, randn(rn, in, ln, rnp));
end
X = cores_prod_single_tw(G, Core);
end

function plot_runtime_vs_R(summaryTable, R_list, figDir)

% Plot mean runtime vs R with shaded mean +/- std bands
figure('Name','Runtime vs R','Color','w');

fontSize = 16;

ax = gca;
ax.Color = 'w';
ax.XColor = 'k';
ax.YColor = 'k';
ax.GridColor = [0.5 0.5 0.5];
ax.MinorGridColor = [0.35 0.35 0.35];
ax.FontSize = fontSize;
ax.LineWidth = 1.2;

hold on;
grid on;
box on;

methods = {'baseline','fast CPU','fast GPU'};

% Color palette, consistent with I-, N-, and L-sweep plots
colors = [
    0.30 0.75 0.93   % baseline  - blue/cyan
    0.95 0.55 0.20   % fast CPU  - orange
    0.40 0.85 0.45   % fast GPU  - green
];

for mi = 1:numel(methods)
    m = methods{mi};
    c = colors(mi,:);

    mask = summaryTable.method == categorical({m});
    Tm = summaryTable(mask,:);

    if isempty(Tm)
        continue;
    end

    % Ensure order by R
    [~,ord] = ismember(R_list, Tm.R);
    ord(ord==0) = [];
    Tm = Tm(ord,:);

    if isempty(Tm)
        continue;
    end

    x = Tm.R;
    y = Tm.mean_time_s;
    s = Tm.std_time_s;

    validMean = isfinite(x) & isfinite(y) & y > 0;
    xMean = x(validMean);
    yMean = y(validMean);
    sMean = s(validMean);

    if isempty(xMean)
        continue;
    end

    % Draw shaded band only where std is strictly positive.
    % Zero STD values are ignored for the uncertainty band.
    validBand = isfinite(sMean) & (sMean > 0);

    if any(validBand)
        idxValid = find(validBand);
        splitPts = [0; find(diff(idxValid) > 1); numel(idxValid)];
        for kk = 1:numel(splitPts)-1
            segIdx = idxValid(splitPts(kk)+1 : splitPts(kk+1));

            if numel(segIdx) < 2
                continue;
            end

            xSeg = xMean(segIdx);
            ySeg = yMean(segIdx);
            sSeg = sMean(segIdx);
            
            yLow  = max(ySeg - sSeg, eps);
            yHigh = ySeg + sSeg;

            fill([xSeg; flipud(xSeg)], ...
                 [yLow; flipud(yHigh)], ...
                 c, ...
                 'FaceAlpha', 0.4, ...
                 'EdgeColor', 'none', ...
                 'HandleVisibility', 'off');
        end
    end

    plot(xMean, yMean, '-o', ...
        'DisplayName', m, ...
        'Color', c, ...
        'LineWidth', 1.5, ...
        'MarkerSize', 8, ...
        'MarkerFaceColor', c, ...
        'MarkerEdgeColor', 'k');
end

set(gca, 'YScale', 'log');
xticks(unique(R_list));
xtickformat('%d');

xlabel('Outer rank R (R_n=R)', ...
    'Color', 'k', ...
    'FontSize', fontSize + 2);
    
ylabel('Runtime (s): mean \pm 1 std', ...
    'Color', 'k', ...
    'FontSize', fontSize + 2);

title('Scalability: runtime vs outer rank R', ...
    'Color', 'k', ...
    'FontSize', fontSize + 4);

lgd = legend('Location','northwest');
lgd.TextColor = 'k';
lgd.Color = 'w';
lgd.FontSize = fontSize + 2;

save_current_figure(figDir, 'runtime_vs_R');

end

function plot_memory_vs_R(summaryTable, R_list, figDir)

% Plot peak memory vs R using median values only.
% Variability bands are intentionally omitted because sampled peak-memory
% measurements are affected by MATLAB/GPU memory-pool reuse and transient
% allocator effects, which can create misleading jump-like uncertainty regions.

figure('Name','Peak memory vs R','Color','w');

fontSize = 16;

ax = gca;
ax.Color = 'w';
ax.XColor = 'k';
ax.YColor = 'k';
ax.GridColor = [0.5 0.5 0.5];
ax.MinorGridColor = [0.35 0.35 0.35];
ax.FontSize = fontSize;
ax.LineWidth = 1.2;

hold on;
grid on;
box on;

toGB = @(x) x / 1024^3;

curves = {
    'baseline', 'median_peakMatlab_bytes', 'baseline (MATLAB)',       '-',  'o'
    'fast CPU', 'median_peakMatlab_bytes', 'fast CPU (MATLAB)',       '-',  's'
    'fast GPU', 'median_peakMatlab_bytes', 'fast GPU (MATLAB)',       '-',  '^'
    'fast GPU', 'median_peakGpu_bytes',    'fast GPU (GPU device)',   '--', 'd'
};


% Color palette, consistent with I-, N-, and L-sweep plots
colors = [
    0.30 0.75 0.93   % baseline MATLAB - blue/cyan
    0.95 0.55 0.20   % fast CPU MATLAB - orange
    0.40 0.85 0.45   % fast GPU MATLAB - green
    0.70 0.35 0.95   % fast GPU device - purple
];

for ci = 1:size(curves,1)

    methodName  = curves{ci,1};
    medianCol   = curves{ci,2};
    displayName = curves{ci,3};
    lineStyle   = curves{ci,4};
    markerStyle = curves{ci,5};
    c = colors(ci,:);

    % Skip this curve if the median column is not available
    if ~ismember(medianCol, summaryTable.Properties.VariableNames)
        warning('Skipping "%s": column "%s" is missing in summaryTable.', ...
            displayName, medianCol);
        continue;
    end

    mask = summaryTable.method == categorical({methodName});
    Tm = summaryTable(mask,:);

    if isempty(Tm)
        continue;
    end

    % Ensure order by R
    [~,ord] = ismember(R_list, Tm.R);
    ord(ord==0) = [];
    Tm = Tm(ord,:);

    if isempty(Tm)
        continue;
    end

    x = Tm.R;
    y = toGB(Tm.(medianCol));

    validMean = isfinite(x) & isfinite(y) & y > 0;
    xMean = x(validMean);
    yMean = y(validMean);

    if isempty(xMean)
        continue;
    end
    
    if strcmp(displayName, 'fast GPU (GPU device)')
        markerFaceColor = 'w';
    else
        markerFaceColor = c;
    end

    plot(xMean, yMean, ...
        'LineStyle', lineStyle, ...
        'Marker', markerStyle, ...
        'DisplayName', displayName, ...
        'Color', c, ...
        'LineWidth', 1.5, ...
        'MarkerSize', 8, ...
        'MarkerFaceColor', markerFaceColor, ...
        'MarkerEdgeColor', 'k');
end

set(gca, 'YScale', 'log');
xticks(unique(R_list));
xtickformat('%d');

xlabel('Outer rank R (R_n=R)', ...
    'Color', 'k', ...
    'FontSize', fontSize + 2);

ylabel('Peak memory increase (GB): median', ...
    'Color', 'k', ...
    'FontSize', fontSize + 2);

title('Scalability: peak memory vs outer rank R', ...
    'Color', 'k', ...
    'FontSize', fontSize + 4);

lgd = legend('Location','northwest');
lgd.TextColor = 'k';
lgd.Color = 'w';
lgd.FontSize = fontSize + 2;

save_current_figure(figDir, 'memory_vs_R');

end

function plot_residual_vs_R(summaryTable, R_list, figDir)

% Plot external residual error vs R as grouped bars with std whiskers
figure('Name','Residual error vs R','Color','w');

fontSize = 16;

ax = gca;
ax.Color = 'w';
ax.XColor = 'k';
ax.YColor = 'k';
ax.GridColor = [0.5 0.5 0.5];
ax.MinorGridColor = [0.35 0.35 0.35];
ax.FontSize = fontSize;
ax.LineWidth = 1.2;

hold on;
grid on;
box on;

methods = {'baseline','fast CPU','fast GPU'};

% Color palette, consistent with runtime and peak-memory plots
colors = [
    0.30 0.75 0.93   % baseline  - blue/cyan
    0.95 0.55 0.20   % fast CPU  - orange
    0.40 0.85 0.45   % fast GPU  - green
];

nR = numel(R_list);
nMethods = numel(methods);

Y = NaN(nR, nMethods);
S = NaN(nR, nMethods);

for mi = 1:nMethods
    m = methods{mi};

    mask = summaryTable.method == categorical({m});
    Tm = summaryTable(mask,:);

    if isempty(Tm)
        continue;
    end

    % Ensure order by R
    [~,ord] = ismember(R_list, Tm.R);
    validOrd = ord > 0;

    rows = ord(validOrd);
    cols = find(validOrd);

    Y(cols, mi) = Tm.mean_RES(rows);
    S(cols, mi) = Tm.std_RES(rows);
end

% Grouped bar plot
xCats = categorical(string(R_list), string(R_list), 'Ordinal', true);
b = bar(xCats, Y, 'grouped');

for mi = 1:nMethods
    b(mi).FaceColor = colors(mi,:);
    b(mi).EdgeColor = 'k';
    b(mi).LineWidth = 0.8;
    b(mi).DisplayName = methods{mi};
end

% Add std whiskers. Zero STD values are not drawn as whiskers.
for mi = 1:nMethods
    x = b(mi).XEndPoints;
    y = Y(:,mi);
    s = S(:,mi);

    valid = isfinite(x(:)) & isfinite(y(:)) & isfinite(s(:)) & y(:) > 0 & s(:) > 0;

    errorbar(x(valid), y(valid), s(valid), ...
        'k', ...
        'LineStyle', 'none', ...
        'LineWidth', 1.2, ...
        'CapSize', 10, ...
        'HandleVisibility', 'off');
end


xlabel('Outer rank R (R_n=R)', ...
    'Color', 'k', ...
    'FontSize', fontSize + 2);

ylabel('Residual error: mean \pm 1 std', ...
    'Color', 'k', ...
    'FontSize', fontSize + 2);

title('Scalability: residual error vs outer rank R', ...
    'Color', 'k', ...
    'FontSize', fontSize + 4);

lgd = legend('Location','northwest');
lgd.TextColor = 'k';
lgd.Color = 'w';
lgd.FontSize = fontSize + 2;

save_current_figure(figDir, 'residual_vs_R');

end

function plot_iters_vs_R(summaryTable, R_list, figDir)

% Plot number of iterations vs R as grouped bars with std whiskers
figure('Name','Iterations vs R','Color','w');

fontSize = 16;

ax = gca;
ax.Color = 'w';
ax.XColor = 'k';
ax.YColor = 'k';
ax.GridColor = [0.5 0.5 0.5];
ax.MinorGridColor = [0.35 0.35 0.35];
ax.FontSize = fontSize;
ax.LineWidth = 1.2;

hold on;
grid on;
box on;

methods = {'baseline','fast CPU','fast GPU'};

% Color palette, consistent with residual-error plots
colors = [
    0.30 0.75 0.93   % baseline  - blue/cyan
    0.95 0.55 0.20   % fast CPU  - orange
    0.40 0.85 0.45   % fast GPU  - green
];

nR = numel(R_list);
nMethods = numel(methods);

Y = NaN(nR, nMethods);
S = NaN(nR, nMethods);

for mi = 1:nMethods
    m = methods{mi};

    mask = summaryTable.method == categorical({m});
    Tm = summaryTable(mask,:);

    if isempty(Tm)
        continue;
    end

    % Ensure order by R
    [~,ord] = ismember(R_list, Tm.R);
    validOrd = ord > 0;

    rows = ord(validOrd);
    cols = find(validOrd);

    Y(cols, mi) = Tm.mean_iters(rows);
    S(cols, mi) = Tm.std_iters(rows);
end

% Grouped bar plot
xCats = categorical(string(R_list), string(R_list), 'Ordinal', true);
b = bar(xCats, Y, 'grouped');

for mi = 1:nMethods
    b(mi).FaceColor = colors(mi,:);
    b(mi).EdgeColor = 'k';
    b(mi).LineWidth = 0.8;
    b(mi).DisplayName = methods{mi};
end

% Add std whiskers. Zero STD values are not drawn as whiskers.
for mi = 1:nMethods
    x = b(mi).XEndPoints;
    y = Y(:,mi);
    s = S(:,mi);

    valid = isfinite(x(:)) & isfinite(y(:)) & isfinite(s(:)) & y(:) > 0 & s(:) > 0;

    errorbar(x(valid), y(valid), s(valid), ...
        'k', ...
        'LineStyle', 'none', ...
        'LineWidth', 1.2, ...
        'CapSize', 10, ...
        'HandleVisibility', 'off');
end


xlabel('Outer rank R (R_n=R)', ...
    'Color', 'k', ...
    'FontSize', fontSize + 2);

ylabel('Number of iterations: mean \pm 1 std', ...
    'Color', 'k', ...
    'FontSize', fontSize + 2);

title('Iterations vs outer rank R', ...
    'Color', 'k', ...
    'FontSize', fontSize + 4);

lgd = legend('Location','northwest');
lgd.TextColor = 'k';
lgd.Color = 'w';
lgd.FontSize = fontSize + 2;

save_current_figure(figDir, 'iterations_vs_R');

end

% ========================= Memory monitoring helpers =========================
function mon = memmon_start(trackGPU, key, period_s)
    
    global MEMMON;
    if isempty(MEMMON), MEMMON = struct(); end

    % Make key safe for use as MATLAB struct field name
    key = matlab.lang.makeValidName(key);

    initMatlab = NaN;
    if ispc
        try
            m = memory;
            initMatlab = double(m.MemUsedMATLAB);
        catch
        end
    end
    
    initGpu = NaN;
    if trackGPU
        try
            g = gpuDevice;
            initGpu = double(g.TotalMemory - g.AvailableMemory);
        catch
        end
    end

    MEMMON.(key) = struct( ...
        'trackGPU', logical(trackGPU), ...
        'initMatlab', initMatlab, ...
        'initGpu', initGpu, ...
        'peakMatlabDelta', 0, ...
        'peakGpuDelta', 0);
    
    if nargin < 3 || isempty(period_s)
        period_s = 0.5;
    end
    
    t = timer( ...
        'ExecutionMode','fixedSpacing', ...
        'Period',period_s, ...
        'BusyMode','drop', ...
        'TimerFcn',@(~,~)memmon_poll(key));
    
    start(t);
    mon = struct('timer',t,'key',key);
end

function memmon_poll(key)
global MEMMON;

if ispc && ~isnan(MEMMON.(key).initMatlab)
    try
        m = memory;
        used = double(m.MemUsedMATLAB);
        delta = max(0, used - MEMMON.(key).initMatlab);
        MEMMON.(key).peakMatlabDelta = max(MEMMON.(key).peakMatlabDelta, delta);
    catch
    end
end

if MEMMON.(key).trackGPU && ~isnan(MEMMON.(key).initGpu)
    try
        g = gpuDevice;
        used = double(g.TotalMemory - g.AvailableMemory);
        delta = max(0, used - MEMMON.(key).initGpu);
        MEMMON.(key).peakGpuDelta = max(MEMMON.(key).peakGpuDelta, delta);
    catch
    end
end
end

function stats = memmon_stop(mon)
global MEMMON;
try
    % Take one final synchronous sample. This prevents all-zero memory curves
    % when the algorithm finishes before the timer has a chance to fire.
    memmon_poll(mon.key);
catch
end
try
    stop(mon.timer); delete(mon.timer);
catch
end
stats = MEMMON.(mon.key);
end

function period = smooth_period_by_load(loadVal, loadMin, loadMax, periodMin, periodMax)

if loadMax <= loadMin
    alpha = 0;
else
    alpha = (double(loadVal) - double(loadMin)) / ...
            (double(loadMax) - double(loadMin));
end

alpha = max(0, min(1, alpha));

% Logarithmic interpolation gives smoother scaling over orders of magnitude.
period = exp(log(periodMin) + alpha * (log(periodMax) - log(periodMin)));

% MATLAB timer supports only millisecond-level precision.
period = round(period * 1000) / 1000;
period = max(period, 2e-3);

end


function save_current_figure(figDir, baseName)

if nargin < 1 || isempty(figDir)
    figDir = pwd;
end

if ~exist(figDir, 'dir')
    mkdir(figDir);
end

fig = gcf;

set(fig, 'InvertHardcopy', 'off');
set(fig, 'PaperPositionMode', 'auto');

% Hide interactive axes toolbar before exporting
axs = findall(fig, 'Type', 'axes');
for k = 1:numel(axs)
    try
        axs(k).Toolbar.Visible = 'off';
    catch
    end
end

drawnow;

pngFile = fullfile(figDir, [baseName '.png']);
epsFile = fullfile(figDir, [baseName '.eps']);

exportgraphics(fig, pngFile, 'Resolution', 300);

try
    exportgraphics(fig, epsFile, 'ContentType', 'vector');
catch
    print(fig, epsFile, '-depsc', '-painters');
end

fprintf('Saved figure: %s\n', pngFile);
fprintf('Saved figure: %s\n', epsFile);

end


