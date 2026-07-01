clear; clc;
addpath(fullfile(fileparts(pwd), 'third_party', 'Baseline_TW_TC'));

% =========================================================================
% Sweep over inner ranks L with two measurement modes:
%   MODE = 'runtime' -> runtime only (no peak-memory sampling)
%   MODE = 'memory'  -> peak memory only (runs algorithms, no timing)
%
% Fixed:
%   - I = [30 30 30 30]
%   - Outer ranks R_n = 2 for all modes
%
% Sweep:
%   - Inner ranks L_n = L for all modes, L in [2 3 4 5 6 7]
% =========================================================================

fprintf('Working folder: %s\n', pwd);
fprintf('MATLAB version: %s\n', version);

% ----------------------- user controls -----------------------
MODE = 'memory';       % 'runtime' | 'memory' | 'both'
MC_runtime = 10;        % Monte Carlo runs for MODE='runtime'
MC_memory  = 10;        % Monte Carlo runs for MODE='memory' (can be large for big L)

% ----------------------- fixed settings -----------------------
dims = [30 30 30 30];   % Mode sizes: I_1=I_2=I_3=I_4=30 (dense tensor 30^4 entries)
Rval = 2;               % Outer rank value (uniform): R_n = 2 for all modes
L_list = 2:7;           % Sweep range for inner ranks (uniform): L_n = L, L in {2,...,7}

opts.maxit = 50;        % Maximum number of PAM iterations
opts.rho   = 1;         % Proximal/regularization parameter (stabilizes updates; larger => stronger damping)
opts.tol   = 1e-8;      % Stopping tolerance on relative change (RSE) between iterates

Omega = [];             % Mask tensor for completion only; empty => dense decomposition (no missing entries)

I_fixed = dims(1);
N = length(dims);
R = Rval * ones(1,N);

% Memory polling periods (seconds), used only in MODE='memory'
period_base = 0.05;
period_fast = 0.03;
period_gpu  = 0.02;

% Memory measurement method:
%   'sampled' : timer-based sampling of MATLAB/GPU memory
%   'proxy'   : deterministic analytical proxy estimates
%   'hybrid'  : proxy for small L values, sampled for larger L values
MEM_METHOD = 'sampled';  % 'sampled' | 'proxy' | 'hybrid'

fprintf('Execution MODE: %s | MEM_METHOD: %s\n', MODE, MEM_METHOD);

% ----------------------- GPU availability -----------------------
hasGPU = (exist('gpuDeviceCount','file')==2) && gpuDeviceCount>0;
if hasGPU
    g = gpuDevice;
    fprintf('GPU detected: %s\n', g.Name);
    a = gpuArray.ones(1024,1024,'double'); 
    wait(g);
else
    fprintf('GPU not detected (or no Parallel Computing Toolbox). GPU method will be skipped.\n');
end

% Optional: one warm-up call for GPU-full (excluded from timing)
if hasGPU
    try
        Lw = 2*ones(1,N);
        opts_w = opts; opts_w.R = [R; Lw];
        Yw = make_tw_tensor(dims, R, Lw);
        fast_twd_gpu(Yw, Omega, opts_w);
        wait(g);
    catch
    end
end

% ----------------------- results storage -----------------------
rows = {};
rowi = 0;
colnames = {'L','mc','method','time_s','RES','iters','peakMatlab_bytes','peakGpu_bytes','ok','err'};
t0 = datetime('now');

% ----------------------- sweep -----------------------
for ll = 1:numel(L_list)
    Lval = L_list(ll);
    L = Lval * ones(1,N);
    opts.R = [R; L];

    fprintf('\n=============================================================\n');
    fprintf('Sweep L=%d, dims=[%s], N=%d, I=%d, R=%d, MODE=%s\n', Lval, num2str(dims), N, I_fixed, Rval, MODE);
    fprintf('=============================================================\n');

    if strcmpi(MODE,'runtime')
        MC = MC_runtime;
    else
        MC = MC_memory;
    end

    for mc = 1:MC
        rng(200000 + 1000*Lval + mc, 'twister');

        Y = make_tw_tensor(dims, R, L);
        Ynorm = norm(Y(:));

        fprintf('\n-------------------------------------------------------------\n');
        fprintf('L = %d | Monte Carlo trial = %d / %d\n', Lval, mc, MC);
        fprintf('-------------------------------------------------------------\n');

        [rowi, rows] = run_one('baseline', @()call_baseline(Y, Omega, opts), ...
            Y, Ynorm, false, period_base, MODE, MEM_METHOD, rowi, rows, Lval, mc, dims, Rval);

        [rowi, rows] = run_one('fast CPU', @()call_fast_cpu(Y, Omega, opts), ...
            Y, Ynorm, false, period_fast, MODE, MEM_METHOD, rowi, rows, Lval, mc, dims, Rval);

        if hasGPU
           [rowi, rows] = run_one('fast GPU', @()call_fast_gpu_full(Y, Omega, opts), ...
                            Y, Ynorm, true, period_gpu, MODE, MEM_METHOD, rowi, rows, Lval, mc, dims, Rval);
        else
            rowi = rowi + 1;
            rows(rowi,:) = {Lval, mc, 'fast GPU', NaN, NaN, NaN, NaN, NaN, false, 'GPU not available'};
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
    {'L','method'}, ...
    {'mean','std','median',@iqr}, ...
    {'time_s','RES','iters','peakMatlab_bytes','peakGpu_bytes'});

% Rename function-handle output columns from fun1_* to iqr_*
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

timestamp = datestr(t0,'yyyymmdd_HHMMSS');

scriptPath = mfilename('fullpath');
if isempty(scriptPath)
    scriptPath = which('run_sweep_inner_rank');
end

expDir  = fileparts(scriptPath);
rootDir = fileparts(expDir);

resultsRoot = fullfile(rootDir, 'results');
if ~exist(resultsRoot, 'dir')
    mkdir(resultsRoot);
end

runDir = fullfile(resultsRoot, sprintf('sweep_inner_rank_%s_%s', MODE, timestamp));
if ~exist(runDir, 'dir')
    mkdir(runDir);
end

figDir = fullfile(runDir, 'figures');
if ~exist(figDir, 'dir')
    mkdir(figDir);
end

outname = fullfile(runDir, sprintf('TW_final_sweep_L_%s_%s.mat', MODE, timestamp));

save(outname, 'resultsTable', 'summaryTable', 'MODE', 'MEM_METHOD', 'MC_runtime', 'MC_memory', ...
    'period_base','period_fast','period_gpu', 'L_list', 'dims', 'N', 'Rval', 't0', ...
    'excludeFirstMCForPlots', 'timestamp', 'runDir', 'figDir');

fprintf('\nSaved results to %s\n', outname);
fprintf('Saved figures to %s\n', figDir);

% ----------------------- plots -----------------------
if strcmpi(MODE,'runtime')
    plot_runtime_vs_L(summaryTable, L_list, figDir);
else
    plot_memory_vs_L(summaryTable, L_list, figDir);
end

if any(~isnan(summaryTable.mean_RES) & summaryTable.mean_RES > 0)
    plot_residual_vs_L(summaryTable, L_list, figDir);
end

if any(~isnan(summaryTable.mean_iters) & summaryTable.mean_iters > 0)
    plot_iters_vs_L(summaryTable, L_list, figDir);
end

% =========================================================================
% Local functions
% =========================================================================
function [rowi, rows] = run_one(methodName, fhandle, Y, Ynorm, trackGPU, period_s, MODE, MEM_METHOD, rowi, rows, Lval, mc, dims, Rval)

ok = true;
err = '';

t = NaN;
RES = NaN;
iters = NaN;

peakMat = NaN;
peakGpu = NaN;

doTime = strcmpi(MODE,'runtime') || strcmpi(MODE,'both');
doMem  = strcmpi(MODE,'memory')  || strcmpi(MODE,'both');

if ~doTime && ~doMem
    error('Unknown MODE: %s. Use ''runtime'', ''memory'', or ''both''.', MODE);
end

useSampledMem = false;
useProxyMem   = false;

if doMem
    switch lower(char(MEM_METHOD))
        case 'sampled'
            useSampledMem = true;

        case 'proxy'
            useProxyMem = true;

        case 'hybrid'
            % For small inner ranks, timer-based sampled deltas are often
            % zero or strongly affected by MATLAB/GPU memory pools.
            if Lval <= 4
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
    key = matlab.lang.makeValidName(sprintf('%s_L%d_mc%d', methodName, Lval, mc));

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
        [Yhat, Out] = fhandle();

        if trackGPU
            g = gpuDevice;
            wait(g);
        end
    end

    RES = norm(Y(:)-Yhat(:)) / Ynorm;

    if isfield(Out,'RSE') && ~isempty(Out.RSE)
        iters = numel(Out.RSE);
    else
        iters = NaN;
    end

catch ME
    ok = false;
    err = ME.message;
    fprintf('%s failed: %s\n', methodName, err);
    Yhat = [];
    Out = struct('RSE',[]);
end

if useSampledMem && ~isempty(mon)
    stats = memmon_stop(mon);
    peakMat = stats.peakMatlabDelta;
    peakGpu = stats.peakGpuDelta;
end

if useProxyMem
    peakMat = proxy_peak_matlab_bytes(methodName, dims, Rval, Lval);
    peakGpu = proxy_peak_gpu_bytes(methodName, dims, Rval, Lval, trackGPU);
end

rowi = rowi + 1;
rows(rowi,:) = {Lval, mc, methodName, t, RES, iters, peakMat, peakGpu, ok, err};

end

function [Yhat, Out] = call_baseline(Y, Omega, opts)
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

function plot_runtime_vs_L(summaryTable, L_list, figDir)

% Plot mean runtime vs L with shaded mean +/- std bands
figure('Name','Runtime vs L','Color','w');

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

% Color palette, consistent with I-, N-, and R-sweep plots
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

    % Ensure order by L
    [~,ord] = ismember(L_list, Tm.L);
    ord(ord==0) = [];
    Tm = Tm(ord,:);

    if isempty(Tm)
        continue;
    end

    x = Tm.L;
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
xticks(unique(L_list));
xtickformat('%d');

xlabel('Inner rank L (L_n=L)', ...
    'Color', 'k', ...
    'FontSize', fontSize + 2);

ylabel('Runtime (s): mean \pm 1 std', ...
    'Color', 'k', ...
    'FontSize', fontSize + 2);

title('Scalability: runtime vs inner rank L', ...
    'Color', 'k', ...
    'FontSize', fontSize + 4);

lgd = legend('Location','northwest');
lgd.TextColor = 'k';
lgd.Color = 'w';
lgd.FontSize = fontSize + 2;

save_current_figure(figDir, 'runtime_vs_L');

end

function plot_memory_vs_L(summaryTable, L_list, figDir)

% Plot peak memory vs L using median values only.
% Variability bands are intentionally omitted because sampled peak-memory
% measurements are affected by MATLAB/GPU memory-pool reuse and transient
% allocator effects, which can create misleading jump-like uncertainty regions.
figure('Name','Peak memory vs L','Color','w');

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

% {method name, median column, display name, line style, marker}
curves = {
    'baseline', 'median_peakMatlab_bytes', 'baseline (MATLAB)',       '-',  'o'
    'fast CPU', 'median_peakMatlab_bytes', 'fast CPU (MATLAB)',       '-',  's'
    'fast GPU', 'median_peakMatlab_bytes', 'fast GPU (MATLAB)',       '-',  '^'
    'fast GPU', 'median_peakGpu_bytes',    'fast GPU (GPU device)',   '--', 'd'
};

% Color palette, consistent with I-, N-, and R-sweep plots
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

    % Ensure order by L
    [~,ord] = ismember(L_list, Tm.L);
    ord(ord==0) = [];
    Tm = Tm(ord,:);

    if isempty(Tm)
        continue;
    end

    x = Tm.L;
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
xticks(unique(L_list));
xtickformat('%d');

xlabel('Inner rank L (L_n=L)', ...
    'Color', 'k', ...
    'FontSize', fontSize + 2);

ylabel('Peak memory increase (GB): median', ...
    'Color', 'k', ...
    'FontSize', fontSize + 2);

title('Scalability: peak memory vs inner rank L', ...
    'Color', 'k', ...
    'FontSize', fontSize + 4);

lgd = legend('Location','northwest');
lgd.TextColor = 'k';
lgd.Color = 'w';
lgd.FontSize = fontSize + 2;

save_current_figure(figDir, 'memory_vs_L');

end

function plot_residual_vs_L(summaryTable, L_list, figDir)

% Plot external residual error vs L as grouped bars with std whiskers
figure('Name','Residual error vs L','Color','w');

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

nL = numel(L_list);
nMethods = numel(methods);

Y = NaN(nL, nMethods);
S = NaN(nL, nMethods);

for mi = 1:nMethods
    m = methods{mi};

    mask = summaryTable.method == categorical({m});
    Tm = summaryTable(mask,:);

    if isempty(Tm)
        continue;
    end

    % Ensure order by L
    [~,ord] = ismember(L_list, Tm.L);
    validOrd = ord > 0;

    rows = ord(validOrd);
    cols = find(validOrd);

    Y(cols, mi) = Tm.mean_RES(rows);
    S(cols, mi) = Tm.std_RES(rows);
end

% Grouped bar plot
xCats = categorical(string(L_list), string(L_list), 'Ordinal', true);
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

xlabel('Inner rank L (L_n=L)', ...
    'Color', 'k', ...
    'FontSize', fontSize + 2);

ylabel('Residual error: mean \pm 1 std', ...
    'Color', 'k', ...
    'FontSize', fontSize + 2);

title('Scalability: residual error vs inner rank L', ...
    'Color', 'k', ...
    'FontSize', fontSize + 4);

lgd = legend('Location','northwest');
lgd.TextColor = 'k';
lgd.Color = 'w';
lgd.FontSize = fontSize + 2;

save_current_figure(figDir, 'residual_vs_L');

end

function plot_iters_vs_L(summaryTable, L_list, figDir)

% Plot number of iterations vs L as grouped bars with std whiskers
figure('Name','Iterations vs L','Color','w');

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

nL = numel(L_list);
nMethods = numel(methods);

Y = NaN(nL, nMethods);
S = NaN(nL, nMethods);

for mi = 1:nMethods
    m = methods{mi};

    mask = summaryTable.method == categorical({m});
    Tm = summaryTable(mask,:);

    if isempty(Tm)
        continue;
    end

    % Ensure order by L
    [~,ord] = ismember(L_list, Tm.L);
    validOrd = ord > 0;

    rows = ord(validOrd);
    cols = find(validOrd);

    Y(cols, mi) = Tm.mean_iters(rows);
    S(cols, mi) = Tm.std_iters(rows);
end

% Grouped bar plot
xCats = categorical(string(L_list), string(L_list), 'Ordinal', true);
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

xlabel('Inner rank L (L_n=L)', ...
    'Color', 'k', ...
    'FontSize', fontSize + 2);

ylabel('Number of iterations: mean \pm 1 std', ...
    'Color', 'k', ...
    'FontSize', fontSize + 2);

title('Iterations vs inner rank L', ...
    'Color', 'k', ...
    'FontSize', fontSize + 4);

lgd = legend('Location','northwest');
lgd.TextColor = 'k';
lgd.Color = 'w';
lgd.FontSize = fontSize + 2;

save_current_figure(figDir, 'iterations_vs_L');

end

function bytes = proxy_peak_matlab_bytes(methodName, dims, Rval, Lval)

N = numel(dims);
I_tot = prod(double(dims));
I = double(dims(1));

core_tot = double(Lval)^double(N);
J = double(Rval)^2 * double(Lval);

switch lower(char(methodName))
    case 'baseline'
        bytes = 8 * (core_tot*I_tot + core_tot^2 + 2*I_tot);

    case 'fast cpu'
        bytes = 8 * (J*(I_tot/I) + core_tot^2 + 2*I_tot);

    case 'fast gpu'
        bytes = 8 * (2*I_tot + core_tot^2);

    otherwise
        bytes = NaN;
end

end

function bytes = proxy_peak_gpu_bytes(methodName, dims, Rval, Lval, trackGPU)

if ~trackGPU || ~strcmpi(char(methodName), 'fast GPU')
    bytes = 0;
    return;
end

I_tot = prod(double(dims));
I = double(dims(1));

J = double(Rval)^2 * double(Lval);
core_tot = double(Lval)^double(numel(dims));

bytes = 8 * (3*I_tot + J*(I_tot/I) + core_tot^2 + J^2);

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
    stop(mon.timer); delete(mon.timer);
catch
end
stats = MEMMON.(mon.key);
end
