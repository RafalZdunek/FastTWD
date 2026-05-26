clear; clc;

% -------------------------------------------------------------------------
% Repository path setup.
% This script can be run either from the repository root by
%   run(fullfile('experiments','run_sweep_tensor_size.m'))
% or directly from the experiments directory.
% -------------------------------------------------------------------------
scriptPath = mfilename('fullpath');
if isempty(scriptPath)
    scriptPath = which('run_sweep_tensor_size');
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
% Final experiments: Monte Carlo + sweep over I (dense TW decomposition)
%
% - N = 4 (tensor order)
% - I1=I2= ... =IN = I, I in [10 20 30 40 50]
% - R = L = 3 (all modes)
% - Monte Carlo runs: MC = 10
%
% Methods:
%   1) baseline CPU: inc_TW_TC
%   2) fast CPU: fast_twd_cpu
%   3) fast GPU: fast_twd_gpu
%
% Outputs:
%   - resultsTable (raw per-run results)
%   - summaryTable (means/std by I and method)
%   - figures: runtime vs I, peak memory vs I
%   - saved MAT-file with all outputs
% =========================================================================

fprintf('Working folder: %s\n', pwd);
fprintf('MATLAB version: %s\n', version);

% ----------------------- experiment settings -----------------------
N = 4;                           % Tensor order: I_1=I_2=I_3=I_4=I
I_list = [10 20 30 40 50];       % Mode sizes to test: I_n = 10, 20, 30 for all modes
rank_val = 3;                    % Uniform TW rank value: outer ranks R_n = 3 and inner ranks L_n = 3 for all modes

% -------------------------------------------------------------------
R = rank_val * ones(1,N);
L = rank_val * ones(1,N);
opts.R     = [R; L];

opts.maxit = 50;                 % Maximum number of PAM iterations
opts.rho   = 1;                  % Proximal/regularization parameter (stabilizes updates; larger => stronger damping)
opts.tol   = 1e-8;               % Stopping tolerance on relative change (RSE) between iterates

Omega = [];                      % Mask tensor for completion only; empty => dense decomposition (no missing entries)

% Execution mode:
%   'runtime' : measure runtime only
%   'memory'  : measure peak memory only
%   'both'    : measure both runtime and peak memory
MODE = 'runtime';  % 'runtime' | 'memory' | 'both'
MC_runtime = 10;  % Monte Carlo runs for MODE='runtime'
MC_memory  = 10;  % Monte Carlo runs for MODE='memory' or MODE='both'

% Memory measurement method:
%   'sampled' : timer-based sampling of MATLAB/GPU memory
%   'proxy'   : deterministic analytical proxy estimates
%   'hybrid'  : proxy for small tensors, sampled for larger tensors
MEM_METHOD = 'sampled';  % 'sampled' | 'proxy' | 'hybrid'
fprintf('Execution MODE: %s | MEM_METHOD: %s\n', MODE, MEM_METHOD);

% GPU availability
hasGPU = (exist('gpuDeviceCount','file')==2) && gpuDeviceCount>0;
if hasGPU
    g = gpuDevice; 
    % Warm-up GPU outside timed runs (JIT/init)
    a = gpuArray.ones(1024,1024,'double'); 
    wait(g);
    fprintf('GPU detected: %s\n', g.Name);
else
    fprintf('GPU not detected (or no Parallel Computing Toolbox). GPU method will be skipped.\n');
end

% ----------------------- results storage -----------------------
% Raw results rows
rows = {};
rowi = 0;

% Columns:
% I, mc, method, time_s, RES, iters, peakMatlab_bytes, peakGpu_bytes, ok, err
colnames = {'I','mc','method','time_s','RES','iters','peakMatlab_bytes','peakGpu_bytes','ok','err'};

t0 = datetime('now');

% ----------------------- main sweep -----------------------
for ii = 1:numel(I_list)
    I = I_list(ii);
    dims = I*ones(1,N);

    % Memory monitor polling periods (seconds).
    % Baseline can be long -> slower polling to avoid overhead.
  
    if  I <= 10
        period_base = 1e-2;  % baseline
        period_fast = 1e-3;
        period_gpu  = 1e-3;
    elseif I <= 20
        period_base = 1e-2;  % baseline 
        period_fast = 1e-3;
        period_gpu  = 1e-3;
    elseif I <= 30
        period_base = 1e-1;  % baseline 
        period_fast = 1e-2;
        period_gpu  = 1e-2;
    elseif I <= 40
        period_base = 1;  % baseline 
        period_fast = .1;
        period_gpu  = .1;
    else
        period_base = 2;  % baseline 
        period_fast = .5;
        period_gpu  = .2;
    end

    fprintf('\n=============================================================\n');
    fprintf('Sweep I=%d, dims=[%s], R=L=%d, MODE=%s\n', I, num2str(dims), rank_val, MODE);
    fprintf('=============================================================\n');
    
    if strcmpi(MODE,'runtime')
        MC = MC_runtime;
    else
        MC = MC_memory;
    end

    for mc = 1:MC
        % Fix seed per run for reproducibility of synthetic data
        rng(100000 + 1000*I + mc, 'twister');

        % Generate synthetic dense TW tensor (nonnegative, like your DataBenchmark)
        Y = make_tw_tensor(dims, R, L);
        Ynorm = norm(Y(:));

        fprintf('\n-------------------------------------------------------------\n');
        fprintf('I = %d | Monte Carlo trial = %d / %d\n', I, mc, MC);
        fprintf('-------------------------------------------------------------\n');
 
        % ---------------- baseline CPU ----------------
        [rowi, rows] = run_one('baseline', @()call_baseline(Y, Omega, opts), ...
            Y, Ynorm, false, period_base, rowi, rows, I, mc, MODE, MEM_METHOD, dims, rank_val);

        % ---------------- fast-core CPU ----------------
        [rowi, rows] = run_one('fast CPU', @()call_fast_cpu(Y, Omega, opts), ...
            Y, Ynorm, false, period_fast, rowi, rows, I, mc, MODE, MEM_METHOD, dims, rank_val);

        % ---------------- fast-core GPU ----------------
        if hasGPU
            [rowi, rows] = run_one('fast GPU', @()call_fast_gpu_full(Y, Omega, opts), ...
                Y, Ynorm, true, period_gpu, rowi, rows, I, mc, MODE, MEM_METHOD, dims, rank_val);
        else
            % record skipped GPU
            rowi = rowi + 1;
            rows(rowi,:) = {I, mc, 'fast GPU', NaN, NaN, NaN, NaN, NaN, false, 'GPU not available'};
        end
    end
end

% ----------------------- build tables -----------------------
resultsTable = cell2table(rows, 'VariableNames', colnames);

% Convert method to categorical for cleaner grouping
resultsTable.method = categorical(resultsTable.method);

% Summary by (I, method): mean/std on successful runs only
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

summaryTable = groupsummary(summaryInput, {'I','method'}, ...
    {'mean','std','median',@iqr}, {'time_s','RES','iters','peakMatlab_bytes','peakGpu_bytes'});

% Rename function-handle output columns from fun1_* to iqr_*
summaryTable.Properties.VariableNames = strrep(summaryTable.Properties.VariableNames, 'fun1_', 'iqr_');
timestamp = datestr(t0,'yyyymmdd_HHMMSS');

resultsRoot = fullfile(rootDir, 'results');
if ~exist(resultsRoot, 'dir')
    mkdir(resultsRoot);
end

runDir = fullfile(resultsRoot, sprintf('sweep_tensor_size_%s_%s', MODE, timestamp));
if ~exist(runDir, 'dir')
    mkdir(runDir);
end

figDir = fullfile(runDir, 'figures');
if ~exist(figDir, 'dir')
    mkdir(figDir);
end

outname = fullfile(runDir, sprintf('TW_final_sweep_I_%s_%s.mat', MODE, timestamp));

save(outname, 'resultsTable', 'summaryTable', 'opts', 'MC_runtime', 'MC_memory', ...
    'I_list', 'N', 'R', 'L', 'rank_val', 'MODE', 'MEM_METHOD', 't0', ...
    'excludeFirstMCForPlots', 'timestamp', 'runDir', 'figDir');

fprintf('\nSaved results to %s\n', outname);
fprintf('Saved figures to %s\n', figDir);

% ----------------------- plots -----------------------
if any(~isnan(summaryTable.mean_time_s) & summaryTable.mean_time_s > 0)
    plot_runtime_vs_I(summaryTable, I_list, figDir);
end

if any(~isnan(summaryTable.median_peakMatlab_bytes) & summaryTable.median_peakMatlab_bytes > 0) || ...
        any(~isnan(summaryTable.median_peakGpu_bytes) & summaryTable.median_peakGpu_bytes > 0)
    plot_memory_vs_I(summaryTable, I_list, figDir);
end

if any(~isnan(summaryTable.mean_RES) & summaryTable.mean_RES > 0)
    plot_residual_vs_I(summaryTable, I_list, figDir);
end

if any(~isnan(summaryTable.mean_iters) & summaryTable.mean_iters > 0)
    plot_iters_vs_I(summaryTable, I_list, figDir);
end

% =========================================================================
% Local functions
% =========================================================================

function [rowi, rows] = run_one(methodName, fhandle, Y, Ynorm, trackGPU, period_s, rowi, rows, I, mc, MODE, MEM_METHOD, dims, rank_val)
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
%   'hybrid'  : use proxy for small tensors, sampled otherwise

ok = true;
err = '';

t = NaN;
RES = NaN;
iters = NaN;

peakMatlab = NaN;
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
            % For small tensors, sampled deltas are often zero/NaN due to memory pools.
            % The threshold can be adjusted if needed.
            if I <= 20
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
    key = matlab.lang.makeValidName(sprintf('%s_%d_%d', methodName, I, mc));

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

if useSampledMem && ~isempty(mon)
    stats = memmon_stop(mon);
    peakMatlab = stats.peakMatlabDelta;
    peakGpu    = stats.peakGpuDelta;
end

if useProxyMem
    peakMatlab = proxy_peak_matlab_bytes(methodName, numel(dims), dims, rank_val);
    peakGpu    = proxy_peak_gpu_bytes(methodName, numel(dims), dims, rank_val, trackGPU);
end

rowi = rowi + 1;
rows(rowi,:) = {I, mc, methodName, t, RES, iters, peakMatlab, peakGpu, ok, err};

end

function [Yhat, Out] = call_baseline(Y, Omega, opts)
% Baseline CPU solver 
try
    [Yhat, ~, ~, Out] = inc_TW_TC(Y, Omega, opts);
catch
    [Yhat, ~, ~, Out] = inc_TW_TC(Y, opts);
end
end

function [Yhat, Out] = call_fast_cpu(Y, Omega, opts)
% Fast-core CPU solver
try
    [Yhat, ~, ~, Out] = fast_twd_cpu(Y, Omega, opts);
catch
    [Yhat, ~, ~, Out] = fast_twd_cpu(Y, opts);
end
end

function [Yhat, Out] = call_fast_gpu_full(Y, Omega, opts)
% Fast-core GPU-full solver 
try
    [Yhat, ~, ~, Out] = fast_twd_gpu(Y, Omega, opts);
catch
    [Yhat, ~, ~, Out] = fast_twd_gpu(Y, opts);
end
end

function X = make_tw_tensor(dims, R, L)
% Generate a synthetic dense TW tensor with nonnegative entries.
% dims: 1xN, R: 1xN, L: 1xN
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

function plot_runtime_vs_I(summaryTable, I_list, figDir)

% Plot mean runtime vs I with shaded mean +/- std bands
figure('Name','Runtime vs I','Color','w');

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
    if isempty(Tm), continue; end

    [~,ord] = ismember(I_list, Tm.I);
    ord(ord==0) = [];
    Tm = Tm(ord,:);
    if isempty(Tm), continue; end

    x = Tm.I;
    y = Tm.mean_time_s;
    s = Tm.std_time_s;

    validMean = isfinite(x) & isfinite(y) & y > 0;
    xMean = x(validMean);
    yMean = y(validMean);
    sMean = s(validMean);
    if isempty(xMean), continue; end

    validBand = isfinite(sMean) & (sMean > 0);
    if any(validBand)
        idxValid = find(validBand);
        splitPts = [0; find(diff(idxValid) > 1); numel(idxValid)];
        for kk = 1:numel(splitPts)-1
            segIdx = idxValid(splitPts(kk)+1 : splitPts(kk+1));
            if numel(segIdx) < 2, continue; end
            xSeg = xMean(segIdx);
            ySeg = yMean(segIdx);
            sSeg = sMean(segIdx);
            yLow  = max(ySeg - sSeg, eps);
            yHigh = ySeg + sSeg;
            fill([xSeg; flipud(xSeg)], [yLow; flipud(yHigh)], c, ...
                 'FaceAlpha', 0.4, 'EdgeColor', 'none', 'HandleVisibility', 'off');
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
xticks(unique(I_list));
xtickformat('%d');

xlabel('I (I_1=I_2=I_3=I_4=I)', 'Color', 'k', 'FontSize', fontSize + 2);
ylabel('Runtime (s): mean \pm 1 std', 'Color', 'k', 'FontSize', fontSize + 2);
title('Scalability: runtime vs I', 'Color', 'k', 'FontSize', fontSize + 4);

lgd = legend('Location','northwest');
lgd.TextColor = 'k';
lgd.Color = 'w';
lgd.FontSize = fontSize + 2;

save_current_figure(figDir, 'runtime_vs_I');

end

function plot_memory_vs_I(summaryTable, I_list, figDir)

% Plot peak memory vs I using median values only.
% Variability bands are intentionally omitted because sampled peak-memory
% measurements are affected by MATLAB/GPU memory-pool reuse and transient
% allocator effects, which can create misleading jump-like uncertainty regions.
figure('Name','Peak memory vs I','Color','w');

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

colors = [
    0.30 0.75 0.93
    0.95 0.55 0.20
    0.40 0.85 0.45
    0.70 0.35 0.95
];

numPlotted = 0;

for ci = 1:size(curves,1)
    methodName  = curves{ci,1};
    medianCol   = curves{ci,2};
    displayName = curves{ci,3};
    lineStyle   = curves{ci,4};
    markerStyle = curves{ci,5};
    c = colors(ci,:);

    if ~ismember(medianCol, summaryTable.Properties.VariableNames)
        warning('Skipping "%s": column "%s" is missing in summaryTable.', displayName, medianCol);
        continue;
    end

    mask = summaryTable.method == categorical({methodName});
    Tm = summaryTable(mask,:);
    if isempty(Tm), continue; end

    [~,ord] = ismember(I_list, Tm.I);
    ord(ord==0) = [];
    Tm = Tm(ord,:);
    if isempty(Tm), continue; end

    x = Tm.I;
    y = toGB(Tm.(medianCol));

    validMean = isfinite(x) & isfinite(y) & y > 0;
    xMean = x(validMean);
    yMean = y(validMean);
    if isempty(xMean)
        warning('Skipping "%s": no positive finite memory values to plot.', displayName);
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

    numPlotted = numPlotted + 1;
end

set(gca, 'YScale', 'log');
xticks(unique(I_list));
xtickformat('%d');

xlabel('I (I_1=I_2=I_3=I_4=I)', 'Color', 'k', 'FontSize', fontSize + 2);
ylabel('Peak memory increase (GB): median', 'Color', 'k', 'FontSize', fontSize + 2);
title('Scalability: peak memory vs I', 'Color', 'k', 'FontSize', fontSize + 4);

if numPlotted > 0
    lgd = legend('show', 'Location','northwest');
    lgd.TextColor = 'k';
    lgd.Color = 'w';
    lgd.FontSize = fontSize + 2;
else
    text(0.5, 0.5, 'No memory data available for this MODE', ...
        'Units', 'normalized', ...
        'HorizontalAlignment', 'center', ...
        'FontSize', fontSize + 2, ...
        'Color', 'k');
end

save_current_figure(figDir, 'memory_vs_I');

end

function plot_residual_vs_I(summaryTable, I_list, figDir)

figure('Name','Residual error vs I','Color','w');
fontSize = 16;

ax = gca;
ax.Color = 'w'; ax.XColor = 'k'; ax.YColor = 'k';
ax.GridColor = [0.5 0.5 0.5]; ax.MinorGridColor = [0.35 0.35 0.35];
ax.FontSize = fontSize; ax.LineWidth = 1.2;
hold on; grid on; box on;

methods = {'baseline','fast CPU','fast GPU'};
colors = [0.30 0.75 0.93; 0.95 0.55 0.20; 0.40 0.85 0.45];

nI = numel(I_list);
nMethods = numel(methods);
Y = NaN(nI, nMethods);
S = NaN(nI, nMethods);

for mi = 1:nMethods
    m = methods{mi};
    mask = summaryTable.method == categorical({m});
    Tm = summaryTable(mask,:);
    if isempty(Tm), continue; end
    [~,ord] = ismember(I_list, Tm.I);
    validOrd = ord > 0;
    rows = ord(validOrd);
    cols = find(validOrd);
    Y(cols, mi) = Tm.mean_RES(rows);
    S(cols, mi) = Tm.std_RES(rows);
end

b = bar(categorical(string(I_list)), Y, 'grouped');
for mi = 1:nMethods
    b(mi).FaceColor = colors(mi,:);
    b(mi).EdgeColor = 'k';
    b(mi).LineWidth = 0.8;
    b(mi).DisplayName = methods{mi};
end

for mi = 1:nMethods
    x = b(mi).XEndPoints;
    y = Y(:,mi);
    s = S(:,mi);
    valid = isfinite(x(:)) & isfinite(y(:)) & isfinite(s(:)) & y(:) > 0 & s(:) > 0;
    errorbar(x(valid), y(valid), s(valid), 'k', ...
        'LineStyle', 'none', 'LineWidth', 1.2, 'CapSize', 10, 'HandleVisibility', 'off');
end

xlabel('I (I_1=I_2=I_3=I_4=I)', 'Color', 'k', 'FontSize', fontSize + 2);
ylabel('Residual error: mean \pm 1 std', 'Color', 'k', 'FontSize', fontSize + 2);
title('Scalability: residual error vs I', 'Color', 'k', 'FontSize', fontSize + 4);

lgd = legend('Location','northwest');
lgd.TextColor = 'k'; lgd.Color = 'w'; lgd.FontSize = fontSize + 2;

save_current_figure(figDir, 'residual_vs_I');

end

function plot_iters_vs_I(summaryTable, I_list, figDir)

figure('Name','Iterations vs I','Color','w');
fontSize = 16;

ax = gca;
ax.Color = 'w'; ax.XColor = 'k'; ax.YColor = 'k';
ax.GridColor = [0.5 0.5 0.5]; ax.MinorGridColor = [0.35 0.35 0.35];
ax.FontSize = fontSize; ax.LineWidth = 1.2;
hold on; grid on; box on;

methods = {'baseline','fast CPU','fast GPU'};
colors = [0.30 0.75 0.93; 0.95 0.55 0.20; 0.40 0.85 0.45];

nI = numel(I_list);
nMethods = numel(methods);
Y = NaN(nI, nMethods);
S = NaN(nI, nMethods);

for mi = 1:nMethods
    m = methods{mi};
    mask = summaryTable.method == categorical({m});
    Tm = summaryTable(mask,:);
    if isempty(Tm), continue; end
    [~,ord] = ismember(I_list, Tm.I);
    validOrd = ord > 0;
    rows = ord(validOrd);
    cols = find(validOrd);
    Y(cols, mi) = Tm.mean_iters(rows);
    S(cols, mi) = Tm.std_iters(rows);
end

b = bar(categorical(string(I_list)), Y, 'grouped');
for mi = 1:nMethods
    b(mi).FaceColor = colors(mi,:);
    b(mi).EdgeColor = 'k';
    b(mi).LineWidth = 0.8;
    b(mi).DisplayName = methods{mi};
end

for mi = 1:nMethods
    x = b(mi).XEndPoints;
    y = Y(:,mi);
    s = S(:,mi);
    valid = isfinite(x(:)) & isfinite(y(:)) & isfinite(s(:)) & y(:) > 0 & s(:) > 0;
    errorbar(x(valid), y(valid), s(valid), 'k', ...
        'LineStyle', 'none', 'LineWidth', 1.2, 'CapSize', 10, 'HandleVisibility', 'off');
end

xlabel('I (I_1=I_2=I_3=I_4=I)', 'Color', 'k', 'FontSize', fontSize + 2);
ylabel('Number of iterations: mean \pm 1 std', 'Color', 'k', 'FontSize', fontSize + 2);
title('Scalability: iterations vs I', 'Color', 'k', 'FontSize', fontSize + 4);

lgd = legend('Location','northwest');
lgd.TextColor = 'k'; lgd.Color = 'w'; lgd.FontSize = fontSize + 2;

save_current_figure(figDir, 'iterations_vs_I');

end

% ========================= Figure export helper =========================
function save_current_figure(figDir, baseName)

if nargin < 1 || isempty(figDir)
    figDir = pwd;
end

if nargin < 2 || isempty(baseName)
    baseName = 'figure';
end

if ~exist(figDir, 'dir')
    mkdir(figDir);
end

fig = gcf;
set(fig, 'InvertHardcopy', 'off');
set(fig, 'PaperPositionMode', 'auto');

axs = findall(fig, 'Type', 'axes');
for k = 1:numel(axs)
    try
        axs(k).Toolbar.Visible = 'off';
    catch
        % Axes toolbar is not available in older MATLAB versions.
    end
end

drawnow;

pngFile = fullfile(figDir, [baseName '.png']);
epsFile = fullfile(figDir, [baseName '.eps']);

exportgraphics(fig, pngFile, 'Resolution', 300);

try
    exportgraphics(fig, epsFile, 'ContentType', 'vector');
catch
    try
        print(fig, epsFile, '-depsc', '-vector');
    catch
        print(fig, epsFile, '-depsc', '-painters');
    end
end

fprintf('Saved figure: %s\n', pngFile);
fprintf('Saved figure: %s\n', epsFile);

end


% ========================= Deterministic memory proxy helpers =========================
function bytes = proxy_peak_matlab_bytes(methodName, N, dims, r)
% Deterministic proxy of peak MATLAB host memory (bytes).

I_tot = prod(double(dims));
L_tot = double(r)^double(N);
J = double(r)^3;

switch lower(char(methodName))
    case 'baseline'
        bytes = 8 * (L_tot*I_tot + L_tot^2 + 2*I_tot);
    case 'fast cpu'
        I = double(dims(1));
        bytes = 8 * (J*(I_tot/I) + L_tot^2 + 2*I_tot);
    case 'fast gpu'
        bytes = 8 * (2*I_tot + L_tot^2);
    otherwise
        bytes = NaN;
end
end

function bytes = proxy_peak_gpu_bytes(methodName, N, dims, r, trackGPU)
% Deterministic proxy of peak GPU device memory (bytes).

if ~trackGPU || ~strcmpi(char(methodName), 'fast GPU')
    bytes = 0;
    return;
end

I_tot = prod(double(dims));
L_tot = double(r)^double(N);
J = double(r)^3;
I = double(dims(1));
bytes = 8 * (2*I_tot + J*(I_tot/I) + L_tot^2);
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
    
    % MATLAB timer has millisecond-level precision. Keep the requested
    % sampling strategy, but round to full milliseconds to avoid warnings.
    period_s = max(round(double(period_s) * 1000) / 1000, 1e-3);
    
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

% MATLAB host delta
if ispc && ~isnan(MEMMON.(key).initMatlab)
    try
        m = memory;
        used = double(m.MemUsedMATLAB);
        delta = max(0, used - MEMMON.(key).initMatlab);
        MEMMON.(key).peakMatlabDelta = max(MEMMON.(key).peakMatlabDelta, delta);
    catch
    end
end

% GPU delta
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
