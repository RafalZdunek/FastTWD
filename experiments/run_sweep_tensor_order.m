clear; clc;

% -------------------------------------------------------------------------
% Repository path setup.
% This script can be run either from the repository root, e.g.
%   run(fullfile('experiments','run_sweep_tensor_order.m'))
% or directly from the experiments directory.
% -------------------------------------------------------------------------
scriptPath = mfilename('fullpath');
if isempty(scriptPath)
    scriptPath = which('run_sweep_tensor_order');
end
scriptDir = fileparts(scriptPath);

% If the script is stored in the experiments directory, the repository root
% is one level above. If the script is stored in the repository root, use the
% script directory itself.
[~, scriptFolderName] = fileparts(scriptDir);
if strcmpi(scriptFolderName, 'experiments')
    rootDir = fileparts(scriptDir);
else
    rootDir = scriptDir;
end

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
% Final experiments: Monte Carlo + sweep over tensor order N
%
% Fixed:
%   - I_n = 10 for all modes
%   - R_n = L_n = 3 for all modes
%
% Sweep:
%   - N in [3 4 5 6 7]
%   - Monte Carlo runs: MC=10 
%
% Methods:
%   1) baseline: inc_TW_TC
%   2) fast CPU: fast_twd_cpu
%   3) fast GPU: fast_twd_gpu
%
% Toggle modes (set at top):
%   MODE = 'runtime'  -> measure runtime only (no memory polling)
%   MODE = 'memory'   -> measure peak memory only (no runtime timing)
%   MODE = 'both'     -> measure both (for convenience, but affects runtime slightly)
%
% Outputs:
%   - resultsTable (raw per-run results)
%   - summaryTable (means/std by N and method)
%   - plots: runtime vs N (mean ± std), peak memory vs N (mean ± std)
%   - saved MAT-file with all outputs
% =========================================================================

fprintf('Working folder: %s\n', pwd);
fprintf('MATLAB version: %s\n', version);

% ----------------------- user controls -----------------------
MODE = 'runtime';  % 'runtime' | 'memory' | 'both'
MEM_METHOD = 'sampled'; % 'sampled' | 'proxy' | 'hybrid'

% ----------------------- experiment settings -----------------------
I_fixed = 10;                    % Mode size value (uniform): I_n = 10 for all modes, so dims = [10 ... 10]
N_list = [3 4 5 6 7];            % Tensor orders to test: N = 3, 4, 5, 6, 7, ...
rank_val = 3;                    % Tensor Wheel rank value (uniform): R_n = L_n = 3 for all modes
MC = 10;                         % Number of Monte Carlo trials for each tested mode size

% GPU availability
hasGPU = (exist('gpuDeviceCount','file')==2) && gpuDeviceCount>0;
if hasGPU
    g = gpuDevice; 
    a = gpuArray.ones(1024,1024,'double');  % warm-up kernel
    wait(gpuDevice);
    fprintf('GPU detected: %s\n', g.Name);
else
    fprintf('GPU not detected (or no Parallel Computing Toolbox). GPU method will be skipped.\n');
end

% ----------------------- results storage -----------------------
rows = {};
rowi = 0;

% Columns:
% N, mc, method, time_s, RES, iters, peakMatlab_bytes, peakGpu_bytes, ok, err
colnames = {'N','mc','method','time_s','RES','iters','peakMatlab_bytes','peakGpu_bytes','ok','err'};
t0 = datetime('now');

% ----------------------- main sweep -----------------------
for ni = 1:numel(N_list)
    N = N_list(ni);
    dims = I_fixed * ones(1,N);

    % Sampling periods for memory polling (seconds)
    % (only used when MODE includes 'memory')
    if N > 6
        period_base = 2.0;  % baseline 
        period_fast = 0.5;
        period_gpu  = 0.2;
    elseif N == 6
        period_base = 1.0;  % baseline
        period_fast = 0.5;
        period_gpu  = 0.2;
    elseif N == 5
        period_base = 0.2;  % baseline 
        period_fast = 1e-2;
        period_gpu  = 1e-2;
    elseif N == 4
        period_base = 1e-2;  % baseline 
        period_fast = 2e-3;
        period_gpu  = 2e-3;
    elseif N == 3
        period_base = 2e-3;  % baseline 
        period_fast = 2e-3;
        period_gpu  = 2e-3;
    end

    fprintf('\n=============================================================\n');
    fprintf('Sweep N=%d, dims=[%s], I=10, R=L=%d, MC=%d\n', N, num2str(dims), rank_val, MC);
    fprintf('=============================================================\n');

    R = rank_val * ones(1,N);
    L = rank_val * ones(1,N);
    opts.R     = [R; L];

    opts.maxit = 50;                 % Maximum number of PAM iterations
    opts.rho   = 1;                  % Proximal/regularization parameter (stabilizes updates; larger => stronger damping)
    opts.tol   = 1e-8;               % Stopping tolerance on relative change (RSE) between iterates

    Omega = [];                      % Mask tensor for completion only; empty => dense decomposition (no missing entries)

    for mc = 1:MC
        rng(100000 + 1000*N + mc, 'twister');

        % Generate synthetic dense TW tensor for this N
        Y = make_tw_tensor(dims, R, L);
        Ynorm = norm(Y(:));

        fprintf('\n-------------------------------------------------------------\n');
        fprintf('N = %d | Monte Carlo trial = %d / %d\n', N, mc, MC);
        fprintf('-------------------------------------------------------------\n');

        % baseline CPU
        [rowi, rows] = run_one('baseline', @()call_baseline(Y, Omega, opts), ...
            Y, Ynorm, false, period_base, MODE, MEM_METHOD, rowi, rows, N, mc);
        
        % fast CPU
        [rowi, rows] = run_one('fast CPU', @()call_fast_cpu(Y, Omega, opts), ...
            Y, Ynorm, false, period_fast, MODE, MEM_METHOD, rowi, rows, N, mc);

        % fast GPU
        if hasGPU
            [rowi, rows] = run_one('fast GPU', @()call_fast_gpu_full(Y, Omega, opts), ...
                Y, Ynorm, true, period_gpu, MODE, MEM_METHOD, rowi, rows, N, mc);
        else
            rowi = rowi + 1;
            rows(rowi,:) = {N, mc, 'fast GPU', NaN, NaN, NaN, NaN, NaN, false, 'GPU not available'};
        end
    end
end

% ----------------------- build tables -----------------------
resultsTable = cell2table(rows, 'VariableNames', colnames);
resultsTable.method = categorical(resultsTable.method);

okMask = resultsTable.ok == true;

summaryTable = groupsummary(resultsTable(okMask,:), {'N','method'}, ...
    {'mean','std','median',@iqr}, {'time_s','RES','iters','peakMatlab_bytes','peakGpu_bytes'});

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

resultsRoot = fullfile(rootDir, 'results');
if ~exist(resultsRoot, 'dir')
    mkdir(resultsRoot);
end

runDir = fullfile(resultsRoot, sprintf('sweep_tensor_order_%s_%s', MODE, timestamp));
if ~exist(runDir, 'dir')
    mkdir(runDir);
end

figDir = fullfile(runDir, 'figures');
if ~exist(figDir, 'dir')
    mkdir(figDir);
end

outname = fullfile(runDir, sprintf('TW_final_sweep_N_%s_%s.mat', MODE, timestamp));

save(outname, 'resultsTable', 'summaryTable', 'opts', 'MODE', 'MEM_METHOD', ...
    'MC', 'I_fixed', 'N_list', 'rank_val', 't0', 'timestamp', ...
    'runDir', 'figDir');

fprintf('\nSaved results to %s\n', outname);
fprintf('Saved figures to %s\n', figDir);

% ----------------------- plots -----------------------
if strcmpi(MODE,'runtime') || strcmpi(MODE,'both')
    plot_runtime_vs_N(summaryTable, N_list, figDir);
end
if strcmpi(MODE,'memory') || strcmpi(MODE,'both')
    plot_memory_vs_N(summaryTable, N_list, figDir);
end
if any(~isnan(summaryTable.mean_RES) & summaryTable.mean_RES > 0)
    plot_residual_vs_N(summaryTable, N_list, figDir);
end

if any(~isnan(summaryTable.mean_iters) & summaryTable.mean_iters > 0)
    plot_iters_vs_N(summaryTable, N_list, figDir);
end


% =========================================================================
% Local functions
% =========================================================================

function [rowi, rows] = run_one(methodName, fhandle, Y, Ynorm, trackGPU, period_s, MODE, MEM_METHOD, rowi, rows, N, mc)

% Run a single method; either measure runtime only, memory only, or both.
ok = true; err = '';
t = NaN; RES = NaN; iters = NaN;
peakMat = NaN; peakGpu = NaN;

doTime = strcmpi(MODE,'runtime') || strcmpi(MODE,'both');
doMem  = strcmpi(MODE,'memory')  || strcmpi(MODE,'both');

switch lower(MEM_METHOD)
    case 'sampled'
        useProxy = false;
    case 'proxy'
        useProxy = doMem;
    case 'hybrid'
        useProxy = doMem && (N <= 4);
    otherwise
        error('Unknown MEM_METHOD: %s. Use sampled, proxy, or hybrid.', MEM_METHOD);
end

if doMem && ~useProxy
    mon = memmon_start(trackGPU, sprintf('%s_%d_%d', methodName, N, mc), period_s);
end

try
    if doTime
        tic;
        [Yhat, Out] = fhandle();
        if trackGPU, wait(gpuDevice); end
        t = toc;
    else
        [Yhat, Out] = fhandle();
        if trackGPU, wait(gpuDevice); end
    end

    RES = norm(Y(:)-Yhat(:)) / Ynorm;
    iters = numel(Out.RSE);
catch ME
    ok = false;
    err = ME.message;
    Yhat = [];
    Out = struct('RSE',[]);
end

if doMem
    if useProxy
        peakMat = proxy_peak_matlab_bytes(methodName, N, size(Y), 3);  % r=3 fixed here
        if strcmpi(methodName,'fast GPU')
            peakGpu = proxy_peak_gpu_bytes(N, size(Y), 3);
        else
            peakGpu = NaN;
        end
    else
        stats = memmon_stop(mon);
        peakMat = stats.peakMatlabDelta;
        peakGpu = stats.peakGpuDelta;
    end
end

rowi = rowi + 1;
rows(rowi,:) = {N, mc, methodName, t, RES, iters, peakMat, peakGpu, ok, err};

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


function plot_runtime_vs_N(summaryTable, N_list, figDir)

% Plot mean runtime vs N with shaded mean +/- std bands
figure('Name','Runtime vs N','Color','w');

fontSize = 16;

ax = gca;
disableDefaultInteractivity(ax);
ax.Toolbar.Visible = 'off';
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

    if isempty(Tm)
        continue;
    end

    % Ensure order by N
    [~,ord] = ismember(N_list, Tm.N);
    ord(ord==0) = [];
    Tm = Tm(ord,:);

    if isempty(Tm)
        continue;
    end

    x = Tm.N;
    y = Tm.mean_time_s;
    s = Tm.std_time_s;

    validMean = isfinite(x) & isfinite(y) & y > 0;
    xMean = x(validMean);
    yMean = y(validMean);
    sMean = s(validMean);

    if isempty(xMean)
        continue;
    end

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
xticks(unique(N_list));
xtickformat('%d');

xlabel('Tensor order N', ...
    'Color', 'k', ...
    'FontSize', fontSize + 2);

ylabel('Runtime (s): mean \pm 1 std', ...
    'Color', 'k', ...
    'FontSize', fontSize + 2);

title('Scalability: runtime vs tensor order N', ...
    'Color', 'k', ...
    'FontSize', fontSize + 4);

lgd = legend('Location','northwest');
lgd.TextColor = 'k';
lgd.Color = 'w';
lgd.FontSize = fontSize + 2;

save_current_figure(figDir, 'runtime_vs_N');

end

function plot_memory_vs_N(summaryTable, N_list, figDir)

% Plot peak memory vs N using median values only.
% Variability bands are intentionally omitted because sampled peak-memory
% measurements may be affected by MATLAB/GPU memory-pool reuse and transient
% allocator effects.
figure('Name','Peak memory vs N','Color','w');

fontSize = 16;

ax = gca;
disableDefaultInteractivity(ax);
ax.Toolbar.Visible = 'off';
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

% Four separate memory curves are shown:
%   1) baseline MATLAB host memory
%   2) fast CPU MATLAB host memory
%   3) fast GPU MATLAB host memory
%   4) fast GPU device memory
%
% NOTE: The GPU method appears twice intentionally: host-side MATLAB memory
% and GPU-device memory are different quantities.
curves = {
    'baseline', 'median_peakMatlab_bytes', 'baseline (MATLAB)',       '-',  'o'
    'fast CPU', 'median_peakMatlab_bytes', 'fast CPU (MATLAB)',       '-',  's'
    'fast GPU', 'median_peakMatlab_bytes', 'fast GPU (MATLAB)',       '-',  '^'
    'fast GPU', 'median_peakGpu_bytes',    'fast GPU (GPU device)',   '--', 'd'
};

colors = [
    0.30 0.75 0.93   % baseline MATLAB - blue/cyan
    0.95 0.55 0.20   % fast CPU MATLAB - orange
    0.40 0.85 0.45   % fast GPU MATLAB - green
    0.70 0.35 0.95   % fast GPU device - purple
];

nPlotted = 0;

for ci = 1:size(curves,1)

    methodName  = curves{ci,1};
    medianCol   = curves{ci,2};
    displayName = curves{ci,3};
    lineStyle   = curves{ci,4};
    markerStyle = curves{ci,5};
    c = colors(ci,:);

    if ~ismember(medianCol, summaryTable.Properties.VariableNames)
        warning('Skipping "%s": column "%s" is missing in summaryTable.', ...
            displayName, medianCol);
        continue;
    end

    mask = summaryTable.method == categorical({methodName});
    Tm = summaryTable(mask,:);

    if isempty(Tm)
        warning('Skipping "%s": method "%s" is missing in summaryTable.', ...
            displayName, methodName);
        continue;
    end

    % Ensure order by N
    [~,ord] = ismember(N_list, Tm.N);
    ord(ord==0) = [];
    Tm = Tm(ord,:);

    if isempty(Tm)
        continue;
    end

    x = Tm.N;
    y = toGB(Tm.(medianCol));

    % Log-scale cannot display zero or negative values. NaNs usually mean
    % that the script was run in MODE='runtime', so no memory was measured.
    valid = isfinite(x) & isfinite(y) & y > 0;

    x = x(valid);
    y = y(valid);

    if isempty(x)
        warning('Skipping "%s": no positive finite memory values found. Use MODE=''memory'' or MODE=''both'' for memory plots.', ...
            displayName);
        continue;
    end

    if strcmp(displayName, 'fast GPU (GPU device)')
        markerFaceColor = 'w';
    else
        markerFaceColor = c;
    end

    plot(x, y, ...
        'LineStyle', lineStyle, ...
        'Marker', markerStyle, ...
        'DisplayName', displayName, ...
        'Color', c, ...
        'LineWidth', 1.8, ...
        'MarkerSize', 8, ...
        'MarkerFaceColor', markerFaceColor, ...
        'MarkerEdgeColor', 'k');

    nPlotted = nPlotted + 1;
end

set(gca, 'YScale', 'log');
xticks(unique(N_list));
xtickformat('%d');

xlabel('Tensor order N', ...
    'Color', 'k', ...
    'FontSize', fontSize + 2);

ylabel('Peak memory increase (GB): median', ...
    'Color', 'k', ...
    'FontSize', fontSize + 2);

title('Scalability: peak memory vs tensor order N', ...
    'Color', 'k', ...
    'FontSize', fontSize + 4);

if nPlotted > 0
    lgd = legend('show', 'Location','northwest');
    lgd.TextColor = 'k';
    lgd.Color = 'w';
    lgd.FontSize = fontSize + 2;
else
    text(0.5, 0.5, 'No memory data to plot. Run with MODE = ''memory'' or MODE = ''both''.', ...
        'Units','normalized', ...
        'HorizontalAlignment','center', ...
        'FontSize', fontSize + 2, ...
        'Color','k');
end

save_current_figure(figDir, 'memory_vs_N');

end

function plot_residual_vs_N(summaryTable, N_list, figDir)

% Plot external residual error vs N as grouped bars with std whiskers
figure('Name','Residual error vs N','Color','w');

fontSize = 16;

ax = gca;
disableDefaultInteractivity(ax);
ax.Toolbar.Visible = 'off';
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

nN = numel(N_list);
nMethods = numel(methods);

Y = NaN(nN, nMethods);
S = NaN(nN, nMethods);

for mi = 1:nMethods
    m = methods{mi};

    mask = summaryTable.method == categorical({m});
    Tm = summaryTable(mask,:);

    if isempty(Tm)
        continue;
    end

    % Ensure order by N
    [~,ord] = ismember(N_list, Tm.N);
    validOrd = ord > 0;

    rows = ord(validOrd);
    cols = find(validOrd);

    Y(cols, mi) = Tm.mean_RES(rows);
    S(cols, mi) = Tm.std_RES(rows);
end

xCats = categorical(string(N_list), string(N_list), 'Ordinal', true);
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

xlabel('Tensor order N', ...
    'Color', 'k', ...
    'FontSize', fontSize + 2);

ylabel('Residual error: mean \pm 1 std', ...
    'Color', 'k', ...
    'FontSize', fontSize + 2);

title('Residual error vs tensor order N', ...
    'Color', 'k', ...
    'FontSize', fontSize + 4);

lgd = legend('Location','northwest');
lgd.TextColor = 'k';
lgd.Color = 'w';
lgd.FontSize = fontSize + 2;

save_current_figure(figDir, 'residual_vs_N');

end

function plot_iters_vs_N(summaryTable, N_list, figDir)

% Plot number of iterations vs N as grouped bars with std whiskers
figure('Name','Iterations vs N','Color','w');

fontSize = 16;

ax = gca;
disableDefaultInteractivity(ax);
ax.Toolbar.Visible = 'off';
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

nN = numel(N_list);
nMethods = numel(methods);

Y = NaN(nN, nMethods);
S = NaN(nN, nMethods);

for mi = 1:nMethods
    m = methods{mi};

    mask = summaryTable.method == categorical({m});
    Tm = summaryTable(mask,:);

    if isempty(Tm)
        continue;
    end

    % Ensure order by N
    [~,ord] = ismember(N_list, Tm.N);
    validOrd = ord > 0;

    rows = ord(validOrd);
    cols = find(validOrd);

    Y(cols, mi) = Tm.mean_iters(rows);
    S(cols, mi) = Tm.std_iters(rows);
end

xCats = categorical(string(N_list), string(N_list), 'Ordinal', true);
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

xlabel('Tensor order N', ...
    'Color', 'k', ...
    'FontSize', fontSize + 2);

ylabel('Number of iterations: mean \pm 1 std', ...
    'Color', 'k', ...
    'FontSize', fontSize + 2);

title('Iterations vs tensor order N', ...
    'Color', 'k', ...
    'FontSize', fontSize + 4);

lgd = legend('Location','northwest');
lgd.TextColor = 'k';
lgd.Color = 'w';
lgd.FontSize = fontSize + 2;

save_current_figure(figDir, 'iterations_vs_N');

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
        disableDefaultInteractivity(axs(k));
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

% ========================= Proxy peak memory (for N <= 4) =========================
function bytes = proxy_peak_matlab_bytes(methodName, N, dims, r)
% Deterministic proxy of peak MATLAB host memory (bytes) that avoids zeros caused by memory pools.
% Assumes uniform ranks R_n=L_n=r and double precision (8 bytes/entry).

I_tot = prod(double(dims));
L_tot = double(r)^double(N);
J = double(r)^3; % J = R*L*R = r^3

switch lower(char(methodName))
    case 'baseline'
        % A: L_tot x I_tot, H: L_tot^2, plus X and Xhat (2*I_tot)
        bytes = 8 * (L_tot*I_tot + L_tot^2 + 2*I_tot);
    case 'fast cpu'
        % Q: J x (I_tot/I), H: L_tot^2, plus X and Xhat
        I = double(dims(1));
        bytes = 8 * (J*(I_tot/I) + L_tot^2 + 2*I_tot);
    case 'fast gpu'
        % Host side dominated by full tensors (X and Xhat) + small Gram
        bytes = 8 * (2*I_tot + L_tot^2);
    otherwise
        bytes = NaN;
end
end

function bytes = proxy_peak_gpu_bytes(N, dims, r)
% Deterministic proxy of peak GPU device memory (bytes) for GPU-full solver.
% Dominant device residents: X_g, Xhat_g (2*I_tot), plus Q_n and small J^2.

I_tot = prod(double(dims));
I = double(dims(1));
J = double(r)^3;
bytes = 8 * (3*I_tot + J*(I_tot/I) + J^2);
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
