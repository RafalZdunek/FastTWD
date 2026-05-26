% MAIN  Entry point for the FastTWD quick benchmark.
%
% This script configures the MATLAB path for the repository and runs the
% compact benchmark located in experiments/run_quick_benchmark.m. The benchmark
% compares FastTWD CPU, FastTWD GPU when available, and the GPLv3 baseline TW
% implementation when the baseline files are present.

clear; clc;

rootDir = fileparts(mfilename('fullpath'));
if isempty(rootDir)
    rootDir = pwd;
end

addpath(rootDir);
startupInfo = startup_fasttwd(true);

benchmarkFile = fullfile(startupInfo.rootDir, 'experiments', 'run_quick_benchmark.m');
if exist(benchmarkFile, 'file') ~= 2
    error('FastTWD:Main:MissingBenchmark', ...
        'Cannot find the quick benchmark script: %s', benchmarkFile);
end

run(benchmarkFile);
