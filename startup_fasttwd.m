function info = startup_fasttwd(includeBaseline)
%STARTUP_FASTTWD Configure MATLAB paths for the FastTWD repository.
%
%   startup_fasttwd()
%   startup_fasttwd(includeBaseline)
%   info = startup_fasttwd(includeBaseline)
%
% The function adds the project source code, experiment scripts, and experiment
% utilities to the MATLAB path. If includeBaseline is true, it also adds the
% GPLv3 baseline Tensor Wheel implementation, if found.
%
% Directory layout assumed by this startup file:
%
%   FastTWD/
%   ├── src/
%   ├── experiments/
%   ├── experiments/utils/
%   └── third_party/Baseline_TW_TC/
%
% For compatibility, the function also accepts the alternative baseline layout:
%
%   third_party/code_TWDec/Baseline_TW_TC/

if nargin < 1 || isempty(includeBaseline)
    includeBaseline = true;
end

rootDir = fileparts(mfilename('fullpath'));
if isempty(rootDir)
    rootDir = pwd;
end

info = struct();
info.rootDir = rootDir;
info.pathsAdded = {};
info.baselineDir = '';
info.hasBaseline = false;

% Add only project paths needed for normal use and experiments.
info = add_project_path(info, fullfile(rootDir, 'src'), '-begin');
info = add_project_path(info, fullfile(rootDir, 'experiments'), '-end');
info = add_project_path(info, fullfile(rootDir, 'experiments', 'utils'), '-end');

if includeBaseline
    baselineCandidates = { ...
        fullfile(rootDir, 'third_party', 'Baseline_TW_TC'), ...
        fullfile(rootDir, 'third_party', 'code_TWDec', 'Baseline_TW_TC'), ...
        fullfile(rootDir, 'Baseline_TW_TC') ...
    };

    for k = 1:numel(baselineCandidates)
        candidate = baselineCandidates{k};
        if exist(fullfile(candidate, 'inc_TW_TC.m'), 'file') == 2
            info.baselineDir = candidate;
            info.hasBaseline = true;
            info = add_project_path(info, candidate, '-end');
            break;
        end
    end
end

% Report only when the function is called without an output argument.
if nargout == 0
    fprintf('FastTWD startup completed.\n');
    fprintf('Repository root: %s\n', info.rootDir);
    if info.hasBaseline
        fprintf('Baseline TW directory: %s\n', info.baselineDir);
    elseif includeBaseline
        fprintf('Baseline TW directory: not found; baseline benchmarks will be skipped.\n');
    end
end

end

function info = add_project_path(info, pathName, positionFlag)
%ADD_PROJECT_PATH Add an existing directory to the MATLAB path once.
if exist(pathName, 'dir') == 7
    % Avoid accumulating duplicate copies of the same project path.
    try
        rmpath(pathName);
    catch
    end
    addpath(pathName, positionFlag);
    info.pathsAdded{end+1} = pathName;
end
end
