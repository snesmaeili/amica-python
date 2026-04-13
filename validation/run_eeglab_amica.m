%% run_eeglab_amica.m — Run EEGLAB AMICA on sub-01 for topoplot comparison
%
% Applies the SAME preprocessing as the Python validation pipeline,
% runs AMICA with matching parameters, generates topoplots.
%
% Usage (via sbatch):
%   module load matlab/2023a.3
%   matlab -nodisplay -nosplash -r "run('validation/run_eeglab_amica.m'); exit"

%% Setup paths
eeglab_path = dir(fullfile('/home/sesma/refs', 'eeglab*'));
if isempty(eeglab_path)
    error('EEGLAB not found in /home/sesma/refs/');
end
eeglab_dir = fullfile(eeglab_path(1).folder, eeglab_path(1).name);
addpath(eeglab_dir);

% Add AMICA plugin
addpath('/home/sesma/refs/sccn-amica');

% Initialize EEGLAB without GUI, skip network plugin check
% Set option to not check for updates (compute nodes may lack internet)
try
    pop_editoptions('option_checkversion', 0);
catch
end
eeglab nogui;

outdir = '/home/sesma/amica-python/validation/results/eeglab_amica';
if ~exist(outdir, 'dir'), mkdir(outdir); end

%% Load sub-01
fprintf('Loading sub-01...\n');
set_file = '/home/sesma/scratch/ds004505/sourcedata/Merged/sub-01/sub-01_Merged.set';
EEG = pop_loadset('filename', 'sub-01_Merged.set', ...
    'filepath', '/home/sesma/scratch/ds004505/sourcedata/Merged/sub-01/');
fprintf('Loaded: %d ch, %d samples, sfreq=%g Hz\n', EEG.nbchan, EEG.pnts, EEG.srate);

%% Classify and pick channels (match Python pipeline)
% Keep only scalp EEG — drop noise (N-*), unused (None*), EMG, acc, cometas
scalp_idx = [];
drop_labels = {};
for i = 1:EEG.nbchan
    label = EEG.chanlocs(i).labels;
    if startsWith(label, 'N-') || startsWith(label, 'None') || ...
       contains(label, 'ISCM') || contains(label, 'SSCM') || ...
       contains(label, 'STrap') || contains(label, 'ITrap') || ...
       startsWith(label, 'CGY') || startsWith(label, 'CWR') || ...
       startsWith(label, 'NGY') || startsWith(label, 'NWR') || ...
       startsWith(label, 'Imu_') || startsWith(label, 'Emg_')
        drop_labels{end+1} = label;
    else
        scalp_idx(end+1) = i;
    end
end
EEG = pop_select(EEG, 'channel', scalp_idx);
fprintf('After channel selection: %d channels\n', EEG.nbchan);

%% Drop channels not in standard_1005 montage (O9, FP2)
% These are the same channels Python drops
drop_ch = {'O9', 'FP2'};
for i = 1:length(drop_ch)
    idx = find(strcmp({EEG.chanlocs.labels}, drop_ch{i}));
    if ~isempty(idx)
        EEG = pop_select(EEG, 'nochannel', idx);
        fprintf('Dropped %s\n', drop_ch{i});
    end
end

%% Look up channel locations
EEG = pop_chanedit(EEG, 'lookup', fullfile(eeglab_dir, 'plugins', 'dipfit', 'standard_BEM', 'elec', 'standard_1005.elc'));
fprintf('Channel locations loaded: %d channels\n', EEG.nbchan);

%% Resample to 250 Hz
if EEG.srate ~= 250
    EEG = pop_resample(EEG, 250);
    fprintf('Resampled to %g Hz\n', EEG.srate);
end

%% Filter 1-100 Hz
EEG = pop_eegfiltnew(EEG, 'locutoff', 1.0, 'hicutoff', 100.0);
fprintf('Filtered: 1-100 Hz\n');

%% Average reference
EEG = pop_reref(EEG, []);
fprintf('Average referenced\n');

fprintf('Final: %d ch, %d samples, sfreq=%g Hz\n', EEG.nbchan, EEG.pnts, EEG.srate);

%% Run AMICA
fprintf('\n=== Running AMICA ===\n');
amica_outdir = fullfile(outdir, 'amicaout');
if exist(amica_outdir, 'dir'), rmdir(amica_outdir, 's'); end

% Write data to temp .fdt in correct Fortran column-major layout
tmp_fdt = fullfile(amica_outdir, 'input.fdt');
if ~exist(amica_outdir, 'dir'), mkdir(amica_outdir); end
fid = fopen(tmp_fdt, 'w');
fwrite(fid, single(EEG.data), 'float32');  % MATLAB writes column-major by default
fclose(fid);

% Write param file for amica17_narval
param_file = fullfile(amica_outdir, 'params.param');
fid = fopen(param_file, 'w');
fprintf(fid, 'files %s\n', tmp_fdt);
fprintf(fid, 'outdir %s\n', amica_outdir);
fprintf(fid, 'num_models 1\n');
fprintf(fid, 'num_mix_comps 3\n');
fprintf(fid, 'data_dim %d\n', EEG.nbchan);
fprintf(fid, 'field_dim %d\n', EEG.pnts);
fprintf(fid, 'pcakeep 60\n');
fprintf(fid, 'max_threads 4\n');
fprintf(fid, 'block_size 256\n');
fprintf(fid, 'max_iter 500\n');
fprintf(fid, 'writestep 500\n');
fprintf(fid, 'do_history 0\n');
fprintf(fid, 'dble_data 0\n');
fprintf(fid, 'lrate 0.01\n');
fprintf(fid, 'use_grad_norm 1\n');
fprintf(fid, 'use_min_dll 1\n');
fprintf(fid, 'min_grad_norm 0.000001\n');
fprintf(fid, 'min_dll 0.000000001\n');
fprintf(fid, 'do_approx_sphere 1\n');
fprintf(fid, 'do_reject 0\n');
fprintf(fid, 'do_newton 1\n');
fprintf(fid, 'newt_start 50\n');
fprintf(fid, 'num_samples 1\n');
fprintf(fid, 'field_blocksize 1\n');
fprintf(fid, 'minlrate 0.00000001\n');
fprintf(fid, 'lratefact 0.5\n');
fprintf(fid, 'rholrate 0.05\n');
fprintf(fid, 'rho0 1.5\n');
fprintf(fid, 'minrho 1.0\n');
fprintf(fid, 'maxrho 2.0\n');
fprintf(fid, 'rholratefact 0.5\n');
fprintf(fid, 'newt_ramp 10\n');
fprintf(fid, 'newtrate 1.0\n');
fprintf(fid, 'max_decs 3\n');
fprintf(fid, 'invsigmax 100.0\n');
fprintf(fid, 'invsigmin 0.00000001\n');
fprintf(fid, 'do_mean 1\n');
fprintf(fid, 'do_sphere 1\n');
fprintf(fid, 'doPCA 1\n');
fprintf(fid, 'doscaling 1\n');
fprintf(fid, 'fix_init 0\n');
fprintf(fid, 'share_comps 0\n');
fclose(fid);

% Use amica17 compiled against StdEnv/2020 (same as MATLAB's environment).
% Same algorithm as the Python version (validated in FORTRAN_VALIDATION_GUIDE.md).
amica_bin = '/home/sesma/refs/sccn-amica/amica17_std2020';
setenv('OMP_NUM_THREADS', '4');

fprintf('Running: %s %s\n', amica_bin, param_file);
[status, cmdout] = system(sprintf('%s %s', amica_bin, param_file));
fprintf('%s\n', cmdout);
if status ~= 0
    warning('amica17 exited with status %d', status);
end

fprintf('AMICA completed\n');

%% Load AMICA results
mods = loadmodout15(amica_outdir);
EEG.icaweights = mods.W;
EEG.icasphere = mods.S(1:size(mods.W,1), :);
EEG.icawinv = pinv(EEG.icaweights * EEG.icasphere);
EEG.icachansind = 1:EEG.nbchan;

fprintf('W: %dx%d, S: %dx%d\n', size(EEG.icaweights), size(EEG.icasphere));
fprintf('icawinv: %dx%d\n', size(EEG.icawinv));

%% Generate topoplots
fprintf('\n=== Generating topoplots ===\n');
n_plot = min(20, size(EEG.icaweights, 1));

figure('visible', 'off', 'Position', [100 100 1200 900]);
for i = 1:n_plot
    subplot(4, 5, i);
    topoplot(EEG.icawinv(:, i), EEG.chanlocs, ...
        'electrodes', 'pts', ...
        'style', 'both', ...
        'numcontour', 6, ...
        'shading', 'interp');
    title(sprintf('IC%d', i));
    colormap(jet);
end
sgtitle('EEGLAB AMICA - sub-01');
print(fullfile(outdir, 'topoplots_eeglab.png'), '-dpng', '-r150');
fprintf('Saved topoplots to %s\n', fullfile(outdir, 'topoplots_eeglab.png'));

%% Save LL trajectory
LL = mods.LL;
save(fullfile(outdir, 'eeglab_results.mat'), 'LL', 'mods');
fprintf('Saved results to %s\n', fullfile(outdir, 'eeglab_results.mat'));

%% Print summary
fprintf('\n=== SUMMARY ===\n');
fprintf('n_iter: %d\n', length(LL));
fprintf('LL: first=%.4f final=%.4f\n', LL(1), LL(end));
if isfield(mods, 'rho')
    rho = mods.rho;
    fprintf('rho range: [%.3f, %.3f]\n', min(rho(:)), max(rho(:)));
    fprintf('rho at floor: %d/%d\n', sum(abs(rho(:) - 1.0) < 1e-6), numel(rho));
end

fprintf('\nDONE\n');
