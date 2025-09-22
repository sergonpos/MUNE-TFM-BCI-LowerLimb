% ---------------- Configuration ----------------
clear; clc;
folder = 'C:\Users\SGP02\MEDUSA\v2024\data';
javaaddpath('C:\Users\SGP02\OneDrive\Documentos\Neurotech UPM\TFM\mongo-java-driver-3.12.14.jar');
pattern = '*.rec.bson';

SEGMENTS_S = [  5 125;
              130 250;
              255 375;
              380 500];  % [start end] (s), rows correspond to LABELS
LABELS     = {'MI + ATT','BL','MI + ATT','BL'};
EPOCH_LEN_S = 5;         % seconds

% Welch params (roughly match Python: nperseg=256, noverlap=128)
NPERSEG  = 256;
NOVERLAP = 128;

% Normalization band for PSD
FMIN_NORM = 0.5;
FMAX_NORM = 30;

% Filters
LOWCUT  = 0.5;
HIGHCUT = 30.0;
NOTCH_HZ = 50.0;
NOTCH_Q  = 30.0;

% ---------------- Load & process all files ----------------
files = dir(fullfile(folder, pattern));
fprintf('Found %d files matching %s\n', numel(files), pattern);

X_all = [];         % will become [N_ep, L, C]
y_all = strings(0,1);
meta_all = table();

for i = 1:numel(files)
    fname = fullfile(files(i).folder, files(i).name);
    try
        doc = readBSON(fname);                             % -> struct with doc.eeg.fs, doc.eeg.signal
    catch ME
        warning('Could not read %s: %s', files(i).name, ME.message);
        continue;
    end

    if ~isfield(doc,'eeg') || ~isfield(doc.eeg,'signal') || ~isfield(doc.eeg,'fs')
        warning('File %s missing eeg.signal or eeg.fs. Skipping.', files(i).name);
        continue;
    end

    fs = double(doc.eeg.fs);
    signal = ensureTimeFirst(doc.eeg.signal);              % [samples, channels]
    signal = preprocessContinuous(signal, fs, LOWCUT, HIGHCUT, NOTCH_HZ, NOTCH_Q);

    [X, y, meta] = processOneDoc(signal, fs, files(i).name, SEGMENTS_S, LABELS, EPOCH_LEN_S);

    if ~isempty(X)
        % Concatenate
        if isempty(X_all)
            X_all = X;
        else
            X_all = cat(1, X_all, X);
        end
        y_all  = [y_all; y]; %#ok<AGROW>
        meta_all = [meta_all; meta]; %#ok<AGROW>
    end
end

if isempty(X_all)
    error('No epochs were generated. Check input files and segment settings.');
end

fprintf('X: [%d, %d, %d] (N_ep, L, C)\n', size(X_all,1), size(X_all,2), size(X_all,3));
fprintf('y: [%d] | unique: %s\n', numel(y_all), strjoin(unique(y_all), ', '));
fprintf('meta rows: %d\n', height(meta_all));
disp(head(meta_all));


%% ---- Split epochs into MI+ATT (trials 1 & 3) and BL (trials 2 & 4) ----
% Prerequisites in workspace: X_all (N_ep x L x C), y_all, meta_all (table)

% Minimal checks
assert(exist('X_all','var')==1 && ndims(X_all)==3, 'X_all missing or wrong shape');
assert(exist('meta_all','var')==1 && istable(meta_all), 'meta_all table is required');
assert(ismember('trial_idx', meta_all.Properties.VariableNames), 'meta_all must contain trial_idx');
if ~ismember('label', meta_all.Properties.VariableNames)
    meta_all.label = strings(height(meta_all),1);  % just in case
end

N = size(X_all,1);
assert(height(meta_all)==N, 'meta_all rows (%d) must match X_all epochs (%d)', height(meta_all), N);

% Primary mask by trial_idx
is_miatt_trials = ismember(meta_all.trial_idx, [1 3]);
is_bl_trials    = ismember(meta_all.trial_idx, [2 4]);

% (Optional) verification by label, in case some row lacks proper trial_idx
lab = upper(strtrim(string(meta_all.label)));
is_miatt_labels = lab=="MI + ATT" | lab=="MI+ATT" | lab=="MI_ATT";
is_bl_labels    = lab=="BL" | lab=="BASELINE" | lab=="REST";

% Combine (trial_idx prevails; labels as fallback)
is_MIATT = is_miatt_trials | (is_miatt_labels & ~is_bl_trials);
is_BL    = is_bl_trials    | (is_bl_labels    & ~is_miatt_trials);

% Quick warnings
if ~any(is_MIATT), warning('No MI+ATT epochs found (trials 1&3).'); end
if ~any(is_BL),    warning('No BL epochs found (trials 2&4).');     end

% Extract 3D matrices (epochs)
X_MIATT = X_all(is_MIATT, :, :);   % [N_miatt x L x C]
X_BL    = X_all(is_BL,    :, :);   % [N_bl    x L x C]

fprintf('X_MIATT: [%d x %d x %d] | X_BL: [%d x %d x %d]\n', ...
    size(X_MIATT,1), size(X_MIATT,2), size(X_MIATT,3), ...
    size(X_BL,1),    size(X_BL,2),    size(X_BL,3));

% Convert to continuous [samples x channels] by stacking epochs
toContinuous = @(X3) reshape(X3, [], size(X3,3));  % stack (N_ep*L) x C
DATA_MIATT = toContinuous(X_MIATT);   % [samples x C]
DATA_BL    = toContinuous(X_BL);      % [samples x C]

fprintf('DATA_MIATT (continuous): [%d x %d]\n', size(DATA_MIATT,1), size(DATA_MIATT,2));
fprintf('DATA_BL    (continuous): [%d x %d]\n', size(DATA_BL,1), size(DATA_BL,2));

idx_MIATT = find(is_MIATT);
idx_BL    = find(is_BL);

%% ==== Mean PSDn by epochs and channels (MI+ATT vs BL) ====
% Requirements: X_MIATT (N1 x L x C), X_BL (N2 x L x C)
% Try to take fs from meta_all.fs if available; otherwise define fs manually.

% ---- Welch and normalization parameters ----
NPERSEG   = 256;
NOVERLAP  = 128;
FMIN_NORM = 1;     % Hz
FMAX_NORM = 30;    % Hz

% ---- fs resolution ----
if exist('fs','var') ~= 1
    if exist('meta_all','var')==1 && istable(meta_all) && ismember('fs', meta_all.Properties.VariableNames)
        ufs = unique(double(meta_all.fs));
        if numel(ufs) > 1
            warning('Multiple sampling rates found in meta_all.fs: %s. Using the first one.', mat2str(ufs));
        end
        fs = ufs(1);
    else
        error('Define fs (Hz) or provide meta_all.fs. Example: fs = 256;');
    end
end

% ---- Vectorized computation of mean PSDn (by epochs and channels) ----
[freqs, PSDN_MIATT_mean, PSDN_MIATT_sem] = psdn_mean_epochs_channels(X_MIATT, fs, NPERSEG, NOVERLAP, FMIN_NORM, FMAX_NORM);
[freqs2, PSDN_BL_mean,    PSDN_BL_sem]   = psdn_mean_epochs_channels(X_BL,    fs, NPERSEG, NOVERLAP, FMIN_NORM, FMAX_NORM);

assert(isequal(freqs, freqs2), 'Frequency grids differ between conditions.');

% Parameters
opts.nperseg   = 256;
opts.noverlap  = 128;
opts.norm_band = [1 30];   % normalize 1–40 Hz
opts.plot_band = [2 30];   % return only 1–40 Hz
opts.hp_hz     = 1.0;      % high-pass 1 Hz (zero-phase). Set 0 to disable
opts.detrend   = true;     % remove epoch mean

% MI+ATT
[f_MI, PSDN_MI_mean, PSDN_MI_sem] = psdn_mean_epochs_channels_clean(X_MIATT, fs, opts);
% BL
[f_BL, PSDN_BL_mean, PSDN_BL_sem] = psdn_mean_epochs_channels_clean(X_BL, fs, opts);

assert(isequal(f_MI,f_BL));
freqs = f_MI;

% Plot
figure('Color','w'); hold on;
plot(freqs, PSDN_MI_mean, 'LineWidth', 1.5, 'DisplayName','MI+ATT');
plot(freqs, PSDN_BL_mean, 'LineWidth', 1.5, 'DisplayName','BL');
xlim([opts.plot_band(1) opts.plot_band(2)]); grid on;
xlabel('Frequency (Hz)', 'FontSize', 20);
ylabel('PSDn', 'FontSize', 20);
set(gca,'FontSize',16);
lgd = legend('Location','best');
lgd.FontSize = 16;            % increase font size
lgd.ItemTokenSize = [50, 30]; % adjust legend box/symbol size

%% ===== Relative band power: compute, boxplots, and topoplots =====
% Needs in workspace: X_MIATT, X_BL, fs (Hz)
% Optional (for topoplots): chanlocs (EEGLAB chanlocs struct, length = size(X_*,3))

addpath('C:\Users\SGP02\OneDrive\Documentos\Neurotech UPM\TFM\eeglab2025.0.0');
eeglab; close;

% ---------- PSD options ----------
opts.nperseg   = 256;
opts.noverlap  = 128;
opts.norm_band = [1 40];   % normalize total power in 1–40 Hz
opts.hp_hz     = 1.0;      % high-pass 1 Hz to reduce LF/offset (set 0 to disable)
opts.detrend   = true;

% ---------- Bands ----------
bands = struct( ...
  'theta',[4 8], ...
  'mu',   [8 12], ...
  'alpha',[8 13], ...
  'beta', [13 30], ...
  'wide', [8 30] );
band_order_all   = {'theta','mu','alpha','beta','wide'};
band_order_attn  = {'theta','alpha','beta'};
band_order_mi    = {'mu','beta','wide'};

% ---------- PSDn per epoch & channel ----------
[freqs, Pn_MI] = psdn_epochs_channels_clean(X_MIATT, fs, opts);   % [N1 x F x C]
[~,     Pn_BL] = psdn_epochs_channels_clean(X_BL,    fs, opts);   % [N2 x F x C]

% ---------- Relative band power per epoch & channel ----------
rel_MI = struct(); rel_BL = struct();
for k = 1:numel(band_order_all)
    nm = band_order_all{k};
    rng = bands.(nm);
    mask = freqs >= rng(1) & freqs <= rng(2);
    rel_MI.(nm) = squeeze(mean(Pn_MI(:,mask,:), 2));  % [N1 x C]
    rel_BL.(nm) = squeeze(mean(Pn_BL(:,mask,:), 2));  % [N2 x C]
end


%% ===== Boxplots by paradigm =====
% Bands by paradigm
band_order_attn = {'theta','alpha','beta'};
band_order_mi   = {'mu','beta','wide'};

% Colors
blColor = [0.85 0.325 0.098];   % BL
miColor = [0.00 0.447 0.741];   % MI

% --- ATTENTION (θ, α, β) ---
figure('Color','w','Name','Attention paradigm — relative band power');
tiledlayout(1, numel(band_order_attn), 'TileSpacing','compact','Padding','compact');

for k = 1:numel(band_order_attn)
    nm = band_order_attn{k};

    % 1) Vector per epoch (average across channels), same as before
    vBL_ep = mean(rel_BL.(nm), 2, 'omitnan');    % N_BL x 1
    vMI_ep = mean(rel_MI.(nm), 2, 'omitnan');    % N_MI x 1

    % 2) File labels per epoch
    files_BL = string(meta_all.file(idx_BL));      % N_BL x 1
    files_MI = string(meta_all.file(idx_MIATT));   % N_MI x 1

    % 3) Aggregate by file (mean across epochs)
    [uBL,~,gBL] = unique(files_BL);
    xBL_file = accumarray(gBL, vBL_ep, [], @(x) mean(x,'omitnan'));   % numel(uBL) x 1

    [uMI,~,gMI] = unique(files_MI);
    xMI_file = accumarray(gMI, vMI_ep, [], @(x) mean(x,'omitnan'));   % numel(uMI) x 1

    % 4) Align common files and build paired vectors
    [uCommon, ia, ib] = intersect(uBL, uMI, 'stable');
    xBL = xBL_file(ia);           % 1 value per file (BL)
    xMI = xMI_file(ib);           % 1 value per file (MI)

    nexttile; hold on;
    % IMPORTANT: indicate 'paired', true to use signrank
    draw_boxpair_colored_outliers(xBL, xMI, upper(nm), blColor, miColor, ...
    'qlo',0.01,'qhi',0.99,'paired',true);
    title(sprintf(upper(nm), numel(uCommon)));
end


% --- MOTOR IMAGERY (μ, β, wide) ---
figure('Color','w','Name','MI paradigm — relative band power');
tiledlayout(1, numel(band_order_mi), 'TileSpacing','compact','Padding','compact');

for k = 1:numel(band_order_mi)
    nm = band_order_mi{k};

    % 1) One value per epoch (average across channels), as baseline
    vBL_ep = mean(rel_BL.(nm), 2, 'omitnan');    % N_BL x 1
    vMI_ep = mean(rel_MI.(nm), 2, 'omitnan');    % N_MI x 1

    % 2) File labels per epoch
    files_BL = string(meta_all.file(idx_BL));      % N_BL x 1
    files_MI = string(meta_all.file(idx_MIATT));   % N_MI x 1

    % 3) Aggregate by file (mean across epochs)
    [uBL,~,gBL] = unique(files_BL);
    xBL_file = accumarray(gBL, vBL_ep, [], @(x) mean(x,'omitnan'));   % numel(uBL) x 1

    [uMI,~,gMI] = unique(files_MI);
    xMI_file = accumarray(gMI, vMI_ep, [], @(x) mean(x,'omitnan'));   % numel(uMI) x 1

    % 4) Align common files (pairing) and build paired vectors
    [uCommon, ia, ib] = intersect(uBL, uMI, 'stable');
    xBL = xBL_file(ia);   % 1 value per file (BL)
    xMI = xMI_file(ib);   % 1 value per file (MI)

    nexttile; hold on;
    % Call with positionals + paired test
    draw_boxpair_colored_outliers(xBL, xMI, upper(nm), blColor, miColor, ...
        'qlo',0.01,'qhi',0.99,'paired',true);

    title(sprintf(upper(nm), numel(uCommon)));
end

% --- whisker extremes and Y adjustment ---
uw = findobj(gca,'Tag','Upper Adjacent Value');   % upper whisker endpoints
lw = findobj(gca,'Tag','Lower Adjacent Value');   % lower whisker endpoints
uwY = cell2mat(get(uw,'YData'));                  % may come as cell -> vector
lwY = cell2mat(get(lw,'YData'));
yl  = [min(lwY(:),[],'omitnan'), max(uwY(:),[],'omitnan')];
if ~isfinite(yl(1)) || ~isfinite(yl(2)) || yl(1)==yl(2)
    yl = ylim;  % fallback
end
pad = 0.06 * max(1e-12, yl(2)-yl(1));
ylim([max(0, yl(1)-pad), yl(2)+pad]);   % relative power >= 0

%% ================== 3×3 TOPOPLOTS PER PARADIGM ==================
locsFolder = 'C:\Users\SGP02\OneDrive\Documentos\Neurotech UPM\TFM\eeglab2025.0.0\sample_locs';
addpath(locsFolder);
locsFile = fullfile(locsFolder,'Standard-10-20-Cap19.locs');

% ==== 2) Read full electrode template (.locs) ====
allstd = readlocs(locsFile);   % EEGLAB auto-detects format

% ==== 3) Build chanlocs in your specific order ====
labelsWanted = {'F3','C3','P3','Fz','Cz','F4','C4','P4'};  % <-- your order
chanlocs = build_chanlocs_from_locs(allstd, labelsWanted);

% (optional) 4) Quick visual check
%figure; topoplot(zeros(1,numel(chanlocs)), chanlocs); title('Layout 8 ch');

% ==== 5) Run the 3×3 topoplots (uses the block provided above) ====
paired = true;   % use paired test at file level

make_topos_rows_axescolormap('ATTENTION',     {'theta','alpha','beta'}, ...
    rel_BL, rel_MI, chanlocs, paired, meta_all, idx_BL, idx_MIATT);

make_topos_rows_axescolormap('MOTOR IMAGERY', {'mu','beta','wide'},     ...
    rel_BL, rel_MI, chanlocs, paired, meta_all, idx_BL, idx_MIATT);


%% =================== Local functions ===================

function S = readBSON(filename)
%READBSON Decode a .bson file into a MATLAB struct.
% Requires MongoDB Java driver 3.x on path (BasicBSONDecoder).
% Falls back to JSON if the file is text.

    fid = fopen(filename,'rb');
    if fid < 0, error('Cannot open %s', filename); end
    cleanup = onCleanup(@() fclose(fid));
    bytes = fread(fid, Inf, '*uint8');

    % Heuristic: if content looks like ASCII/UTF-8 JSON, try jsondecode
    if all(bytes >= 9 & bytes <= 126) && any(bytes==123) && any(bytes==125)
        txt = native2unicode(bytes(:).', 'UTF-8');
        S = jsondecode(txt);
        return;
    end

    % BSON via Java (driver 3.x: org.bson.BasicBSONDecoder)
    try
        import org.bson.*
        dec = BasicBSONDecoder();
        jMap = dec.readObject(bytes);
        S = bsonToStruct(jMap);
    catch ME
        error('BSON decode failed: %s\nHint: add mongo-java-driver-3.12.x.jar via javaaddpath.', ME.message);
    end
end

function S = bsonToStruct(jObj)
%BSONTOSTRUCT Recursively convert a Java Map/List to MATLAB struct/array.
    import java.util.*
    if isa(jObj,'java.util.Map')
        keys = jObj.keySet().toArray();
        S = struct();
        for k = 1:numel(keys)
            key = char(keys(k));
            val = jObj.get(keys(k));
            S.(matlab.lang.makeValidName(key)) = bsonToStruct(val);
        end
    elseif isa(jObj,'java.util.List')
        n = jObj.size();
        cellArr = cell(1,n);
        for i = 1:n
            cellArr{i} = bsonToStruct(jObj.get(i-1));
        end
        % try to make numeric if possible
        if all(cellfun(@(x)isnumeric(x)&&isscalar(x), cellArr))
            S = cellfun(@double, cellArr);
        else
            S = cellArr;
        end
    elseif isa(jObj,'org.bson.types.Binary')
        S = typecast(jObj.getData(), 'uint8');
    elseif isa(jObj,'org.bson.types.ObjectId')
        S = char(jObj.toHexString());
    elseif isa(jObj,'java.lang.Double') || isa(jObj,'java.lang.Integer') || isa(jObj,'java.lang.Long')
        S = double(jObj);
    elseif isa(jObj,'java.lang.Boolean')
        S = logical(jObj);
    elseif isa(jObj,'java.lang.String')
        S = char(jObj);
    else
        try
            S = double(jObj);
        catch
            S = char(string(jObj));
        end
    end
end

function X = ensureTimeFirst(arr)
% Ensure shape [samples x channels] and convert to double.
% Accepts:
%   - numeric [S x C] or [C x S]
%   - 1xC (or Cx1) cell with each channel as vector (1xS or Sx1)
%   - nested cells (e.g., { {channel1}, {channel2}, ... })

    % 1) Unnest if it is a { 1x1 cell } repeatedly
    while iscell(arr) && numel(arr) == 1
        arr = arr{1};
    end

    if iscell(arr)
        % Typical case: cell of channels
        if isvector(arr) && all(cellfun(@(x) isnumeric(x) && isvector(x), arr))
            C = numel(arr);
            % Ensure same number of samples in all channels
            lens = cellfun(@(v) numel(v), arr);
            if any(lens ~= lens(1))
                error('Channels have different lengths: %s', mat2str(lens));
            end
            S = lens(1);
            X = zeros(S, C);
            for c = 1:C
                v = arr{c}(:);         % column
                X(:, c) = double(v);
            end
            return;
        end

        % Another case: 2D cell with scalars (grid) -> try cell2mat
        if all(cellfun(@(x) isnumeric(x) && isscalar(x), arr(:)))
            try
                M = cell2mat(arr);
                X = double(M);
                if size(X,1) < size(X,2), X = X.'; end
                return;
            catch
                error('Cannot convert 2D cell to numeric matrix.');
            end
        end

        error('Unrecognized cell format for eeg.signal.');
    else
        % Already numeric
        if ~isnumeric(arr) || ndims(arr) ~= 2
            error('eeg.signal must be 2D numeric or a cell of vectors.');
        end
        X = double(arr);
        if size(X,1) < size(X,2)
            X = X.';   % [samples x channels]
        end
    end
end


function x = preprocessContinuous(signal, fs, lowcut, highcut, notchHz, notchQ)
% Detrend -> Notch -> Band-pass -> CAR, with padding if the signal is short.
    x = double(signal);
    if size(x,1) < size(x,2), x = x.'; end   % [samples x channels]

    % 0) Remove DC
    x = detrend(x, 'constant');

    % 1) 50 Hz Notch (order 2)
    w0 = notchHz / (fs/2);
    if w0 > 0 && w0 < 1
        [bNotch,aNotch] = iirnotch(w0, w0/notchQ);
        x = safeFiltfilt(bNotch, aNotch, x);
    end

    % 2) Band-pass 0.1–40 Hz (Butterworth, lower order if needed)
    nyq = fs/2;
    lo = max(lowcut/nyq, 1e-6);
    hi = min(highcut/nyq, 0.9999);
    if lo < hi
        ordList = [4 3 2 1];  % try from higher to lower order
        done = false;
        for ord = ordList
            try
                [bBP,aBP] = butter(ord, [lo hi], 'bandpass');
                x = safeFiltfilt(bBP, aBP, x);
                done = true;
                break
            catch
                % try with lower order
            end
        end
        if ~done
            warning('Could not apply a stable band-pass; continuing without it.');
        end
    end

    % 3) CAR
    x = x - mean(x, 2);
end

function y = safeFiltfilt(b, a, x)
% Apply filtfilt with padding if the signal is too short.
    Lx = size(x,1);
    filtOrd = max(length(a), length(b)) - 1;
    minLen = max(3*filtOrd, 1);  % filtfilt criterion

    if Lx > minLen
        y = filtfilt(b, a, x);
        return
    end

    % Reflection padding until reaching sufficient length
    need = minLen + 1 - Lx;
    padL = floor(need/2);
    padR = need - padL;

    if Lx == 1
        xpad = [repmat(x, padL, 1); x; repmat(x, padR, 1)];
    else
        padL = min(padL, Lx-1);
        padR = min(padR, Lx-1);
        left  = flipud(x(1:padL, :));
        right = flipud(x(end-padR+1:end, :));
        xpad = [left; x; right];
    end

    ypad = filtfilt(b, a, xpad);
    y = ypad(padL+1 : padL+Lx, :);
end

function [epochs, epochStarts] = epocharSegment(seg, fs, epochLenS)
% Cut a continuous segment [samples x channels] into non-overlapping epochs
% Return [N_ep, L, C] and vector of start times (s) relative to segment start.
    L = round(epochLenS * fs);
    n = floor(size(seg,1) / L);
    if n == 0
        epochs = zeros(0, L, size(seg,2));
        epochStarts = zeros(0,1);
        return;
    end
    seg = seg(1:n*L, :);
    C = size(seg,2);
    epochs = reshape(seg, L, n, C);
    epochs = permute(epochs, [2 1 3]);   % [n, L, C]
    epochStarts = (0:n-1)' * epochLenS;
end

function [X, y, meta] = processOneDoc(signal, fs, sourceName, segmentsS, labels, epochLenS)
% For one file: cut into the 4 segments, epoch to 5 s, return X [N_ep,L,C], y [N_ep,1], and meta table.
    Xs = {};
    ys = strings(0,1);
    metaRows = [];

    nChannels = size(signal,2);

    for k = 1:size(segmentsS,1)
        a = segmentsS(k,1); b = segmentsS(k,2);
        lab = string(labels{k});
        i0 = max(1, round(a*fs)+1);
        i1 = min(size(signal,1), round(b*fs));
        seg = signal(i0:i1, :);

        [ep, epStarts] = epocharSegment(seg, fs, epochLenS);
        if isempty(ep), continue; end

        % meta per epoch
        nEp = size(ep,1);
        metaRows = [metaRows; table( ...
            repmat(string(sourceName), nEp,1), ...
            repmat(k, nEp,1), ...
            repmat(lab, nEp,1), ...
            (0:nEp-1)' , ...
            a + epStarts, ...
            a + epStarts + epochLenS, ...
            repmat(fs, nEp,1), ...
            repmat(nChannels, nEp,1), ...
            'VariableNames', {'file','trial_idx','label','epoch_idx','epoch_start_s','epoch_end_s','fs','n_channels'} )]; %#ok<AGROW>

        Xs{end+1} = ep; %#ok<AGROW>
        ys = [ys; repmat(lab, nEp, 1)]; %#ok<AGROW>
    end

    if isempty(Xs)
        X = []; y = strings(0,1); meta = metaRows; return;
    end
    X = cat(1, Xs{:});
    y = ys;
    meta = metaRows;
end


function [f, Pn] = psdn_epochs_channels_clean(X, fs, opts)
    % X: [N x L x C] -> Pn: [N x F x C] (PSDn, normalized in opts.norm_band)
    [N,L,C] = size(X);
    nperseg  = min(opts.nperseg, L);
    noverlap = min(opts.noverlap, nperseg-1);

    % reshape to [L x (N*C)]
    X2 = reshape(permute(X,[2 1 3]), L, N*C);

    % detrend & high-pass
    if isfield(opts,'detrend') && opts.detrend
        X2 = X2 - mean(X2,1);
    end
    if isfield(opts,'hp_hz') && opts.hp_hz > 0
        Wn = opts.hp_hz/(fs/2); [b,a] = butter(4, Wn, 'high');
        X2 = filtfilt(b,a,X2);
        X2 = X2 - mean(X2,1);
    end

    [P, f] = pwelch(X2, nperseg, noverlap, nperseg, fs, 'onesided');  % [F x (N*C)]

    % normalize in band
    bandN = f >= opts.norm_band(1) & f <= opts.norm_band(2);
    total = sum(P(bandN,:),1) + eps;
    Pn2 = P ./ total;

    % back to [N x F x C]
    F = size(Pn2,1);
    Pn = permute(reshape(Pn2, F, N, C), [2 1 3]);
end

function s = significance_star(p)
    if p < 0.001, s = '***';
    elseif p < 0.01, s = '**';
    elseif p < 0.05, s = '*';
    else, s = 'n.s.';
    end
end


function [f, psdn_mean, psdn_sem] = psdn_mean_epochs_channels_clean(X, fs, opts)
    % X: [N x L x C]
    [N,L,C] = size(X);
    X2 = reshape(permute(X,[2 1 3]), L, N*C);  % [L x (N*C)] (epoch-channel as "traces")

    % (1) Simple detrend: remove mean per trace
    if isfield(opts,'detrend') && opts.detrend
        X2 = X2 - mean(X2,1);
    end

    % (2) Zero-phase high-pass (optional)
    if isfield(opts,'hp_hz') && opts.hp_hz > 0
        Wn = opts.hp_hz/(fs/2);
        [b,a] = butter(4, Wn, 'high');
        X2 = filtfilt(b,a,X2);
        % in case of slight offset:
        X2 = X2 - mean(X2,1);
    end

    % Welch
    nperseg  = min(opts.nperseg, L);
    noverlap = min(opts.noverlap, nperseg-1);
    [P, f] = pwelch(X2, nperseg, noverlap, nperseg, fs, 'onesided');  % [F x traces]

    % Normalization in band
    fmin = opts.norm_band(1); fmax = opts.norm_band(2);
    bandN = (f >= fmin) & (f <= fmax);
    total_pow = sum(P(bandN,:), 1) + eps;
    Pn = P ./ total_pow;

    % Limit to desired output band
    fminP = opts.plot_band(1); fmaxP = opts.plot_band(2);
    bandP = (f >= fminP) & (f <= fmaxP);
    f = f(bandP);
    Pn = Pn(bandP, :);

    % Mean and SEM across all traces (epochs*channels)
    psdn_mean = mean(Pn, 2);
    psdn_sem  = std(Pn, 0, 2) ./ sqrt(size(Pn,2));
end


function [f, psdn_mean, psdn_sem] = psdn_mean_epochs_channels(X, fs, nperseg, noverlap, fmin, fmax)
% X: [N x L x C], returns:
%   f:          [F x 1] frequencies
%   psdn_mean:  [F x 1] mean normalized PSD (1–40 Hz) across N*C traces
%   psdn_sem:   [F x 1] corresponding SEM (useful for shading)

    if isempty(X), f=[]; psdn_mean=[]; psdn_sem=[]; return; end
    [N,L,C] = size(X);
    nperseg = min(nperseg, L);

    % Reorganize to evaluate all traces (epoch-channel) at once
    X2 = reshape(permute(X, [2 1 3]), L, N*C);   % [L x (N*C)]

    % Vectorized Welch per column
    [P, f] = pwelch(X2, nperseg, noverlap, nperseg, fs, 'onesided');  % P: [F x (N*C)]

    % 1–40 Hz normalization per trace (column)
    band = (f >= fmin) & (f <= fmax);
    total_pow = sum(P(band, :), 1) + eps;
    Pn = P ./ total_pow;   % normalized PSD per column

    % Mean and SEM across all columns (epochs*channels)
    psdn_mean = mean(Pn, 2);
    psdn_sem  = std(Pn, 0, 2) ./ sqrt(size(Pn,2));
end


function draw_boxpair_colored_outliers(x_BL, x_MI, ttl, blColor, miColor, varargin)
% Draw a pair of boxplots (BL vs MI) with:
% - outliers 'o' colored by group
% - soft box fill
% - per-panel YLim with robust quantiles
% Options: 'qlo','qhi' for quantiles (defaults 0.01,0.99)

    p = inputParser;
    addParameter(p,'qlo',0.01,@(x)isnumeric(x)&&isscalar(x));
    addParameter(p,'qhi',0.99,@(x)isnumeric(x)&&isscalar(x));
    addParameter(p,'paired',false,@(x)islogical(x)&&isscalar(x));
    parse(p,varargin{:});
    qlo = p.Results.qlo; qhi = p.Results.qhi; isPaired = p.Results.paired;

    % Data + groups
    X = [x_BL; x_MI];
    G = [repmat(categorical("BL"), numel(x_BL), 1); ...
         repmat(categorical("MI"), numel(x_MI), 1)];

    % Boxplot (no fill; paint later) + outliers as 'o'
    boxplot(X, G, 'Symbol','ko', 'OutlierSize',5, 'Colors',[0 0 0; 0 0 0]);
    set(findobj(gca,'Tag','Median'),'LineWidth',1.5);
    grid on; box on;
    ylabel('Relative Power', 'FontSize', 20);
    title(ttl);
    set(gca,'XTickLabel',{'BL','MI+ATT'}, 'FontSize', 18);

    % ---- Soft fill of boxes and group-colored outlines ----
    hBox = findobj(gca,'Tag','Box');              % reversed order
    for j = 1:numel(hBox)
        xData = get(hBox(j),'XData');  yData = get(hBox(j),'YData');
        xCenter = round(mean(xData));            % 1 = BL, 2 = MI
        if xCenter == 1
            col = blColor;
        else
            col = miColor;
        end
        % outline
        set(hBox(j),'Color',col,'LineWidth',1.2);
        % fill
        patch(xData, yData, col, 'FaceAlpha',0.18, 'EdgeColor','none');
    end

    % ---- Color outliers by group (based on their X) ----
    hOut = findobj(gca,'Tag','Outliers');
    for j = 1:numel(hOut)
        xg = mean(get(hOut(j),'XData'));
        if round(xg)==1
            set(hOut(j),'Marker','o','MarkerEdgeColor',blColor,'MarkerFaceColor','w','LineWidth',1.0);
        else
            set(hOut(j),'Marker','o','MarkerEdgeColor',miColor,'MarkerFaceColor','w','LineWidth',1.0);
        end
    end

    % ---- Robust Y scale per panel ----
    lo = quantile([x_BL; x_MI], qlo, 'all'); 
    hi = quantile([x_BL; x_MI], qhi, 'all');
    if ~isfinite(lo) || ~isfinite(hi) || lo==hi
        lo = min([x_BL; x_MI],[],'omitnan'); 
        hi = max([x_BL; x_MI],[],'omitnan');
    end
    pad = 0.06 * max(1e-12, hi - lo);
    ylim([max(0, lo - pad) , hi + pad]);    % relative power ≥ 0

    % ---- Significance bar and asterisk (unpaired) ----
    if isPaired
        assert(numel(x_BL)==numel(x_MI), ...
            'For a paired test, x_BL and x_MI must have the same length (same number of aligned files).');
        m = isfinite(x_BL) & isfinite(x_MI);
        if any(m)
            [pval,~,~] = signrank(x_MI(m), x_BL(m));
        else
            pval = NaN;
        end
    else
        [pval,~,~] = ranksum(x_MI, x_BL);
    end

    yl = ylim; rngY = yl(2) - yl(1);
    ystar = yl(2) - 0.05*rngY;
    plot([1 2],[ystar ystar],'k-','LineWidth',1);
    text(1.5, ystar, significance_star(pval), ...
        'HorizontalAlignment','center','VerticalAlignment','bottom','FontSize',20);
end


function chanlocs = build_chanlocs_from_locs(allstd, labelsWanted)
    % Ensure homogeneous fields
    req = {'labels','X','Y','Z','theta','radius','sph_theta','sph_phi','sph_radius','type','urchan','ref'};
    for j = 1:numel(req)
        if ~isfield(allstd, req{j}), [allstd.(req{j})] = deal([]); end
    end
    fn = fieldnames(allstd(1));
    blank = cell2struct(cell(size(fn)), fn, 1);

    chanlocs = repmat(blank, 1, numel(labelsWanted));
    allLabels = {allstd.labels};

    for k = 1:numel(labelsWanted)
        lab = labelsWanted{k};
        idx = find(strcmpi(allLabels, lab), 1);
        if isempty(idx)
            warning('Canal "%s" no encontrado en %s. Se crea sin coords.', lab, 'Standard-10-20-Cap19.locs');
            s = blank; s.labels = lab;
        else
            s = allstd(idx);
            for j = 1:numel(fn)
                if ~isfield(s, fn{j}), s.(fn{j}) = []; end
            end
        end
        s.urchan = k;
        chanlocs(k) = s;
    end
end


function [Ufiles, M_file_chan] = per_file_channel_mean(M, files)
% M: [N_epochs x C], files: [N_epochs x 1] string/cellstr
    [Ufiles,~,g] = unique(files);
    C = size(M,2);
    M_file_chan = nan(numel(Ufiles), C);
    for i = 1:numel(Ufiles)
        rows = (g==i);
        M_file_chan(i,:) = mean(M(rows,:), 1, 'omitnan'); % mean across epochs
    end
end



function make_topos_rows_axescolormap(figName, band_list, rel_BL, rel_MI, ...
                                      chanlocs, paired, meta_all, idx_BL, idx_MIATT)
    C = numel(chanlocs);
    f = figure('Color','w','Name',['Topomaps — ' figName]);
    tl = tiledlayout(f,3,numel(band_list),'TileSpacing','compact','Padding','compact');

    ax = gobjects(3,numel(band_list));
    CL = nan(numel(band_list),2);   % limits per band (for BL/MI)

    % --- file labels per EPOCH ---
    files_BL = string(meta_all.file(idx_BL));       % N_BL x 1
    files_MI = string(meta_all.file(idx_MIATT));    % N_MI x 1

    for j = 1:numel(band_list)
        nm = band_list{j};

        % ========== (A) Aggregation by file and channel ==========
        % rel_*.(nm): [N_epochs x C]
        [uBL, BL_file_chan] = per_file_channel_mean(rel_BL.(nm), files_BL); % [F_BL x C]
        [uMI, MI_file_chan] = per_file_channel_mean(rel_MI.(nm), files_MI); % [F_MI x C]

        % Align common files (pairing)
        [uCommon, ia, ib] = intersect(uBL, uMI, 'stable');
        BLc = BL_file_chan(ia,:);   % [F x C]  per-file BL
        MIc = MI_file_chan(ib,:);   % [F x C]  per-file MI

        % ========== (B) Means to display in rows 1–2 ==========
        mBL = mean(BLc, 1, 'omitnan');   % 1 x C (mean across files)
        mMI = mean(MIc, 1, 'omitnan');   % 1 x C

        % Common BL/MI scale per band
        clim = [min([mBL mMI],[],'all'), max([mBL mMI],[],'all')];
        if ~isfinite(clim(1)) || ~isfinite(clim(2)) || clim(1)==clim(2), clim = [0 1]; end
        CL(j,:) = clim;

        % ---- Row 1: BL ----
        ax(1,j) = nexttile(tl, j);
        topoplot(mBL, chanlocs, 'maplimits',clim, ...
                 'electrodes','on','interplimits','head','gridscale',64,'whitebk','on');
        axis(ax(1,j),'off'); set(ax(1,j),'Color','w'); colorbar(ax(1,j),'FontSize',12);
        title([upper(nm) ' — BL'], 'FontSize',18);

        % ---- Row 2: MI ----
        ax(2,j) = nexttile(tl, numel(band_list)+j);
        topoplot(mMI, chanlocs, 'maplimits',clim, ...
                 'electrodes','on','interplimits','head','gridscale',64,'whitebk','on');
        axis(ax(2,j),'off'); set(ax(2,j),'Color','w'); colorbar(ax(2,j),'FontSize',12);
        title([upper(nm) ' — MI+ATT'], 'FontSize',18);

        % ========== (C) p-values per channel (paired by file) ==========
        p = nan(1,C);
        for c = 1:C
            x = MIc(:,c); y = BLc(:,c);          % vectors [F x 1], F = #common files
            m = isfinite(x) & isfinite(y);
            if any(m)
                if paired
                    p(c) = signrank(x(m), y(m)); % PAIRED by file
                else
                    p(c) = ranksum(x(m), y(m));  % (not recommended here)
                end
            end
        end

        ax(3,j) = nexttile(tl, 2*numel(band_list)+j);
        topoplot(p, chanlocs, 'maplimits',[0 0.05], ...
                 'electrodes','on','interplimits','head','gridscale',64,'whitebk','on');
        axis(ax(3,j),'off'); set(ax(3,j),'Color','w'); colorbar(ax(3,j),'FontSize',12);
        title([upper(nm) ' — p-value'], 'FontSize',18);
    end

    % ====== colormaps and limits ======
    for j = 1:numel(band_list)
        colormap(ax(1,j),'jet');  caxis(ax(1,j), CL(j,:));
        colormap(ax(2,j),'jet');  caxis(ax(2,j), CL(j,:));
        colormap(ax(3,j),'hot');  caxis(ax(3,j), [0 0.05]); % smaller p = darker
    end
end

