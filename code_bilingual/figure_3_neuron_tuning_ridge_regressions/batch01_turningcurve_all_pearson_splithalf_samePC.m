%% cross-language tuning similarity via ridge regression on shared mBERT PCs
% fits ridge regression from shared-PCA mBERT embeddings to neural firing
% rates, then compares english and spanish encoding weights per neuron.
% includes permutation tests, binomial tests, and split-half reliability.
%
% required input:
%   - matched spike data files (firing rates per language)
%   - mBERT embedding files per language
%
% output:
%   - neuron-wise cross-language weight similarity
%   - permutation-based significance (global and per-neuron)
%   - binomial tests on proportion of significant neurons
%   - split-half reliability vs cross-language comparison
% xinyuanyan

clear all; close all;

%% ============================================================

data_files = {
    'matched_podcast1_spike_data.mat',
    'matched_podcast2_spike_data.mat',
    'matched_podcast3_spike_data.mat'
};

% mBERT embedding paths
mbert_dir = '2_mbert_whisper_kurzgesagt_words_matched';
mbert_file = 'mbert_features_extracted.mat';
mbert_layer = 12;           % layer index (1-indexed)
mbert_col_start = 5;        % skip metadata columns

% analysis parameters
n_components = 100;         % number of shared PCs
n_perms = 1000;             % permutations for significance testing
n_splits = 1000;            % split-half iterations
k_grid = logspace(1, 19, 20); % ridge penalty grid, see Zada et al., 2025
K_folds = 5;                % cv folds for ridge k selection
rng_seed = 1;

% colors
eng_color = [37, 48, 122] / 255;
spa_color = [210, 34, 37] / 255;
shared_color = [150, 100, 200] / 255;

%% ============================================================
%  load neural data
%  ============================================================
FR_english_all = [];
FR_spanish_all = [];

for f = 1:length(data_files)
    data = load(data_files{f});
    FR_english_all = [FR_english_all, extract_firing_rates(data.matched_spike_data.english)];
    FR_spanish_all = [FR_spanish_all, extract_firing_rates(data.matched_spike_data.spanish)];
end

% transpose to words x neurons
neural_eng = FR_english_all';
neural_spa = FR_spanish_all';
[n_words, n_neurons] = size(neural_eng);

%% ============================================================
%  load mBERT embeddings
%  ============================================================
mbert_eng_data = load(fullfile(mbert_dir, 'english', mbert_file));
eng_emb = [mbert_eng_data.mbert_features.podcast_1{mbert_layer}(:, mbert_col_start:end);
           mbert_eng_data.mbert_features.podcast_2{mbert_layer}(:, mbert_col_start:end);
           mbert_eng_data.mbert_features.podcast_3{mbert_layer}(:, mbert_col_start:end)];

mbert_spa_data = load(fullfile(mbert_dir, 'spanish', mbert_file));
spa_emb = [mbert_spa_data.mbert_features.podcast_1{mbert_layer}(:, mbert_col_start:end);
           mbert_spa_data.mbert_features.podcast_2{mbert_layer}(:, mbert_col_start:end);
           mbert_spa_data.mbert_features.podcast_3{mbert_layer}(:, mbert_col_start:end)];

n_components = min(n_components, size(eng_emb, 2));

%% ============================================================
%  shared PCA across languages
%  ============================================================
X_all = [eng_emb; spa_emb];
[~, score_all, ~, ~, explained_all] = pca(X_all, 'NumComponents', n_components);

n_eng_words = size(eng_emb, 1);
embed_eng_pca = score_all(1:n_eng_words, :);
embed_spa_pca = score_all(n_eng_words+1:end, :);

%% ============================================================
%  language contribution per PC
%  ============================================================
ssq_eng = sum(embed_eng_pca.^2, 1);
ssq_spa = sum(embed_spa_pca.^2, 1);
frac_eng = ssq_eng ./ (ssq_eng + ssq_spa);
frac_spa = ssq_spa ./ (ssq_eng + ssq_spa);

% per-PC t-tests with FDR correction
pvals_pc = zeros(1, n_components);
for pc = 1:n_components
    [~, pvals_pc(pc)] = ttest2(embed_eng_pca(:, pc), embed_spa_pca(:, pc));
end
[h_fdr, ~] = fdr_bh(pvals_pc, 0.05);

% cross-language PC score correlations
r_pc = zeros(n_components, 1);
for pc = 1:n_components
    r_pc(pc) = corr(embed_eng_pca(:, pc), embed_spa_pca(:, pc));
end

%% ============================================================
%  cross-validate ridge k per neuron (shared across languages)
%  ============================================================
if isempty(gcp('nocreate')); parpool; end

best_k = zeros(n_neurons, 1);
parfor n = 1:n_neurons
    best_k(n) = select_k_cv_both(embed_eng_pca, neural_eng(:, n), ...
        embed_spa_pca, neural_spa(:, n), k_grid, K_folds);
end

%% ============================================================
%  fit final ridge models with cv-selected k
%  ============================================================
W_eng = zeros(n_neurons, n_components);
W_spa = zeros(n_neurons, n_components);

parfor n = 1:n_neurons
    [bE, ~] = ridge_fit(embed_eng_pca, neural_eng(:, n), best_k(n));
    [bS, ~] = ridge_fit(embed_spa_pca, neural_spa(:, n), best_k(n));
    W_eng(n, :) = bE(:)';
    W_spa(n, :) = bS(:)';
end

%% ============================================================
%  cross-language similarity (global and neuron-wise)
%  ============================================================
correlation_global = corr(W_eng(:), W_spa(:), 'rows', 'complete');

neuron_similarity = zeros(n_neurons, 1);
for n = 1:n_neurons
    neuron_similarity(n) = corr(W_eng(n, :)', W_spa(n, :)', 'rows', 'complete');
end

%% ============================================================
%  permutation tests (global + neuron-wise)
%  ============================================================
rng(rng_seed);
perm_mat = zeros(n_words, n_perms);
for p = 1:n_perms
    perm_mat(:, p) = randperm(n_words);
end

null_corrs_global = zeros(n_perms, 1);
null_corrs_neuron = zeros(n_perms, n_neurons);

parfor p = 1:n_perms
    y_spa_perm = neural_spa(perm_mat(:, p), :);
    W_spa_perm = zeros(n_neurons, n_components);
    for n = 1:n_neurons
        [bS_perm, ~] = ridge_fit(embed_spa_pca, y_spa_perm(:, n), best_k(n));
        W_spa_perm(n, :) = bS_perm(:)';
    end
    null_corrs_global(p) = corr(W_eng(:), W_spa_perm(:), 'rows', 'complete');
    cvec = zeros(n_neurons, 1);
    for n = 1:n_neurons
        c = corr(W_eng(n, :)', W_spa_perm(n, :)', 'rows', 'complete');
        if isnan(c); c = 0; end
        cvec(n) = c;
    end
    null_corrs_neuron(p, :) = cvec;
end

% two-tailed p-values
p_value_global = mean(abs(null_corrs_global) >= abs(correlation_global));
pvals_neuron = mean(abs(null_corrs_neuron) >= abs(neuron_similarity)', 1)';

%% ============================================================
%  binomial tests
%  ============================================================
valid_idx = ~isnan(neuron_similarity) & ~isnan(pvals_neuron);
valid_similarity = neuron_similarity(valid_idx);
valid_pvals = pvals_neuron(valid_idx);
n_valid = sum(valid_idx);

p_threshold = 0.05;
n_sig_positive = sum(valid_similarity > 0 & valid_pvals < p_threshold);
n_sig_negative = sum(valid_similarity < 0 & valid_pvals < p_threshold);
n_positive = sum(valid_similarity > 0);
n_negative = sum(valid_similarity < 0);

% binomial tests
p_binom = zeros(4, 1);
p_binom(1) = myBinomTest(n_sig_positive, n_valid, 0.025, 'right');
p_binom(2) = myBinomTest(n_sig_negative, n_valid, 0.025, 'right');
p_binom(3) = myBinomTest(n_positive, n_valid, 0.5, 'right');
p_binom(4) = myBinomTest(n_negative, n_valid, 0.5, 'right');

% store results
binomial_results = struct( ...
    'n_valid', n_valid, ...
    'n_positive', n_positive, ...
    'n_negative', n_negative, ...
    'n_sig_positive', n_sig_positive, ...
    'n_sig_negative', n_sig_negative, ...
    'p_threshold', p_threshold, ...
    'binomial_pvals', p_binom', ...
    'proportions', struct( ...
        'sig_positive', n_sig_positive/n_valid, ...
        'sig_negative', n_sig_negative/n_valid, ...
        'positive', n_positive/n_valid, ...
        'negative', n_negative/n_valid));

%% ============================================================
%  neuron-wise similarity bar plot
%  ============================================================
fig1 = figure('Position', [100, 100, 600, 400], 'Color', 'w');
bar(1:n_neurons, neuron_similarity, 'FaceColor', shared_color, 'EdgeColor', 'k'); hold on;
sig_idx = find(pvals_neuron < 0.05);
for i = 1:length(sig_idx)
    offset = 0.02 * sign(neuron_similarity(sig_idx(i)));
    text(sig_idx(i), neuron_similarity(sig_idx(i)) + offset, '*', ...
        'FontSize', 14, 'HorizontalAlignment', 'center', 'Color', 'r');
end
mean_sim = mean(neuron_similarity);
plot([0.5, n_neurons+0.5], [mean_sim, mean_sim], 'r--', 'LineWidth', 2);
xlabel('neuron #'); ylabel('pearson r');
xlim([0.5, n_neurons+0.5]);
ylim([min(neuron_similarity)-0.1, max(neuron_similarity)+0.15]);
axis square; grid off;
set(fig1, 'PaperPositionMode', 'auto');
print(fig1, 'figure_neuron_similarity.pdf', '-dpdf', '-bestfit');

%% ============================================================
%  split-half reliability vs cross-language tuning
%  ============================================================
rng(123);
n_half = floor(n_words / 2);

perm_eng_sh = zeros(n_words, n_splits, 'uint32');
perm_spa_sh = zeros(n_words, n_splits, 'uint32');
for s = 1:n_splits
    perm_eng_sh(:, s) = uint32(randperm(n_words));
    perm_spa_sh(:, s) = uint32(randperm(n_words));
end

within_corrs_eng = nan(n_neurons, n_splits, 'single');
within_corrs_spa = nan(n_neurons, n_splits, 'single');

parfor s = 1:n_splits
    idxE = double(perm_eng_sh(:, s));
    e1 = idxE(1:n_half); e2 = idxE(n_half+1:end);
    idxS = double(perm_spa_sh(:, s));
    s1 = idxS(1:n_half); s2 = idxS(n_half+1:end);

    XE1 = embed_eng_pca(e1, :); XE2 = embed_eng_pca(e2, :);
    XS1 = embed_spa_pca(s1, :); XS2 = embed_spa_pca(s2, :);

    ce = zeros(n_neurons, 1);
    cs = zeros(n_neurons, 1);
    for n = 1:n_neurons
        [bE1, ~] = ridge_fit(XE1, neural_eng(e1, n), best_k(n));
        [bE2, ~] = ridge_fit(XE2, neural_eng(e2, n), best_k(n));
        rE = corr(bE1, bE2, 'rows', 'complete'); if isnan(rE), rE = 0; end
        ce(n) = rE;

        [bS1, ~] = ridge_fit(XS1, neural_spa(s1, n), best_k(n));
        [bS2, ~] = ridge_fit(XS2, neural_spa(s2, n), best_k(n));
        rS = corr(bS1, bS2, 'rows', 'complete'); if isnan(rS), rS = 0; end
        cs(n) = rS;
    end
    within_corrs_eng(:, s) = single(ce);
    within_corrs_spa(:, s) = single(cs);
end

% global comparison
cross_vec = neuron_similarity(:);
global_within_means = zeros(n_splits, 1);
for s = 1:n_splits
    global_within_means(s) = mean([within_corrs_eng(:, s); within_corrs_spa(:, s)], 'omitnan');
end
global_cross_mean = mean(cross_vec, 'omitnan');
p_right_global = mean(global_within_means >= global_cross_mean);

% per-neuron within vs cross
per_neuron_within_med = mean(double([within_corrs_eng, within_corrs_spa]), 2, 'omitnan');
[p_wil_within, ~, stats_within] = signrank(per_neuron_within_med, cross_vec, 'tail', 'right');

%% ============================================================
%  split-half figures
%  ============================================================
% figure: global null distribution vs cross-language mean
fig2 = figure('Position', [120 120 300 300], 'Color', 'w');
histogram(global_within_means, 50, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'none'); hold on;
yl = ylim;
plot([global_cross_mean global_cross_mean], yl, 'r-', 'LineWidth', 3);
xlabel('mean split-half reliability');
ylabel('count');
legend({'within-language', 'cross-language mean'}, 'Location', 'best');
axis square; grid off; box off;
set(fig2, 'PaperPositionMode', 'auto');
print(fig2, 'figure_splithalf_vs_crosslang.pdf', '-dpdf', '-bestfit');

% figure: per-neuron within vs cross scatter
fig3 = figure('Position', [140 140 400 400], 'Color', 'w'); hold on;
scatter(per_neuron_within_med, cross_vec, 18, 'filled', 'MarkerFaceAlpha', 0.35);
mx = max([per_neuron_within_med; cross_vec]);
mn = min([per_neuron_within_med; cross_vec]);
plot([mn mx], [mn mx], 'k--', 'LineWidth', 1.5);
xlabel('within-language split-half reliability');
ylabel('cross-language tuning similarity');
axis square; grid on; box off;
set(fig3, 'PaperPositionMode', 'auto');
print(fig3, 'figure_within_vs_cross_scatter.pdf', '-dpdf', '-bestfit');

%% ============================================================
%  save results and clean up
%  ============================================================
save('results_tuningcurve_K.mat', ...
    'W_eng', 'W_spa', 'best_k', 'neuron_similarity', 'pvals_neuron', ...
    'correlation_global', 'p_value_global', 'binomial_results', ...
    'within_corrs_eng', 'within_corrs_spa', 'cross_vec', ...
    'per_neuron_within_med', 'p_wil_within');

delete(gcp('nocreate'));

%% ============================================================
%  helper functions
%  ============================================================
function fr = extract_firing_rates(language_struct)
% concatenate firing rates across all brain regions.
    regions = fieldnames(language_struct);
    fr = [];
    for r = 1:length(regions)
        fr = [fr; language_struct.(regions{r}).firing_rates.firing_rates];
    end
end

function [b, b0] = ridge_fit(X, y, k)
% ridge regression with unpenalized intercept via centering.
    muX = mean(X, 1);
    muy = mean(y, 1);
    Xc = bsxfun(@minus, X, muX);
    yc = y - muy;
    p = size(X, 2);
    b = (Xc' * Xc + k * eye(p)) \ (Xc' * yc);
    b0 = muy - muX * b;
end

function kbest = select_k_cv_both(Xe, ye, Xs, ys, k_grid, K)
% choose ridge k by minimizing average cv error across both languages.
    cvE = cvpartition(size(Xe, 1), 'KFold', K);
    cvS = cvpartition(size(Xs, 1), 'KFold', K);
    nK = numel(k_grid);
    loss = zeros(nK, 1);
    for ik = 1:nK
        k = k_grid(ik);
        nmseE = zeros(K, 1);
        for f = 1:K
            tr = training(cvE, f); te = test(cvE, f);
            [b, b0] = ridge_fit(Xe(tr, :), ye(tr), k);
            yhat = Xe(te, :) * b + b0;
            nmseE(f) = sum((ye(te) - yhat).^2) / sum((ye(te) - mean(ye(tr))).^2 + eps);
        end
        nmseS = zeros(K, 1);
        for f = 1:K
            tr = training(cvS, f); te = test(cvS, f);
            [b, b0] = ridge_fit(Xs(tr, :), ys(tr), k);
            yhat = Xs(te, :) * b + b0;
            nmseS(f) = sum((ys(te) - yhat).^2) / sum((ys(te) - mean(ys(tr))).^2 + eps);
        end
        loss(ik) = 0.5 * (mean(nmseE) + mean(nmseS));
    end
    [~, idx] = min(loss);
    kbest = k_grid(idx);
end

function p = myBinomTest(k, n, p0, tail)
% binomial test (right, left, or two-tailed).
    if nargin < 4, tail = 'right'; end
    switch lower(tail)
        case 'right'
            p = 1 - binocdf(k-1, n, p0);
        case 'left'
            p = binocdf(k, n, p0);
        case 'two'
            p = min(1, 2 * min(binocdf(k, n, p0), 1 - binocdf(k-1, n, p0)));
    end
end