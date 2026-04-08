%% cross-language semantic subspace alignment analysis
% compares english and spanish neural semantic subspaces using:
%   - cosine RDMs -> kernel-centered gram matrices
%   - eigendecomposition with PSD guards
%   - dimensionality (participation ratio + 95% variance explained)
%   - elsayed variance-capture alignment
%   - within-language split-half reliability (neuron splits)
%
% required input:
%   - matched spike data files (firing rates per language)
%
% output:
%   - subspace alignment metrics
%   - split-half reliability distributions
%   - figures saved as PDF

% xinyuanyan, got help from Michael Yoo

clear all; close all;

%% ============================================================
%  user configuration
%  ============================================================
data_files = {
    'matched_podcast1_spike_data.mat',
    'matched_podcast2_spike_data.mat',
    'matched_podcast3_spike_data.mat'
};

n_splits = 500;
n_pool = 32;
rng_seed = 123;

%% ============================================================
%  load and combine data
%  ============================================================
FR_english_all = [];
FR_spanish_all = [];

for f = 1:length(data_files)
    data = load(data_files{f});
    FR_english_all = [FR_english_all, extract_firing_rates(data.matched_spike_data.english)];
    FR_spanish_all = [FR_spanish_all, extract_firing_rates(data.matched_spike_data.spanish)];
end

% words x neurons
eng_neural = FR_english_all';
spa_neural = FR_spanish_all';
[n_words, n_neurons] = size(eng_neural);

%% ============================================================
%  cosine RDMs -> kernel-centered gram matrices
%  ============================================================
G_eng = centered_cosine_gram(eng_neural);
G_spa = centered_cosine_gram(spa_neural);

%% ============================================================
%  eigendecomposition and dimensionality
%  ============================================================
[V_e, evals_e] = eig_psd(G_eng);
[V_s, evals_s] = eig_psd(G_spa);

eff_dim_e = participation_ratio(evals_e);
eff_dim_s = participation_ratio(evals_s);

ve_e = evals_e / sum(evals_e);
ve_s = evals_s / sum(evals_s);
k95_e = find(cumsum(ve_e) >= 0.95, 1);
k95_s = find(cumsum(ve_s) >= 0.95, 1);

% common subspace dimensionality (capped)
k = min([k95_e, k95_s, 1000]);

Ue = V_e(:, 1:k);
Us = V_s(:, 1:k);
sum_topk_e = sum(evals_e(1:k));
sum_topk_s = sum(evals_s(1:k));

%% ============================================================
%  elsayed variance-capture alignment
%  ============================================================
align_eng_by_spa = elsayed_alignment_var(G_eng, Us, sum_topk_e);
align_spa_by_eng = elsayed_alignment_var(G_spa, Ue, sum_topk_s);
align_sym = 0.5 * (align_eng_by_spa + align_spa_by_eng);

%% ============================================================
%  figure: cumulative variance + effective dimensionality
%  ============================================================
fig1 = figure('Color', 'w', 'Position', [100 100 800 350]);
tiledlayout(1, 2, 'TileSpacing', 'compact');

nexttile;
plot(cumsum(ve_e)*100, '-', 'LineWidth', 1.5); hold on;
plot(cumsum(ve_s)*100, '-', 'LineWidth', 1.5);
yline(95, '--');
xlabel('components'); ylabel('cumulative variance (%)');
title(sprintf('k95: eng=%d, spa=%d (k=%d)', k95_e, k95_s, k));
legend({'english', 'spanish'}, 'Location', 'best'); grid on;

nexttile;
bar([eff_dim_e, eff_dim_s]);
set(gca, 'XTickLabel', {'english', 'spanish'});
ylabel('d_{eff}');
title('effective dimensionality'); grid on;

set(fig1, 'PaperPositionMode', 'auto');
print(fig1, 'figure_subspace_dimensionality.pdf', '-dpdf', '-bestfit');

%% ============================================================
%  within-language split-half reliability (neuron splits)
%  ============================================================
rng(rng_seed);

if isempty(gcp('nocreate')); parpool(n_pool); end

elsayed_eng_sh = zeros(n_splits, 1);
elsayed_spa_sh = zeros(n_splits, 1);

parfor b = 1:n_splits
    % english: split neurons
    jidx = randperm(n_neurons);
    j1 = jidx(1:floor(n_neurons/2));
    j2 = jidx(floor(n_neurons/2)+1:end);

    G_e1 = centered_cosine_gram(eng_neural(:, j1));
    G_e2 = centered_cosine_gram(eng_neural(:, j2));
    [Ve1, le1] = eig_psd(G_e1);
    [Ve2, le2] = eig_psd(G_e2);

    ke1 = find(cumsum(le1./sum(le1)) >= 0.95, 1);
    if isempty(ke1), ke1 = numel(le1); end
    ke2 = find(cumsum(le2./sum(le2)) >= 0.95, 1);
    if isempty(ke2), ke2 = numel(le2); end
    kbE = max(1, min([k, ke1, ke2, numel(le1), numel(le2)]));

    a1 = elsayed_alignment_var(G_e1, Ve2(:,1:kbE), sum(le1(1:kbE)));
    a2 = elsayed_alignment_var(G_e2, Ve1(:,1:kbE), sum(le2(1:kbE)));
    elsayed_eng_sh(b) = 0.5 * (a1 + a2);

    % spanish: split neurons
    jidx = randperm(n_neurons);
    j1 = jidx(1:floor(n_neurons/2));
    j2 = jidx(floor(n_neurons/2)+1:end);

    G_s1 = centered_cosine_gram(spa_neural(:, j1));
    G_s2 = centered_cosine_gram(spa_neural(:, j2));
    [Vs1, ls1] = eig_psd(G_s1);
    [Vs2, ls2] = eig_psd(G_s2);

    ks1 = find(cumsum(ls1./sum(ls1)) >= 0.95, 1);
    if isempty(ks1), ks1 = numel(ls1); end
    ks2 = find(cumsum(ls2./sum(ls2)) >= 0.95, 1);
    if isempty(ks2), ks2 = numel(ls2); end
    kbS = max(1, min([k, ks1, ks2, numel(ls1), numel(ls2)]));

    a1 = elsayed_alignment_var(G_s1, Vs2(:,1:kbS), sum(ls1(1:kbS)));
    a2 = elsayed_alignment_var(G_s2, Vs1(:,1:kbS), sum(ls2(1:kbS)));
    elsayed_spa_sh(b) = 0.5 * (a1 + a2);
end

% summary statistics
mean_e = mean(elsayed_eng_sh);  ci_e = quantile(elsayed_eng_sh, [.025 .975]);
mean_s = mean(elsayed_spa_sh);  ci_s = quantile(elsayed_spa_sh, [.025 .975]);

%% ============================================================
%  figure: split-half distributions
%  ============================================================
fig2 = figure('Color', 'w', 'Position', [100 100 1000 400]);
tiledlayout(1, 2, 'TileSpacing', 'compact');

nexttile;
histogram(elsayed_eng_sh, 40, 'Normalization', 'pdf'); hold on;
xline(mean_e, 'r', 'LineWidth', 2);
xline(ci_e(1), 'k--'); xline(ci_e(2), 'k--');
xlabel('elsayed split-half (eng)'); ylabel('pdf');
title(sprintf('eng mean=%.3f ci=[%.3f, %.3f]', mean_e, ci_e(1), ci_e(2)));
grid on;

nexttile;
histogram(elsayed_spa_sh, 40, 'Normalization', 'pdf'); hold on;
xline(mean_s, 'r', 'LineWidth', 2);
xline(ci_s(1), 'k--'); xline(ci_s(2), 'k--');
xlabel('elsayed split-half (spa)'); ylabel('pdf');
title(sprintf('spa mean=%.3f ci=[%.3f, %.3f]', mean_s, ci_s(1), ci_s(2)));
grid on;

set(fig2, 'PaperPositionMode', 'auto');
print(fig2, 'figure_splithalf_elsayed.pdf', '-dpdf', '-bestfit');

%% ============================================================
%  save results and clean up
%  ============================================================
save('results_subspace_alignment.mat', ...
    'eff_dim_e', 'eff_dim_s', 'k95_e', 'k95_s', 'k', ...
    'align_eng_by_spa', 'align_spa_by_eng', 'align_sym', ...
    'elsayed_eng_sh', 'elsayed_spa_sh', '-v7.3');

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

function G = centered_cosine_gram(X)
% compute kernel-centered cosine gram matrix from raw features.
    RDM = squareform(pdist(X, 'cosine'));
    K = 1 - RDM;
    K = (K + K') / 2;
    n = size(X, 1);
    J = eye(n) - ones(n) / n;
    G = J * K * J;
end

function [V_sorted, evals_sorted] = eig_psd(G)
% eigendecompose with PSD guard: symmetrize, sort descending, clip negatives.
    G = (G + G') / 2;
    [V, D] = eig(G);
    [evals_sorted, idx] = sort(real(diag(D)), 'descend');
    V_sorted = V(:, idx);
    tol = max(size(G, 1), 10) * eps(max(evals_sorted));
    keep = evals_sorted > tol;
    evals_sorted = evals_sorted(keep);
    V_sorted = V_sorted(:, keep);
end

function deff = participation_ratio(evals)
% participation ratio as effective dimensionality.
    s1 = sum(evals);
    s2 = sum(evals.^2);
    deff = (s1^2) / s2;
end

function a = elsayed_alignment_var(G_A, U_B, sum_topk_eigs_A)
% elsayed variance-capture alignment index (elsayed et al., 2016).
    a = trace(U_B' * G_A * U_B) / sum_topk_eigs_A;
end