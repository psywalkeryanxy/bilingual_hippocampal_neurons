%% === revised procrustes analyses (speed-optimized) ===
% analysis 5:  expanded anchor set size (3 to 100) -> find optimal k
%              null only at selected anchor sizes (1000 perms)
% analysis 4a: target-to-neighbor similarity vs prediction accuracy (no perm null)
% analysis 4b: neighbor-cluster tightness vs prediction accuracy (no perm null)
% analysis 1R: sliding window with optimal k (no null -- use analysis 5 null as reference)
% analysis 6:  anchor structure quality: redundancy vs dimensionality (no perm null)
% xinyuanyan


clear all; close all;

%% ============================================================
%  user configuration
%  ============================================================
data_files = {
    'matched_podcast1_spike_data.mat',
    'matched_podcast2_spike_data.mat',
    'matched_podcast3_spike_data.mat'
};

min_anchors = 3;
max_anchors = 100;
n_perms = 1000;
n_max_samples = 5000;
null_check_sizes = [3, 5, 7, 10, 15, 20, 30, 50, 75, 100];
rng_seed = 42;

% colors
purple_base = [127 63 152] / 255;
purple_light = purple_base + 0.6 * (1 - purple_base);
purple_dark = purple_base * 0.7;

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

FR_eng = FR_english_all;
FR_spa = FR_spanish_all;

%% ============================================================
%  l2 normalize
%  ============================================================
FR_eng = FR_eng ./ (vecnorm(FR_eng, 2, 1) + eps);
FR_spa = FR_spa ./ (vecnorm(FR_spa, 2, 1) + eps);

[n_neurons, n_words] = size(FR_eng);

if n_words < max_anchors + 1
    max_anchors = n_words - 1;
end
null_check_sizes = null_check_sizes(null_check_sizes <= max_anchors);

%% ============================================================
%  precompute pairwise distances and neighbors
%  ============================================================
eng_cosine_dist = pdist2(FR_eng', FR_eng', 'cosine');
eng_cosine_sim = 1 - eng_cosine_dist;

[sorted_dists, sorted_neighbors] = sort(eng_cosine_dist, 2, 'ascend');
sorted_neighbors = sorted_neighbors(:, 2:end);
sorted_sims = 1 - sorted_dists(:, 2:end);

rng(rng_seed);
n_samples = min(n_max_samples, n_words);
sample_idx = randsample(n_words, n_samples);

if isempty(gcp('nocreate')); parpool; end

%% ============================================================
%  analysis 5: expanded anchor set size (3 to max) -> optimal k
%  ============================================================
anchor_sizes = min_anchors:max_anchors;
n_sizes = length(anchor_sizes);

% observed for all sizes
size_results = zeros(n_sizes, 1);
size_results_std = zeros(n_sizes, 1);

for s = 1:n_sizes
    obs_cosines = procrustes_predict(FR_eng, FR_spa, sorted_neighbors, ...
        sample_idx, anchor_sizes(s), 1);
    size_results(s) = mean(obs_cosines);
    size_results_std(s) = std(obs_cosines) / sqrt(n_samples);
end

% null at selected sizes
null_at_check = struct('size', {}, 'null_mean', {}, 'null_std', {}, 'p_value', {});

for c = 1:length(null_check_sizes)
    nn_count = null_check_sizes(c);
    s_idx = find(anchor_sizes == nn_count);
    [nm, ns, ~, pv] = compute_null(FR_eng, FR_spa, ...
        sample_idx, nn_count, n_perms, size_results(s_idx));
    null_at_check(c).size = nn_count;
    null_at_check(c).null_mean = nm;
    null_at_check(c).null_std = ns;
    null_at_check(c).p_value = pv;
end

% interpolate null
null_sizes = [null_at_check.size];
null_means = [null_at_check.null_mean];
null_stds = [null_at_check.null_std];
null_mean_interp = interp1(null_sizes, null_means, anchor_sizes, 'pchip');
null_std_interp = interp1(null_sizes, null_stds, anchor_sizes, 'pchip');

size_effect = size_results - null_mean_interp';

% optimal k
[best_cosine, best_idx] = max(size_results);
optimal_k = anchor_sizes(best_idx);

% angles
angles_obs = acosd(size_results);
angles_null = acosd(null_mean_interp');

% fit exponential saturation model to effect size
x_data = anchor_sizes';
y_effect = size_effect;
exp_sat_model = @(p, x) p(1) * (1 - exp(-p(2) * (x - min_anchors))) + p(3);
try
    exp_sat_params = lsqcurvefit(exp_sat_model, [0.02, 0.1, y_effect(1)], ...
        x_data, y_effect, [0, 0, -1], [1, 10, 1], ...
        optimoptions('lsqcurvefit', 'Display', 'off'));
    saturation_95 = min_anchors - log(0.05) / exp_sat_params(2);
catch
    exp_sat_params = [NaN NaN NaN];
    saturation_95 = NaN;
end

%% ============================================================
%  analysis 4a: target-to-neighbor similarity vs accuracy
%  ============================================================
k_anchor = optimal_k;
target_to_neighbor_sim = zeros(n_samples, 1);
prediction_accuracy = zeros(n_samples, 1);

for i = 1:n_samples
    word_i = sample_idx(i);
    e_t = FR_eng(:, word_i);
    y_true = FR_spa(:, word_i);
    nn = sorted_neighbors(word_i, 1:k_anchor);
    target_to_neighbor_sim(i) = mean(sorted_sims(word_i, 1:k_anchor));
    Xe = FR_eng(:, nn); Xs = FR_spa(:, nn);
    muE = mean(Xe, 2); muS = mean(Xs, 2);
    [U, ~, V] = svd((Xs - muS) * (Xe - muE)', 'econ');
    Q = U * V';
    y_hat = muS + Q * (e_t - muE);
    y_hat = y_hat / (norm(y_hat) + eps);
    prediction_accuracy(i) = dot(y_hat, y_true);
end

[r_4a, p_4a] = corr(target_to_neighbor_sim, prediction_accuracy, 'type', 'Spearman');

%% ============================================================
%  analysis 4b: neighbor-cluster tightness vs accuracy
%  ============================================================
neighbor_tightness = zeros(n_samples, 1);

parfor i = 1:n_samples
    word_i = sample_idx(i);
    nn = sorted_neighbors(word_i, 1:k_anchor);
    anchor_sims = eng_cosine_sim(nn, nn);
    upper_tri = triu(anchor_sims, 1);
    neighbor_tightness(i) = mean(upper_tri(upper_tri ~= 0));
end

[r_4b, p_4b] = corr(neighbor_tightness, prediction_accuracy, 'type', 'Spearman');
[r_ab, p_ab] = corr(target_to_neighbor_sim, neighbor_tightness, 'type', 'Spearman');

%% ============================================================
%  analysis 1R: sliding window with optimal k
%  ============================================================
max_start = min(100, size(sorted_neighbors, 2) - k_anchor);
window_results = zeros(max_start, 1);
window_results_std = zeros(max_start, 1);

for start_nn = 1:max_start
    obs_cosines = procrustes_predict(FR_eng, FR_spa, sorted_neighbors, ...
        sample_idx, k_anchor, start_nn);
    window_results(start_nn) = mean(obs_cosines);
    window_results_std(start_nn) = std(obs_cosines) / sqrt(n_samples);
end

% reference null from analysis 5
null_ref_idx = find([null_at_check.size] == optimal_k);
if isempty(null_ref_idx)
    [~, null_ref_idx] = min(abs([null_at_check.size] - optimal_k));
end
ref_null_mean = null_at_check(null_ref_idx).null_mean;
ref_null_std = null_at_check(null_ref_idx).null_std;

%% ============================================================
%  analysis 6: anchor structure -- dimensionality
%  ============================================================
anchor_eff_dim = zeros(n_samples, 1);

for i = 1:n_samples
    word_i = sample_idx(i);
    nn = sorted_neighbors(word_i, 1:k_anchor);
    Xe = FR_eng(:, nn);
    Xe_centered = Xe - mean(Xe, 2);
    sv = svd(Xe_centered, 'econ');
    sv_sq = sv.^2;
    if sum(sv_sq) > 0
        sv_norm = sv_sq / sum(sv_sq);
        anchor_eff_dim(i) = (sum(sv_norm))^2 / sum(sv_norm.^2);
    end
end

[r_dim, p_dim] = corr(anchor_eff_dim, prediction_accuracy, 'type', 'Spearman');

%% ============================================================
%  figure: 6-panel summary
%  ============================================================
fig = figure('Position', [100 100 800 800], 'Color', 'w');

% panel 1: observed vs null (anchor size)
subplot(3, 2, 1); hold on;
fill([anchor_sizes, fliplr(anchor_sizes)], ...
    [null_mean_interp + 2*null_std_interp, ...
    fliplr(null_mean_interp - 2*null_std_interp)], ...
    [0.85 0.85 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
plot(anchor_sizes, null_mean_interp, 'k--', 'LineWidth', 1.5);
plot(anchor_sizes, size_results, '-', 'Color', purple_base, 'LineWidth', 2.5);
fill([anchor_sizes, fliplr(anchor_sizes)], ...
    [size_results' + size_results_std', ...
    fliplr(size_results' - size_results_std')], ...
    purple_light, 'EdgeColor', 'none', 'FaceAlpha', 0.3);
plot(null_sizes, null_means, 'ko', 'MarkerSize', 6, 'MarkerFaceColor', 'k');
plot(optimal_k, best_cosine, 'o', 'Color', purple_dark, 'MarkerSize', 10, ...
    'LineWidth', 2, 'MarkerFaceColor', purple_base);
text(optimal_k + 2, best_cosine, sprintf('k=%d', optimal_k), ...
    'FontSize', 10, 'Color', purple_dark, 'FontWeight', 'bold');
xlabel('number of anchors'); ylabel('mean cosine similarity');
title('observed vs null');
legend({'null +/-2SD', 'null mean', 'observed', 'obs +/-SEM', ...
    'null computed', 'optimal k'}, 'Location', 'southeast', 'FontSize', 8);
grid off; box off; axis square; xlim([min_anchors max_anchors]);

% panel 2: angular error
subplot(3, 2, 2); hold on;
plot(anchor_sizes, angles_obs, '-', 'Color', purple_base, 'LineWidth', 2.5);
plot(anchor_sizes, angles_null, 'k--', 'LineWidth', 1.5);
fill([anchor_sizes, fliplr(anchor_sizes)], ...
    [angles_obs', fliplr(angles_null')], ...
    purple_light, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
[min_angle, min_idx] = min(angles_obs);
plot(anchor_sizes(min_idx), min_angle, 'o', 'Color', purple_dark, ...
    'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', purple_base);
text(anchor_sizes(min_idx) + 3, min_angle + 0.5, ...
    sprintf('min: %.1f deg @ %d', min_angle, anchor_sizes(min_idx)), ...
    'FontSize', 10, 'Color', purple_dark);
xlabel('number of anchors'); ylabel('prediction angle (degrees)');
title('angular error');
legend({'observed', 'null', 'signal gap'}, 'Location', 'best');
grid off; box off; axis square;
xlim([min_anchors max_anchors]); set(gca, 'YDir', 'reverse');

% panel 4: cluster tightness vs accuracy
subplot(3, 2, 4); hold on;
scatter(neighbor_tightness, prediction_accuracy, 20, 'filled', ...
    'MarkerFaceColor', purple_base, 'MarkerFaceAlpha', 0.3);
pc = polyfit(neighbor_tightness, prediction_accuracy, 1);
xl = linspace(min(neighbor_tightness), max(neighbor_tightness), 100);
plot(xl, polyval(pc, xl), '-', 'Color', purple_dark, 'LineWidth', 2);
xlabel('neighbor-cluster tightness'); ylabel('prediction accuracy');
title(sprintf('4b: rho=%.3f, p=%.2g (k=%d)', r_4b, p_4b, optimal_k));
grid off; box off; axis square;

% panel 5: sliding window
subplot(3, 2, 5); hold on;
fill([1 max_start max_start 1], ...
    [ref_null_mean + 2*ref_null_std, ref_null_mean + 2*ref_null_std, ...
    ref_null_mean - 2*ref_null_std, ref_null_mean - 2*ref_null_std], ...
    [0.85 0.85 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
yline(ref_null_mean, ':', 'Color', [0.5 0.5 0.5], 'LineWidth', 0.5);
errorbar(1:max_start, window_results, window_results_std, ...
    'o-', 'Color', purple_base, 'LineWidth', 2, 'MarkerSize', 1, ...
    'MarkerFaceColor', purple_base);
xlabel('window start position (k-th nearest neighbor)');
ylabel('mean cosine similarity');
title(sprintf('sliding window (k=%d)', k_anchor));
legend({'null +/-2SD', 'observed +/- SEM'}, 'Location', 'best');
grid off; box off; axis square; xlim([0 max_start + 1]);

% panel 6: anchor effective dimensionality vs accuracy
subplot(3, 2, 6); hold on;
scatter(anchor_eff_dim, prediction_accuracy, 15, 'filled', ...
    'MarkerFaceColor', purple_base, 'MarkerFaceAlpha', 0.3);
pc = polyfit(anchor_eff_dim, prediction_accuracy, 1);
xl = linspace(min(anchor_eff_dim), max(anchor_eff_dim), 100);
plot(xl, polyval(pc, xl), '-', 'Color', purple_dark, 'LineWidth', 2);
xlabel('effective dimensionality'); ylabel('prediction accuracy');
title(sprintf('eff. dim: rho=%.3f, p=%.2g', r_dim, p_dim));
grid off; box off; axis square;

set(fig, 'PaperPositionMode', 'auto');
print(fig, 'figure_procrustes_analyses.pdf', '-dpdf', '-bestfit');

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

function cosines = procrustes_predict(FR_eng, FR_spa, sorted_neighbors, ...
        sample_idx, k_anchor, start_nn)
% procrustes-based cross-language prediction for sampled words.
    n_samp = length(sample_idx);
    cosines = zeros(n_samp, 1);
    for i = 1:n_samp
        word_i = sample_idx(i);
        e_t = FR_eng(:, word_i);
        y_true = FR_spa(:, word_i);
        nn = sorted_neighbors(word_i, start_nn:(start_nn + k_anchor - 1));
        Xe = FR_eng(:, nn); Xs = FR_spa(:, nn);
        muE = mean(Xe, 2); muS = mean(Xs, 2);
        [U, ~, V] = svd((Xs - muS) * (Xe - muE)', 'econ');
        Q = U * V';
        y_hat = muS + Q * (e_t - muE);
        y_hat = y_hat / (norm(y_hat) + eps);
        cosines(i) = dot(y_hat, y_true);
    end
end

function [null_mean, null_std, null_dist, p_value] = ...
        compute_null(FR_eng, FR_spa, sample_idx, k_anchor, n_perms, obs_mean)
% permutation null for procrustes prediction.
    n_words_local = size(FR_eng, 2);
    n_samp = length(sample_idx);
    null_dist = zeros(n_perms, 1);
    parfor perm = 1:n_perms
        perm_cosines = zeros(n_samp, 1);
        for i = 1:n_samp
            word_i = sample_idx(i);
            e_t = FR_eng(:, word_i);
            spa_target = randi(n_words_local);
            while spa_target == word_i
                spa_target = randi(n_words_local);
            end
            y_act = FR_spa(:, spa_target);
            candidates = setdiff(1:n_words_local, word_i);
            nn_perm = randsample(candidates, k_anchor);
            Xe = FR_eng(:, nn_perm); Xs = FR_spa(:, nn_perm);
            muE = mean(Xe, 2); muS = mean(Xs, 2);
            [U, ~, V] = svd((Xs - muS) * (Xe - muE)', 'econ');
            Q = U * V';
            y_hat = muS + Q * (e_t - muE);
            y_hat = y_hat / (norm(y_hat) + eps);
            perm_cosines(i) = dot(y_hat, y_act);
        end
        null_dist(perm) = mean(perm_cosines);
    end
    null_mean = mean(null_dist);
    null_std = std(null_dist);
    p_value = sum(null_dist >= obs_mean) / n_perms;
end