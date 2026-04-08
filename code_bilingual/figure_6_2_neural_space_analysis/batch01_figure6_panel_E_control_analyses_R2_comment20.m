%% ================================================================
%  reviewer comment #20: participation correlation robustness
%  
%  concern: "the effect is being largely driven by a small population 
%  of neurons (at least for K and A1)"
%
%  analyses:
%  1) spearman rank correlation (robust to outliers)
%  2) remove top N% neurons and recompute correlation
%  3) remove bottom N% neurons and recompute correlation
% ================================================================
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

percentiles_to_remove = 5:5:30;

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
%  neuron-space covariance and participation strength
%  ============================================================
J = eye(n_words) - ones(n_words) / n_words;

C_eng = eng_neural' * J * eng_neural;
C_spa = spa_neural' * J * spa_neural;
C_eng = (C_eng + C_eng') / 2;
C_spa = (C_spa + C_spa') / 2;

[W_e, vals_e] = eig_psd(C_eng, n_neurons);
[W_s, vals_s] = eig_psd(C_spa, n_neurons);

ve_e = vals_e / sum(vals_e);
ve_s = vals_s / sum(vals_s);
k95_e = find(cumsum(ve_e) >= 0.95, 1);
k95_s = find(cumsum(ve_s) >= 0.95, 1);
kN = min([k95_e, k95_s, n_neurons]);

piE = sum(W_e(:, 1:kN).^2, 2);
piS = sum(W_s(:, 1:kN).^2, 2);

%% ============================================================
%  analysis 1: spearman vs pearson correlation
%  ============================================================
[r_pearson, p_pearson] = corr(piE, piS, 'Type', 'Pearson');
[r_spearman, p_spearman] = corr(piE, piS, 'Type', 'Spearman');

%% ============================================================
%  analysis 2: leave-out top N% neurons
%  ============================================================
pi_mean = (piE + piS) / 2;
n_pct = length(percentiles_to_remove);

results_top = struct();
results_top.percentiles = percentiles_to_remove;
results_top.r_spearman = zeros(n_pct, 1);
results_top.p_spearman = zeros(n_pct, 1);
results_top.n_remaining = zeros(n_pct, 1);

for i = 1:n_pct
    threshold = prctile(pi_mean, 100 - percentiles_to_remove(i));
    keep_idx = pi_mean < threshold;
    [results_top.r_spearman(i), results_top.p_spearman(i)] = ...
        corr(piE(keep_idx), piS(keep_idx), 'Type', 'Spearman');
    results_top.n_remaining(i) = sum(keep_idx);
end

%% ============================================================
%  analysis 3: leave-out bottom N% neurons
%  ============================================================
results_bottom = struct();
results_bottom.percentiles = percentiles_to_remove;
results_bottom.r_spearman = zeros(n_pct, 1);
results_bottom.p_spearman = zeros(n_pct, 1);
results_bottom.n_remaining = zeros(n_pct, 1);

for i = 1:n_pct
    threshold = prctile(pi_mean, percentiles_to_remove(i));
    keep_idx = pi_mean > threshold;
    [results_bottom.r_spearman(i), results_bottom.p_spearman(i)] = ...
        corr(piE(keep_idx), piS(keep_idx), 'Type', 'Spearman');
    results_bottom.n_remaining(i) = sum(keep_idx);
end

%% ============================================================
%  figure: leave-out robustness curve
%  ============================================================
purple_color = [127 63 152] / 255;
green_color = [0.2 0.6 0.4];

fig = figure('Color', 'w', 'Position', [100 100 200 200]);

plot([0, percentiles_to_remove], [r_spearman; results_top.r_spearman], ...
    'o-', 'Color', purple_color, 'LineWidth', 2, 'MarkerSize', 8, ...
    'MarkerFaceColor', purple_color); hold on;
plot([0, percentiles_to_remove], [r_spearman; results_bottom.r_spearman], ...
    's-', 'Color', green_color, 'LineWidth', 2, 'MarkerSize', 8, ...
    'MarkerFaceColor', green_color);
yline(0, 'k:', 'LineWidth', 1);

% significance markers (top removal)
add_sig_markers(percentiles_to_remove, results_top.r_spearman, ...
    results_top.p_spearman, purple_color, 0.03);

% significance markers (bottom removal)
add_sig_markers(percentiles_to_remove, results_bottom.r_spearman, ...
    results_bottom.p_spearman, green_color, -0.03);

xlabel('N% neurons removed');
ylabel('spearman \rho');
legend({'remove top N%', 'remove bottom N%'}, 'Location', 'southwest');
xlim([-2 32]);
axis square; grid off; box off;

set(fig, 'PaperPositionMode', 'auto');
print(fig, 'figure_participation_leaveout.pdf', '-dpdf', '-bestfit');

%%
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

function [V_sorted, evals_sorted] = eig_psd(G, n_dim)
% eigendecompose, sort descending, clip small/negative eigenvalues.
    G = (G + G') / 2;
    [V, D] = eig(G);
    [evals_sorted, idx] = sort(real(diag(D)), 'descend');
    V_sorted = V(:, idx);
    tol = max(n_dim, 10) * eps(max(evals_sorted));
    keep = evals_sorted > tol;
    evals_sorted = evals_sorted(keep);
    V_sorted = V_sorted(:, keep);
end

function add_sig_markers(x_vals, r_vals, p_vals, color, offset)
% add significance stars above/below data points.
    for i = 1:length(x_vals)
        if p_vals(i) < 0.001
            star = '***';
        elseif p_vals(i) < 0.01
            star = '**';
        elseif p_vals(i) < 0.05
            star = '*';
        else
            continue;
        end
        text(x_vals(i), r_vals(i) + offset, star, ...
            'HorizontalAlignment', 'center', 'FontSize', 9, 'Color', color);
    end
end