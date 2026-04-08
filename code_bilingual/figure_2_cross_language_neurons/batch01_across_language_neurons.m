%% word-pair analysis: spanish vs english neural responses
% per-neuron cross-language correlation with permutation testing,
% binomial tests, and visualization.
%
% note: this script uses study 1 (kurzgesagt) data as an example.
% the same analysis pipeline applies to other datasets by replacing
% the data loading section with appropriate file paths and structures.
%
% required input:
%   - matched spike data files (firing rates per language)
%
% output:
%   - per-neuron correlation distributions
%   - pie chart of significant proportions
%   - correlation histogram
%   - figures saved as PDF
% xinyuanyan
clear all; close all;

rng(123);

%% ============================================================
%  user configuration
%  ============================================================
data_files = {
    'matched_podcast1_spike_data.mat',
    'matched_podcast2_spike_data.mat',
    'matched_podcast3_spike_data.mat'
};

n_permutations = 1000;

% colors
positive_color = [127, 63, 152] / 255;
negative_color = [125, 125, 125] / 255;
nonsig_color = [255, 255, 255] / 255;

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

[n_neurons, n_words] = size(FR_english_all);

%% ============================================================
%  per-neuron correlation with permutation test
%  ============================================================
if isempty(gcp('nocreate')); parpool; end

neuron_correlations = zeros(n_neurons, 1);
neuron_p_values = zeros(n_neurons, 1);

for n = 1:n_neurons
    eng_resp = FR_english_all(n, :);
    spa_resp = FR_spanish_all(n, :);

    if var(eng_resp) == 0 || var(spa_resp) == 0
        neuron_correlations(n) = NaN;
        neuron_p_values(n) = NaN;
        continue;
    end

    actual_corr = corr(eng_resp', spa_resp');
    neuron_correlations(n) = actual_corr;

    null_corrs = zeros(n_permutations, 1);
    parfor perm = 1:n_permutations
        shuffled_spa = spa_resp(randperm(length(spa_resp)));
        null_corrs(perm) = corr(eng_resp', shuffled_spa');
    end

    neuron_p_values(n) = mean(abs(null_corrs) >= abs(actual_corr));
end

%% ============================================================
%  summarize results
%  ============================================================
valid_idx = ~isnan(neuron_correlations);
corrs_valid = neuron_correlations(valid_idx);
pvals_valid = neuron_p_values(valid_idx);
n_valid = sum(valid_idx);

sig_pos = pvals_valid < 0.05 & corrs_valid > 0;
sig_neg = pvals_valid < 0.05 & corrs_valid < 0;

n_sig_pos = sum(sig_pos);
n_sig_neg = sum(sig_neg);
n_nonsig = sum(~(sig_pos | sig_neg));
n_positive = sum(corrs_valid > 0);

% binomial tests
p_binom_pos = myBinomTest(n_positive, n_valid, 0.5, 'right');
p_binom_sig_pos = myBinomTest(n_sig_pos, n_valid, 0.025, 'right');
p_binom_sig_neg = myBinomTest(n_sig_neg, n_valid, 0.025, 'right');

results = struct( ...
    'n_neurons', n_neurons, ...
    'n_valid', n_valid, ...
    'correlations', corrs_valid, ...
    'p_values', pvals_valid, ...
    'mean_corr', mean(corrs_valid), ...
    'median_corr', median(corrs_valid), ...
    'std_corr', std(corrs_valid), ...
    'n_sig_positive', n_sig_pos, ...
    'n_sig_negative', n_sig_neg, ...
    'n_nonsig', n_nonsig, ...
    'n_positive', n_positive, ...
    'n_negative', sum(corrs_valid < 0), ...
    'p_binom', p_binom_pos, ...
    'p_binom_sig_pos', p_binom_sig_pos, ...
    'p_binom_sig_neg', p_binom_sig_neg, ...
    'sig_pos_neuron_idx', find(neuron_p_values < 0.05 & neuron_correlations > 0), ...
    'nan_neuron_idx', find(~valid_idx));

%% ============================================================
%  figure 1: pie chart of significant proportions
%  ============================================================
pct_pos = 100 * n_sig_pos / n_valid;
pct_neg = 100 * n_sig_neg / n_valid;
pct_ns = 100 * n_nonsig / n_valid;

fig1 = figure('Position', [100, 100, 500, 500], 'Color', 'w');
p = pie([pct_pos, pct_neg, pct_ns], ...
    {sprintf('sig. pos.\n%.1f%%', pct_pos), ...
     sprintf('sig. neg.\n%.1f%%', pct_neg), ...
     sprintf('non-sig.\n%.1f%%', pct_ns)});
p(1).FaceColor = positive_color;
p(3).FaceColor = negative_color;
p(5).FaceColor = nonsig_color;
title(sprintf('n=%d neurons', n_valid), 'FontWeight', 'bold');

set(fig1, 'PaperPositionMode', 'auto');
print(fig1, 'figure_significant_correlations_pie.pdf', '-dpdf', '-bestfit');

%% ============================================================
%  figure 2: correlation distribution
%  ============================================================
fig2 = figure('Position', [100, 100, 500, 500], 'Color', 'w');
histogram(corrs_valid, 30, 'FaceColor', [0.4 0.4 0.8], ...
    'EdgeColor', 'none', 'FaceAlpha', 0.7); hold on;
xline(0, 'k--', 'LineWidth', 2);

sig_pos_r = corrs_valid(sig_pos);
sig_neg_r = corrs_valid(sig_neg);
if ~isempty(sig_pos_r)
    plot(sig_pos_r, zeros(size(sig_pos_r)), 'r^', 'MarkerSize', 4, 'MarkerFaceColor', 'r');
end
if ~isempty(sig_neg_r)
    plot(sig_neg_r, zeros(size(sig_neg_r)), 'bv', 'MarkerSize', 4, 'MarkerFaceColor', 'b');
end

xlabel('correlation coefficient (r)');
ylabel('number of neurons');
title(sprintf('%d neurons, mean r=%.3f', n_valid, results.mean_corr), 'FontWeight', 'bold');
xlim([-0.25 0.25]);
grid off; axis square;

set(fig2, 'PaperPositionMode', 'auto');
print(fig2, 'figure_correlation_distribution.pdf', '-dpdf', '-bestfit');

%% ============================================================
%  save results
%  ============================================================
save('results_wordpair_correlations.mat', 'results');

%% ============================================================
%  helper functions
%  ============================================================
function fr = extract_firing_rates(language_struct)
% concatenate firing rates across all brain regions.
    regions = fieldnames(language_struct);
    fr = [];
    for r = 1:length(regions)
        region_data = language_struct.(regions{r});
        if isstruct(region_data) && isfield(region_data, 'firing_rates')
            if isstruct(region_data.firing_rates)
                fr = [fr; region_data.firing_rates.firing_rates];
            else
                fr = [fr; region_data.firing_rates];
            end
        end
    end
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