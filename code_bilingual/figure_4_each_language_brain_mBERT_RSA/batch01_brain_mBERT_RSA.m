%% RSA: Neural-Semantic Correspondence Analysis
% Compares neural and semantic (mBERT) representational dissimilarity
% matrices via Spearman correlation, with scatter plot visualization.
%
% Required input:
%   - Matched spike data files (firing rates per language)
%   - mBERT embedding files per language (layer specified below)
%
% Output:
%   - Two-panel scatter plot (Spanish, English) saved as PDF
% xinyuanyan
clear all; close all;

%% ============================================================
%  USER CONFIGURATION
%  ============================================================
data_files = {
    'matched_podcast1_spike_data.mat',
    'matched_podcast2_spike_data.mat',
    'matched_podcast3_spike_data.mat'
};

% mBERT embedding files (one per language)
mbert_dir = '2_mbert_whisper_kurzgesagt_words_matched';
mbert_file = 'mbert_features_extracted.mat';

% mBERT layer to use (1-indexed; layer 11 + 1 = 12th entry)
mbert_layer = 12;

% Columns to use from mBERT features (skip metadata columns)
mbert_col_start = 5;

% Visualization subsampling
n_samples = 1000;
rng_seed = 42;

%% ============================================================
%  LOAD NEURAL DATA
%  ============================================================
FR_english_all = [];
FR_spanish_all = [];

for f = 1:length(data_files)
    data = load(data_files{f});
    FR_english_all = [FR_english_all, extract_firing_rates(data.matched_spike_data.english)];
    FR_spanish_all = [FR_spanish_all, extract_firing_rates(data.matched_spike_data.spanish)];
end

% Transpose to words x neurons for pdist
neural_embedding_eng = FR_english_all';
neural_embedding_spa = FR_spanish_all';

%% ============================================================
%  LOAD mBERT EMBEDDINGS
%  ============================================================
mbert_spa = load(fullfile(mbert_dir, 'spanish', mbert_file));
embedding_spa = [mbert_spa.mbert_features.podcast_1{mbert_layer}(:, mbert_col_start:end);
                 mbert_spa.mbert_features.podcast_2{mbert_layer}(:, mbert_col_start:end);
                 mbert_spa.mbert_features.podcast_3{mbert_layer}(:, mbert_col_start:end)];

mbert_eng = load(fullfile(mbert_dir, 'english', mbert_file));
embedding_eng = [mbert_eng.mbert_features.podcast_1{mbert_layer}(:, mbert_col_start:end);
                 mbert_eng.mbert_features.podcast_2{mbert_layer}(:, mbert_col_start:end);
                 mbert_eng.mbert_features.podcast_3{mbert_layer}(:, mbert_col_start:end)];

%% ============================================================
%  COMPUTE RDMs AND RSA CORRELATIONS
%  ============================================================
neural_rdm_spa = pdist(neural_embedding_spa, 'cosine');
semantic_rdm_spa = pdist(embedding_spa, 'cosine');
rdm_corr_spa = corr(neural_rdm_spa', semantic_rdm_spa', 'Type', 'Spearman', 'Rows', 'complete');

neural_rdm_eng = pdist(neural_embedding_eng, 'cosine');
semantic_rdm_eng = pdist(embedding_eng, 'cosine');
rdm_corr_eng = corr(neural_rdm_eng', semantic_rdm_eng', 'Type', 'Spearman', 'Rows', 'complete');

%% ============================================================
%  FIGURE: RSA SCATTER PLOTS
%  ============================================================
eng_color = [37, 48, 122] / 255;
spa_color = [210, 34, 37] / 255;

fig = figure('Position', [50 50 800 800], 'Color', 'white');

% Subsample for visualization
rng(rng_seed);
n_pairs = length(neural_rdm_spa);
idx_spa = randperm(n_pairs, min(n_samples, n_pairs));
rng(rng_seed);
n_pairs_eng = length(neural_rdm_eng);
idx_eng = randperm(n_pairs_eng, min(n_samples, n_pairs_eng));

% --- Spanish ---
subplot(2, 1, 1); hold on;
scatter(neural_rdm_spa(idx_spa), semantic_rdm_spa(idx_spa), ...
    15, 'filled', 'MarkerFaceColor', spa_color, 'MarkerFaceAlpha', 0.05);
p_spa = polyfit(neural_rdm_spa, semantic_rdm_spa, 1);
x_range = [min(neural_rdm_spa), max(neural_rdm_spa)];
plot(x_range, polyval(p_spa, x_range), 'r-', 'LineWidth', 2);
xlabel('Neural Distance', 'FontSize', 20);
ylabel('Semantic Distance', 'FontSize', 20);
title(sprintf('Spanish: r = %.3f', rdm_corr_spa), 'FontSize', 20, 'FontWeight', 'bold');
set(gca, 'FontSize', 16); xlim([0 1]); ylim([0 1]); axis square; grid off;

% --- English ---
subplot(2, 1, 2); hold on;
scatter(neural_rdm_eng(idx_eng), semantic_rdm_eng(idx_eng), ...
    15, 'filled', 'MarkerFaceColor', eng_color, 'MarkerFaceAlpha', 0.05);
p_eng = polyfit(neural_rdm_eng, semantic_rdm_eng, 1);
x_range = [min(neural_rdm_eng), max(neural_rdm_eng)];
plot(x_range, polyval(p_eng, x_range), 'r-', 'LineWidth', 2);
xlabel('Neural Distance', 'FontSize', 20);
ylabel('Semantic Distance', 'FontSize', 20);
title(sprintf('English: r = %.3f', rdm_corr_eng), 'FontSize', 20, 'FontWeight', 'bold');
set(gca, 'FontSize', 16); xlim([0 1]); ylim([0 1]); axis square; grid off;

sgtitle('RSA: Neural-Semantic Correspondence', 'FontSize', 24, 'FontWeight', 'bold');

%% ============================================================
%  SAVE AS PDF
%  ============================================================
set(fig, 'PaperPositionMode', 'auto');
print(fig, 'figure_rsa_scatter.pdf', '-dpdf', '-bestfit');

%% ============================================================
%  HELPER FUNCTION
%  ============================================================
function fr = extract_firing_rates(language_struct)
    regions = fieldnames(language_struct);
    fr = [];
    for r = 1:length(regions)
        fr = [fr; language_struct.(regions{r}).firing_rates.firing_rates];
    end
end