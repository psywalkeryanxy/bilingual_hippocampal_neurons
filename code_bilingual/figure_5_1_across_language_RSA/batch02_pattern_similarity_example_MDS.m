%% Cross-Language MDS Visualization
% MDS-based visualization of cross-language neural semantic geometry.
%
% Required input:
%   - Data files with matched spike data (firing rates per language)
%   - matched_word_table.mat with fields .eng and .spa
%
% Output:
%   - Two-panel MDS plot (English, Spanish) saved as PDF

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

word_table_file = 'matched_word_table.mat';
selected_indices = [311:318];

%% ============================================================
%  LOAD AND COMBINE DATA
%  ============================================================
FR_english_all = [];
FR_spanish_all = [];

for f = 1:length(data_files)
    data = load(data_files{f});
    FR_english_all = [FR_english_all, extract_firing_rates(data.matched_spike_data.english)];
    FR_spanish_all = [FR_spanish_all, extract_firing_rates(data.matched_spike_data.spanish)];
end

wt = load(word_table_file);
word_eng = wt.matched_word_table.eng;
word_spa = wt.matched_word_table.spa;

%% ============================================================
%  SELECT WORDS, COMPUTE RDMs, AND RUN MDS
%  ============================================================
n_words = length(selected_indices);
selected_words_eng = word_eng(selected_indices);
selected_words_spa = word_spa(selected_indices);

rdm_eng = 1 - corr(FR_english_all(:, selected_indices));
rdm_spa = 1 - corr(FR_spanish_all(:, selected_indices));
rdm_eng(logical(eye(n_words))) = 0;
rdm_spa(logical(eye(n_words))) = 0;

[mds_coords_eng, stress_eng] = cmdscale(rdm_eng, 2);
[mds_coords_spa, stress_spa] = cmdscale(rdm_spa, 2);

%% ============================================================
%  VISUALIZATION PARAMETERS
%  ============================================================
if n_words <= 10
    node_size = 300; font_size_label = 11; label_display = 'all';
elseif n_words <= 20
    node_size = 200; font_size_label = 9; label_display = 'all';
elseif n_words <= 50
    node_size = 150; font_size_label = 8; label_display = 'selected';
else
    node_size = 100; font_size_label = 7; label_display = 'none';
end

eng_color = [37, 48, 122] / 255;
spa_color = [210, 34, 37] / 255;

%% ============================================================
%  FIGURE: MDS
%  ============================================================
fig = figure('Position', [50, 50, 1200, 500], 'Color', 'white');

% --- English ---
subplot(1, 2, 1); hold on;
scatter(mds_coords_eng(:,1), mds_coords_eng(:,2), node_size, eng_color, ...
    'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 2);
add_labels(mds_coords_eng, selected_words_eng, label_display, font_size_label, eng_color);
xlabel('MDS Dimension 1'); ylabel('MDS Dimension 2');
title(sprintf('English (%d words)', n_words), 'Color', eng_color);
axis square; grid off;

% --- Spanish ---
subplot(1, 2, 2); hold on;
scatter(mds_coords_spa(:,1), mds_coords_spa(:,2), node_size, spa_color, ...
    'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 2);
add_labels(mds_coords_spa, selected_words_spa, label_display, font_size_label, spa_color);
xlabel('MDS Dimension 1'); ylabel('MDS Dimension 2');
title(sprintf('Spanish (%d words)', n_words), 'Color', spa_color);
axis square; grid off;

%% ============================================================
%  SAVE AS PDF plz
%  ============================================================
set(fig, 'PaperPositionMode', 'auto');
print(fig, sprintf('mds_cross_language_%dwords.pdf', n_words), '-dpdf', '-bestfit');

%% ============================================================
%  HELPER FUNCTIONS
%  ============================================================
function fr = extract_firing_rates(language_struct)
    regions = fieldnames(language_struct);
    fr = [];
    for r = 1:length(regions)
        fr = [fr; language_struct.(regions{r}).firing_rates.firing_rates];
    end
end

function add_labels(coords, labels, mode, fsize, color)
    n = size(coords, 1);
    if strcmp(mode, 'all')
        idx = 1:n;
    elseif strcmp(mode, 'selected')
        [~, idx] = sort(vecnorm(coords, 2, 2), 'descend');
        idx = idx(1:min(10, n));
    else
        return;
    end
    for i = idx
        text(coords(i,1), coords(i,2), labels{i}, ...
            'FontSize', fsize, 'FontWeight', 'bold', ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
            'Color', color);
    end
end