%% Cross-Language Neural Pattern Similarity Analysis
% Analyzes the relationship between English and Spanish neural pattern
% similarities using cosine distance RDMs and permutation testing.
%
% Required input:
%   Data files containing matched spike data with fields:
%     - firing_rates: [neurons × words] matrix per language
%     - word labels: cell array of word strings per language
%
% Output:
%   - Paired t-tests per word pair across neurons
%   - Cross-language RDM correlation (Spearman) with permutation test
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

n_perms = 1000;
rng_seed = 42;

%% ============================================================
%  LOAD AND COMBINE DATA
%  ============================================================
FR_english_all = [];
FR_spanish_all = [];
eng_words_all = {};
spa_words_all = {};

for f = 1:length(data_files)
    data = load(data_files{f});

    fr_eng = extract_firing_rates(data.matched_spike_data.english);
    fr_spa = extract_firing_rates(data.matched_spike_data.spanish);

    FR_english_all = [FR_english_all, fr_eng];
    FR_spanish_all = [FR_spanish_all, fr_spa];

    eng_words_all = [eng_words_all; data.cleaned_words.textEng];
    spa_words_all = [spa_words_all; data.cleaned_words.textSpa];
end

eng_words_all = lower(eng_words_all);
spa_words_all = lower(spa_words_all);

[n_neurons, n_words] = size(FR_english_all);



%% ============================================================
%  CROSS-LANGUAGE RDM CORRELATION WITH PERMUTATION TEST
%  ============================================================
eng_rdm = pdist2(FR_english_all', FR_english_all', 'cosine');
spa_rdm = pdist2(FR_spanish_all', FR_spanish_all', 'cosine');

ut_mask = triu(true(n_words), 1);
eng_vec = eng_rdm(ut_mask);
spa_vec = spa_rdm(ut_mask);

obs_r = corr(eng_vec, spa_vec, 'type', 'Spearman');

% Permutation test: shuffle Spanish word order
rng(rng_seed);
perm_idx = zeros(n_words, n_perms, 'uint32');
for p = 1:n_perms
    perm_idx(:, p) = uint32(randperm(n_words));
end

if isempty(gcp('nocreate')); parpool; end

perm_r = zeros(n_perms, 1);
parfor p = 1:n_perms
    idx = perm_idx(:, p);
    spa_perm = spa_rdm(idx, idx);
    perm_r(p) = corr(eng_vec, spa_perm(ut_mask), 'type', 'Spearman');
end

p_perm = mean(abs(perm_r) >= abs(obs_r));

%% ============================================================
%  HELPER FUNCTION
%  ============================================================
function fr = extract_firing_rates(language_struct)
% Concatenate firing rates across brain regions.
    regions = fieldnames(language_struct);
    fr = [];
    for r = 1:length(regions)
        fr = [fr; language_struct.(regions{r}).firing_rates.firing_rates];
    end
end