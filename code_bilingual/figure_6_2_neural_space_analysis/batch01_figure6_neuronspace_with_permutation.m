%% neuron-space subspace comparison with permutation tests
% compares english and spanish semantic subspaces in neuron space:
%   - neuron x neuron covariances (centered across words)
%   - eigendecomposition for neuron-space axes
%   - neuron participation strength correlation
%   - permutation tests (neuron identity shuffle)


% xinyuanyan

clear all; close all;
purple_color = [0.45 0.20 0.70];
%% ============================================================
%  user configuration
%  ============================================================
data_files = {
    'matched_podcast1_spike_data.mat',
    'matched_podcast2_spike_data.mat',
    'matched_podcast3_spike_data.mat'
};

n_perm = 1000;
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
%  neuron-space covariance (centered over words)
%  ============================================================
J = eye(n_words) - ones(n_words) / n_words;

C_eng = eng_neural' * J * eng_neural;
C_spa = spa_neural' * J * spa_neural;
C_eng = (C_eng + C_eng') / 2;
C_spa = (C_spa + C_spa') / 2;

%% ============================================================
%  eigendecomposition in neuron space
%  ============================================================
[W_e, vals_e] = eig_psd(C_eng, n_neurons);
[W_s, vals_s] = eig_psd(C_spa, n_neurons);

ve_e = vals_e / sum(vals_e);
ve_s = vals_s / sum(vals_s);
k95_e = find(cumsum(ve_e) >= 0.95, 1);
k95_s = find(cumsum(ve_s) >= 0.95, 1);
kN = min([k95_e, k95_s, n_neurons]);

deff_e = (sum(vals_e)^2) / sum(vals_e.^2);
deff_s = (sum(vals_s)^2) / sum(vals_s.^2);

Ue_neuron = W_e(:, 1:kN);
Us_neuron = W_s(:, 1:kN);
sum_topk_e = sum(vals_e(1:kN));
sum_topk_s = sum(vals_s(1:kN));


% neuron participation strength
piE = sum(W_e(:, 1:kN).^2, 2);
piS = sum(W_s(:, 1:kN).^2, 2);
[obs_r_participation, obs_p_participation] = corr(piE, piS);

%% ============================================================
%  permutation test: shuffle neuron identities
%  ============================================================
rng(rng_seed);
neuron_perm_idx = zeros(n_neurons, n_perm);
for p = 1:n_perm
    neuron_perm_idx(:, p) = randperm(n_neurons);
end

if isempty(gcp('nocreate')); parpool(n_pool); end

perm_r_participation = zeros(n_perm, 1);

parfor p = 1:n_perm
    idx = neuron_perm_idx(:, p);
    spa_perm = spa_neural(:, idx);

    C_spa_perm = spa_perm' * J * spa_perm;
    C_spa_perm = (C_spa_perm + C_spa_perm') / 2;

    [W_perm, vals_perm] = eig_psd(C_spa_perm, n_neurons);

    if length(vals_perm) >= kN
        Us_perm = W_perm(:, 1:kN);
        sum_topk_perm = sum(vals_perm(1:kN));

        piS_perm = piS(idx);
        perm_r_participation(p) = corr(piE, piS_perm);
    else

        perm_r_participation(p) = NaN;
    end
end



p_participation = mean(perm_r_participation >= obs_r_participation);


%% ============================================================
%  neuron participation scatter
%  ============================================================
fig2 = figure('Color', 'w', 'Position', [200 200 400 400]);
scatter(piE, piS, 106, purple_color, 'filled', ...
    'MarkerFaceAlpha', 0.3, 'MarkerEdgeColor', 'none'); hold on;
plot_range = [min([piE; piS]), max([piE; piS])];
plot(plot_range, plot_range, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);
xlabel('neuron participation (english)');
ylabel('neuron participation (spanish)');
title(sprintf('r = %.3f, p = %.4f', obs_r_participation, p_participation));
axis square; grid on; box off;
xlim([min(piE)*0.95, max(piE)*1.05]);
ylim([min(piS)*0.95, max(piS)*1.05]);

set(fig2, 'PaperPositionMode', 'auto');
print(fig2, 'figure_neuron_participation.pdf', '-dpdf', '-bestfit');


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

function [V_sorted, evals_sorted] = eig_psd(G, n_dim)
% eigendecompose, sort descending, clip small/negative eigenvalues.

    [V, D] = eig(G);
    [evals_sorted, idx] = sort(real(diag(D)), 'descend');
    V_sorted = V(:, idx);
    tol = max(n_dim, 10) * eps(max(evals_sorted));
    keep = evals_sorted > tol;
    evals_sorted = evals_sorted(keep);
    V_sorted = V_sorted(:, keep);



end

