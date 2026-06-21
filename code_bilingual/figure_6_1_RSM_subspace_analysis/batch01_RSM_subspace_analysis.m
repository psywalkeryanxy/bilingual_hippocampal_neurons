%% ================================================================
%  English–Spanish semantic subspace comparison (cosine RDMs)
% ----------------------------------------------------------------
%  - Gram from cosine RDM, kernel-centering
%  - Eigendecomposition with PSD guards
%  - Dimensionality (participation ratio + 95% variance explained)
%  - Elsayed variance-capture alignment (2016, NC paper)
%  - Permutation test (shuffle cross-language word matching)
%  - Within-language split-half reliability (neuron splits; noise ceiling)
%
%  INPUT FORMAT (user supplies):
%     eng_neural : [n_words x n_neurons]  mean firing rates, English
%     spa_neural : [n_words x n_neurons]  mean firing rates, Spanish
%  Rows must be matched across languages (row i of eng_neural and
%  row i of spa_neural refer to the same concept / matched word pair).
% ================================================================

clear all; close all;

%% --------------------------
%  Load data
%  --------------------------
%  Replace the block below with your own loading code. The rest of the
%  script expects two matrices:
%     eng_neural : n_words x n_neurons
%     spa_neural : n_words x n_neurons
%  where row indices are matched across languages.

%% 1) Cosine RDMs -> cosine Gram -> kernel centering
RDM_eng = squareform(pdist(eng_neural, 'cosine'));
RDM_spa = squareform(pdist(spa_neural, 'cosine'));

K_eng = (1 - RDM_eng);  K_eng = (K_eng + K_eng')/2;
K_spa = (1 - RDM_spa);  K_spa = (K_spa + K_spa')/2;

J     = eye(n_words) - ones(n_words)/n_words;
G_eng = J*K_eng*J;   % centered Gram, because these two subspaces both may include the population's baseline firing direction
G_spa = J*K_spa*J;

%% 2) Eigendecompose with PSD guards
[V_e, evals_e] = eig_psd(G_eng);
[V_s, evals_s] = eig_psd(G_spa);

% Dimensionality
eff_dim_e = participation_ratio(evals_e);
eff_dim_s = participation_ratio(evals_s);
ve_e = evals_e/sum(evals_e);  ve_s = evals_s/sum(evals_s);
k95_e = find(cumsum(ve_e)>=0.95, 1);
k95_s = find(cumsum(ve_s)>=0.95, 1);


% Choose k (common k). Hard cap at 1000 for safety; adjust if needed.
k = min([k95_e, k95_s, 1000]);

Ue = V_e(:, 1:k);
Us = V_s(:, 1:k);
sum_topk_eigs_e = sum(evals_e(1:k));
sum_topk_eigs_s = sum(evals_s(1:k));

%% 3) Elsayed variance-capture alignment
% Directional Elsayed indices
align_eng_by_spa = elsayed_alignment_var(G_eng, Us, sum_topk_eigs_e); % ENG <- SPA
align_spa_by_eng = elsayed_alignment_var(G_spa, Ue, sum_topk_eigs_s); % SPA <- ENG
align_sym = 0.5*(align_eng_by_spa + align_spa_by_eng);


%% 4) Permutation test on symmetric Elsayed index (shuffle word pairing)
Nperm = 1000;
elsayed_null = zeros(Nperm, 1);
rng(42);

delete(gcp('nocreate'));
parpool(32);   % adjust it plz

parfor p = 1:Nperm
    perm_idx   = randperm(n_words);
    G_spa_perm = G_spa(perm_idx, perm_idx);          % permute word order

    % Eig + k for permuted Spanish
    [V_s_p, evals_s_p] = eig_psd(G_spa_perm);
    kp = min([k, size(Ue,2), numel(evals_s_p)]);     % consistent k both directions
    Up = V_s_p(:, 1:kp);
    sum_topk_eigs_s_p = sum(evals_s_p(1:kp));

    % Directional indices under permutation
    a_e_by_s = elsayed_alignment_var(G_eng,      Up,           sum_topk_eigs_e);
    a_s_by_e = elsayed_alignment_var(G_spa_perm, Ue(:, 1:kp),  sum_topk_eigs_s_p);
    elsayed_null(p) = 0.5*(a_e_by_s + a_s_by_e);
end

p_perm = (sum(elsayed_null >= align_sym) + 1) / (Nperm + 1);

%% 5) Plots
figure('Color','w','Position',[100 100 1400 400]);
tiledlayout(1, 4, 'TileSpacing','compact');

% Spectra
nexttile;
semilogy(evals_e, '-o', 'LineWidth', 1.5); hold on;
semilogy(evals_s, '-s', 'LineWidth', 1.5);
xlabel('Component'); ylabel('Eigenvalue');
title('eigenvalue Spectrum'); legend({'English','Spanish'}); grid on;

% Cumulative variance
nexttile;
plot(cumsum(ve_e)*100, '-', 'LineWidth', 1.5); hold on;
plot(cumsum(ve_s)*100, '-', 'LineWidth', 1.5);
yline(95, '--'); xlabel('Components'); ylabel('Cum. Var (%)');
title(sprintf('k95: E=%d, S=%d (k=%d)', k95_e, k95_s, k)); grid on;

% Participation ratio
nexttile;
bar([eff_dim_e, eff_dim_s]);
set(gca, 'XTickLabel', {'English','Spanish'}); ylabel('d_{eff}');
title('effective Dimensionality'); grid on;

% Elsayed + null
nexttile;
histogram(elsayed_null, 40, 'Normalization', 'pdf'); hold on;
xline(align_sym, 'r', 'LineWidth', 2);
xlabel('Elsayed (mean of directions)'); ylabel('PDF');
title(sprintf('Elsayed mean = %.3f (p = %.4f)', align_sym, p_perm)); grid on;



%% 6) Within-language split-half reliability (Elsayed alignment; noise ceiling)
%     Split NEURONS (columns), not words. Because if the two half-splits datasets have different
% words, the subspace-misalignment may due to the different words
% so plz Keeps the same stimulus set in
%     both halves and isolates sampling noise in the neural readout.

Nsplit = 1000;
rng(123);

elsayed_eng_splithalf = zeros(Nsplit, 1);
elsayed_spa_splithalf = zeros(Nsplit, 1);

delete(gcp('nocreate'));
parpool(32);   % adjust

parfor b = 1:Nsplit
    %% ---------- English split: split NEURONS ----------
    jidx = randperm(size(eng_neural, 2));
    j1   = jidx(1:floor(numel(jidx)/2));
    j2   = jidx(floor(numel(jidx)/2)+1 : end);

    G_e1 = centered_cosine_gram_from_raw(eng_neural(:, j1));
    G_e2 = centered_cosine_gram_from_raw(eng_neural(:, j2));

    [Ve1, le1] = eig_psd(G_e1);
    [Ve2, le2] = eig_psd(G_e2);

    % Energy-matched k per split, capped by global k
    ke1 = find(cumsum(le1./sum(le1))>=0.95, 1); if isempty(ke1), ke1 = numel(le1); end
    ke2 = find(cumsum(le2./sum(le2))>=0.95, 1); if isempty(ke2), ke2 = numel(le2); end
    kbE = max(1, min([k, ke1, ke2, numel(le1), numel(le2)]));

    Ue1 = Ve1(:, 1:kbE); Ue2 = Ve2(:, 1:kbE);
    se1 = sum(le1(1:kbE)); se2 = sum(le2(1:kbE));

    a_e_1by2 = elsayed_alignment_var(G_e1, Ue2, se1); % E1 <- E2
    a_e_2by1 = elsayed_alignment_var(G_e2, Ue1, se2); % E2 <- E1
    elsayed_eng_splithalf(b) = 0.5*(a_e_1by2 + a_e_2by1);

    %% ---------- Spanish split: split NEURONS ----------
    jidx = randperm(size(spa_neural, 2));
    j1   = jidx(1:floor(numel(jidx)/2));
    j2   = jidx(floor(numel(jidx)/2)+1 : end);

    G_s1 = centered_cosine_gram_from_raw(spa_neural(:, j1));
    G_s2 = centered_cosine_gram_from_raw(spa_neural(:, j2));

    [Vs1, ls1] = eig_psd(G_s1);
    [Vs2, ls2] = eig_psd(G_s2);

    ks1 = find(cumsum(ls1./sum(ls1))>=0.95, 1); if isempty(ks1), ks1 = numel(ls1); end
    ks2 = find(cumsum(ls2./sum(ls2))>=0.95, 1); if isempty(ks2), ks2 = numel(ls2); end
    kbS = max(1, min([k, ks1, ks2, numel(ls1), numel(ls2)]));

    Us1 = Vs1(:, 1:kbS); Us2 = Vs2(:, 1:kbS);
    ss1 = sum(ls1(1:kbS)); ss2 = sum(ls2(1:kbS));

    a_s_1by2 = elsayed_alignment_var(G_s1, Us2, ss1); % S1 <- S2
    a_s_2by1 = elsayed_alignment_var(G_s2, Us1, ss2); % S2 <- S1
    elsayed_spa_splithalf(b) = 0.5*(a_s_1by2 + a_s_2by1);
end

% Summary stats
mean_e = mean(elsayed_eng_splithalf); ci_e = quantile(elsayed_eng_splithalf, [.025 .975]);
mean_s = mean(elsayed_spa_splithalf); ci_s = quantile(elsayed_spa_splithalf, [.025 .975]);


% Noise-corrected cross-language alignment
noise_corrected = align_sym / sqrt(mean_e * mean_s);

% Plot histograms
figure('Color','w','Position',[100 100 1000 400]);
tiledlayout(1, 2, 'TileSpacing','compact');

nexttile;
histogram(elsayed_eng_splithalf, 40, 'Normalization','pdf'); hold on;
xline(mean_e, 'r', 'LineWidth', 2);
xline(ci_e(1), 'k--'); xline(ci_e(2), 'k--');
xlabel('Elsayed split-half (ENG)'); ylabel('PDF'); grid on;
title(sprintf('ENG mean = %.3f  CI = [%.3f, %.3f]', mean_e, ci_e(1), ci_e(2)));

nexttile;
histogram(elsayed_spa_splithalf, 40, 'Normalization','pdf'); hold on;
xline(mean_s, 'r', 'LineWidth', 2);
xline(ci_s(1), 'k--'); xline(ci_s(2), 'k--');
xlabel('Elsayed split-half (SPA)'); ylabel('PDF'); grid on;
title(sprintf('SPA mean = %.3f  CI = [%.3f, %.3f]', mean_s, ci_s(1), ci_s(2)));

% Save them all if you would like to


%% =================================================================
%  Helper functions
%  =================================================================

function G = centered_cosine_gram_from_raw(X)
    % Centered cosine Gram from raw features.
    %   X: n_words x n_neurons_subset
    %   Returns G: n_words x n_words, double-centered.
    RDM = squareform(pdist(X, 'cosine'));
    K   = 1 - RDM;  K = (K + K')/2;
    n   = size(X, 1);
    J   = eye(n) - ones(n)/n;
    G   = J*K*J;
end

function [V_sorted, evals_sorted] = eig_psd(G)
    % Symmetric eigendecomposition with PSD guard: drop near-zero
    % eigenvalues (numerical noise) and sort descending.
    G = (G + G')/2;% it forces exact symmetry right before eig
    [V, D] = eig(G);
    [evals_sorted, idx] = sort(real(diag(D)), 'descend');
    V_sorted = V(:, idx);
    tol      = max(size(G,1), 10) * eps(max(evals_sorted));
    keep     = evals_sorted > tol;
    evals_sorted = evals_sorted(keep);
    V_sorted     = V_sorted(:, keep);
end

function deff = participation_ratio(evals)
    % Effective dimensionality.
    s1 = sum(evals); s2 = sum(evals.^2);
    deff = (s1^2) / s2;
end

function a = elsayed_alignment_var(G_A, U_B, sum_topk_eigs_A)
    % Elsayed variance-capture alignment:
    %   Align(A <- B) = tr(U_B' * G_A * U_B) / sum_{i=1..k} lambda_i(A)
    %
    %   G_A             : centered Gram (n x n) for target A
    %   U_B             : basis (n x k) for source B (top-k eigenvectors)
    %   sum_topk_eigs_A : sum of A's top-k eigenvalues (scalar)
    a = trace(U_B' * G_A * U_B) / sum_topk_eigs_A;
end
