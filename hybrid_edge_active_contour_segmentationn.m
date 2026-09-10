%% Hybrid Edge + Active Contour Moss Segmentation
% Reconstructed from the uploaded project explanation PDF.
% Update testFolder before running.

clear; close all; clc;

%% Set paths
testFolder = 'f:\UNI\CV\Project\vegetation.v15i.png-mask-semantic\test';

imageFiles = dir(fullfile(testFolder, '*.jpg'));
maskFiles  = dir(fullfile(testFolder, '*_mask.png'));

%% Initialization
N = numel(imageFiles);

CR_all = zeros(N,1);
CM_all = zeros(N,1);
FM_all = zeros(N,1);

resultsFolder = fullfile(testFolder, 'results_hybrid_edge_activecontour');

if ~exist(resultsFolder, 'dir')
    mkdir(resultsFolder);
end

%% Loop over images
for i = 1:N

    %% Load RGB and ground-truth mask
    I_rgb = im2double(imread(fullfile(testFolder, imageFiles(i).name)));
    GT = imbinarize(imread(fullfile(testFolder, maskFiles(i).name)));

    %% Preprocess: green channel + contrast + denoising
    G = I_rgb(:,:,2);
    G_enh = adapthisteq(G);
    G_med = medfilt2(G_enh, [3 3]);

    %% Edge detection and initialization
    BW_edge = edge(G_med, 'canny', [0.15 0.4]);
    BW_closed = imclose(BW_edge, strel('disk', 3));
    mask_init = imfill(BW_closed, 'holes');
    mask_init = bwareafilt(mask_init, [100 Inf]);

    %% Active contour refinement (Chan-Vese)
    BW = activecontour(G_med, mask_init, 100, 'Chan-Vese');

    %% Post-processing
    BW = imfill(BW, 'holes');
    BW = bwareafilt(BW, [100 Inf]);

    %% Resize ground truth if needed
    if ~isequal(size(BW), size(GT))
        GT = imresize(GT, size(BW));
    end

    %% Evaluation
    [TP,FP,FN,CR,CM,FM] = evaluation_segmentation(BW, GT); %#ok<ASGLU>

    CR_all(i) = CR;
    CM_all(i) = CM;
    FM_all(i) = FM;

    %% Save predicted mask
    baseName = erase(imageFiles(i).name, '.jpg');
    imwrite(BW, fullfile(resultsFolder, [baseName, '_mask.png']));
end

%% Summary
fprintf('\n--- FINAL AVERAGE (Hybrid Edge + Active Contour) ---\n');
fprintf('CR: %.3f | CM: %.3f | FM: %.3f\n', ...
    mean(CR_all), mean(CM_all), mean(FM_all));

%% Best and Worst visualization
[~, bestIdx] = max(FM_all);
[~, worstIdx] = min(FM_all);

% Best case
I_best = im2double(imread(fullfile(testFolder, imageFiles(bestIdx).name)));
GT_best = imbinarize(imread(fullfile(testFolder, maskFiles(bestIdx).name)));

G_best = I_best(:,:,2);
G_best = adapthisteq(G_best);
G_best = medfilt2(G_best, [3 3]);

BW_edge = edge(G_best, 'canny', [0.15 0.4]);
BW_closed = imclose(BW_edge, strel('disk',3));
mask_init = imfill(BW_closed,'holes');
mask_init = bwareafilt(mask_init,[100 Inf]);

BW_best = activecontour(G_best, mask_init, 100, 'Chan-Vese');
BW_best = imfill(BW_best,'holes');
BW_best = bwareafilt(BW_best,[100 Inf]);

if ~isequal(size(BW_best), size(GT_best))
    GT_best = imresize(GT_best, size(BW_best));
end

overlay_best = imoverlay(I_best, BW_best, [0 1 0]);

figure('Name','Best Case');
subplot(1,3,1); imshow(I_best); title('Original');
subplot(1,3,2); imshow(BW_best); title('Predicted Mask');
subplot(1,3,3); imshow(overlay_best);
title(['Overlay | FM = ', num2str(FM_all(bestIdx),3)]);

saveas(gcf, fullfile(resultsFolder, 'best_case.png'));

% Worst case
I_worst = im2double(imread(fullfile(testFolder, imageFiles(worstIdx).name)));
GT_worst = imbinarize(imread(fullfile(testFolder, maskFiles(worstIdx).name)));

G_worst = I_worst(:,:,2);
G_worst = adapthisteq(G_worst);
G_worst = medfilt2(G_worst, [3 3]);

BW_edge = edge(G_worst, 'canny', [0.15 0.4]);
BW_closed = imclose(BW_edge, strel('disk',3));
mask_init = imfill(BW_closed,'holes');
mask_init = bwareafilt(mask_init,[100 Inf]);

BW_worst = activecontour(G_worst, mask_init, 100, 'Chan-Vese');
BW_worst = imfill(BW_worst,'holes');
BW_worst = bwareafilt(BW_worst,[100 Inf]);

if ~isequal(size(BW_worst), size(GT_worst))
    GT_worst = imresize(GT_worst, size(BW_worst));
end

overlay_worst = imoverlay(I_worst, BW_worst, [0 1 0]);

figure('Name','Worst Case');
subplot(1,3,1); imshow(I_worst); title('Original');
subplot(1,3,2); imshow(BW_worst); title('Predicted Mask');
subplot(1,3,3); imshow(overlay_worst);
title(['Overlay | FM = ', num2str(FM_all(worstIdx),3)]);

saveas(gcf, fullfile(resultsFolder, 'worst_case.png'));

%% Boxplot
figure('Name','FM, CR, CM - Boxplot');
boxplot([CR_all, CM_all, FM_all], 'Labels', {'CR','CM','FM'});
title('Performance Metrics (Test Set)');
saveas(gcf, fullfile(resultsFolder, 'boxplot_metrics.png'));

%% Save numeric results
T = table({imageFiles.name}', CR_all, CM_all, FM_all, ...
    'VariableNames', {'ImageName', 'CR', 'CM', 'FM'});

writetable(T, fullfile(resultsFolder, 'hybrid_test_metrics.xlsx'));

%% Evaluation Function
function [TP, FP, FN, CR, CM, FM] = evaluation_segmentation(BW, GT)
    TP = sum(BW(:)==1 & GT(:)==1);
    FP = sum(BW(:)==1 & GT(:)==0);
    FN = sum(BW(:)==0 & GT(:)==1);

    CR = TP / (TP + FP + eps);
    CM = TP / (TP + FN + eps);
    FM = 2 * CM * CR / (CM + CR + eps);
end
