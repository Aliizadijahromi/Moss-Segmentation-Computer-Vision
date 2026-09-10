%% DeepLabv3+ (ResNet-50) Moss/Vegetation Semantic Segmentation
% Reconstructed from the uploaded project explanation PDF.
% Update trainPath and testPath before running.

clear; close all; clc;

%% Parameters
inputSize = [256 256];
trainPath = 'f:\UNI\CV\Project\vegetation.v15i.png-mask-semantic\train';
testPath  = 'f:\UNI\CV\Project\vegetation.v15i.png-mask-semantic\test';

%% Datastores & Combine
imds = imageDatastore(trainPath, ...
    'FileExtensions', {'.jpg'}, ...
    'ReadFcn', @(x) im2double(imread(x)));

pxds = pixelLabelDatastore(trainPath, ["N","B"], [0 1], ...
    'FileExtensions', {'.png'}, ...
    'ReadFcn', @(x) imread(x));

dsTrain = combine(imds, pxds);

%% Augmentation & Resize
dsTrain = transform(dsTrain, @(data) preprocessData(data, inputSize));

%% DeepLabv3+ (ResNet-50)
numClasses = 2;
lgraph = deeplabv3plusLayers([256 256 3], numClasses, 'resnet50');

tbl = countEachLabel(pxds);
classWeights = 1 ./ (tbl.PixelCount / sum(tbl.PixelCount));

pxLayer = pixelClassificationLayer( ...
    'Name', 'pixelLabels', ...
    'Classes', tbl.Name, ...
    'ClassWeights', classWeights);

lgraph = replaceLayer(lgraph, 'classification', pxLayer);

%% Training
options = trainingOptions('sgdm', ...
    'MaxEpochs', 25, ...
    'MiniBatchSize', 8, ...
    'Shuffle', "every-epoch", ...
    'Plots', 'training-progress');

net = trainNetwork(dsTrain, lgraph, options);

%% ---- ANALYZE TRAINING SET ----
fprintf('\n=== TRAINING SET ANALYSIS ===\n');

f_train = dir(fullfile(trainPath, '*.jpg'));
gt_train = dir(fullfile(trainPath, '*_mask.png'));
N_train = length(f_train);

FM_train = zeros(N_train,1);
CR_train = zeros(N_train,1);
CM_train = zeros(N_train,1);
GT_area_train = zeros(N_train,1);
Seg_area_train = zeros(N_train,1);

for l = 1:N_train
    img = im2double(imresize( ...
        imread(fullfile(trainPath, f_train(l).name)), inputSize));

    mask_gt = imresize( ...
        imread(fullfile(trainPath, gt_train(l).name)), ...
        inputSize, 'nearest');

    mask_gt = logical(mask_gt > 0);

    mask_pred = semanticseg(img, net) == 'B';
    mask_pred = imfill(mask_pred, 'holes');
    mask_pred = imclose(mask_pred, strel('disk',3));
    mask_pred = bwareafilt(mask_pred, 1);

    [TP,FP,FN,CR,CM,FM] = evaluation_segmentation(mask_pred, mask_gt); %#ok<ASGLU>

    FM_train(l) = FM;
    CR_train(l) = CR;
    CM_train(l) = CM;

    GT_area_train(l) = sum(mask_gt(:)) / numel(mask_gt);
    Seg_area_train(l) = sum(mask_pred(:)) / numel(mask_pred);
end

T_train = table({f_train.name}', CR_train, CM_train, FM_train, ...
    GT_area_train, Seg_area_train, ...
    'VariableNames', {'FrameName','CR','CM','FM','GT_area','Seg_area'});

writetable(T_train, 'train_segmentation_metrics.xlsx');

%% ---- ANALYZE TEST SET ----
fprintf('\n=== TEST SET ANALYSIS ===\n');

f_test = dir(fullfile(testPath, '*.jpg'));
gt_test = dir(fullfile(testPath, '*_mask.png'));
N_test = length(f_test);

FM_test = zeros(N_test,1);
CR_test = zeros(N_test,1);
CM_test = zeros(N_test,1);
GT_area_test = zeros(N_test,1);
Seg_area_test = zeros(N_test,1);

for l = 1:N_test
    img = im2double(imresize( ...
        imread(fullfile(testPath, f_test(l).name)), inputSize));

    mask_gt = imresize( ...
        imread(fullfile(testPath, gt_test(l).name)), ...
        inputSize, 'nearest');

    mask_gt = logical(mask_gt > 0);

    mask_pred = semanticseg(img, net) == 'B';
    mask_pred = imfill(mask_pred, 'holes');
    mask_pred = imclose(mask_pred, strel('disk',3));
    mask_pred = bwareafilt(mask_pred, 1);

    [TP,FP,FN,CR,CM,FM] = evaluation_segmentation(mask_pred, mask_gt); %#ok<ASGLU>

    FM_test(l) = FM;
    CR_test(l) = CR;
    CM_test(l) = CM;

    GT_area_test(l) = sum(mask_gt(:)) / numel(mask_gt);
    Seg_area_test(l) = sum(mask_pred(:)) / numel(mask_pred);
end

T_test = table({f_test.name}', CR_test, CM_test, FM_test, ...
    GT_area_test, Seg_area_test, ...
    'VariableNames', {'FrameName','CR','CM','FM','GT_area','Seg_area'});

writetable(T_test, 'test_segmentation_metrics.xlsx');

%% Visualization - F-measure boxplot
figure('Name','Boxplot FM');
boxplot([FM_train; FM_test], ...
    [repmat({'Train'}, N_train,1); repmat({'Test'}, N_test,1)]);
ylabel('F-measure (FM)');
title('FM distribution: Train vs Test');
saveas(gcf,'boxplot_FM_train_test.png');

%% Visualization - Area comparison
figure('Name','Area Percentage Comparison');

subplot(1,2,1);
histogram(GT_area_train,20); hold on;
histogram(Seg_area_train,20);
xlabel('Area ratio'); ylabel('Count');
legend('GT area','Segmented area');
title('TRAIN Area ratios');

subplot(1,2,2);
histogram(GT_area_test,20); hold on;
histogram(Seg_area_test,20);
xlabel('Area ratio'); ylabel('Count');
legend('GT area','Segmented area');
title('TEST Area ratios');

saveas(gcf,'area_comparison_train_test.png');

%% Best / Worst cases
[~,bestTr]  = max(FM_train);
[~,worstTr] = min(FM_train);
[~,bestTs]  = max(FM_test);
[~,worstTs] = min(FM_test);

% Best train
I = im2double(imresize(imread(fullfile(trainPath, f_train(bestTr).name)), inputSize));
BW = semanticseg(I, net) == 'B';
BW = imfill(BW,'holes');
BW = imclose(BW,strel('disk',3));
BW = bwareafilt(BW,1);

figure('Name','Best Train');
subplot(1,2,1); imshow(I); title('Best Train: Original');
subplot(1,2,2); imshow(BW); title(sprintf('Best Train: Mask | FM=%.2f', FM_train(bestTr)));
saveas(gcf,'best_train_case.png');

% Worst train
I = im2double(imresize(imread(fullfile(trainPath, f_train(worstTr).name)), inputSize));
BW = semanticseg(I, net) == 'B';
BW = imfill(BW,'holes');
BW = imclose(BW,strel('disk',3));
BW = bwareafilt(BW,1);

figure('Name','Worst Train');
subplot(1,2,1); imshow(I); title('Worst Train: Original');
subplot(1,2,2); imshow(BW); title(sprintf('Worst Train: Mask | FM=%.2f', FM_train(worstTr)));
saveas(gcf,'worst_train_case.png');

% Best test
I = im2double(imresize(imread(fullfile(testPath, f_test(bestTs).name)), inputSize));
BW = semanticseg(I, net) == 'B';
BW = imfill(BW,'holes');
BW = imclose(BW,strel('disk',3));
BW = bwareafilt(BW,1);

figure('Name','Best Test');
subplot(1,2,1); imshow(I); title('Best Test: Original');
subplot(1,2,2); imshow(BW); title(sprintf('Best Test: Mask | FM=%.2f', FM_test(bestTs)));
saveas(gcf,'best_test_case.png');

% Worst test
I = im2double(imresize(imread(fullfile(testPath, f_test(worstTs).name)), inputSize));
BW = semanticseg(I, net) == 'B';
BW = imfill(BW,'holes');
BW = imclose(BW,strel('disk',3));
BW = bwareafilt(BW,1);

figure('Name','Worst Test');
subplot(1,2,1); imshow(I); title('Worst Test: Original');
subplot(1,2,2); imshow(BW); title(sprintf('Worst Test: Mask | FM=%.2f', FM_test(worstTs)));
saveas(gcf,'worst_test_case.png');

%% Print averages
fprintf('\n--- AVERAGE METRICS ---\n');
fprintf('Train: FM=%.3f, CR=%.3f, CM=%.3f\n', ...
    mean(FM_train), mean(CR_train), mean(CM_train));
fprintf('Test: FM=%.3f, CR=%.3f, CM=%.3f\n', ...
    mean(FM_test), mean(CR_test), mean(CM_test));

%% Local Functions
function [TP, FP, FN, CR, CM, FM] = evaluation_segmentation(BW, GT)
    TP = sum(BW(:)==1 & GT(:)==1);
    FP = sum(BW(:)==1 & GT(:)==0);
    FN = sum(BW(:)==0 & GT(:)==1);

    CR = TP / (TP + FP + eps);
    CM = TP / (TP + FN + eps);
    FM = 2 * CM * CR / (CM + CR + eps);
end

function dataOut = preprocessData(data, inputSize)
    I = data{1};
    GT = data{2};

    rotAngle = 90 * randi(4) - 90;

    I = imrotate(I, rotAngle, 'bilinear', 'crop');
    GT = imrotate(GT, rotAngle, 'nearest', 'crop');

    if rand > 0.5
        I = fliplr(I);
        GT = fliplr(GT);
    end

    if rand > 0.5
        I = flipud(I);
        GT = flipud(GT);
    end

    I = imresize(I, inputSize);
    GT = imresize(GT, inputSize, 'nearest');

    dataOut = {I, GT};
end
