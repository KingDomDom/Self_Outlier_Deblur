clc;
clear;
close all;
addpath(genpath('image'));
addpath(genpath('main_code'));
opts.prescale = 0; %%downsampling
opts.xk_iter = 4; %% the iterations
opts.last_iter = 4; %% larger if image is very noisy
opts.k_thresh = 20;

opts.isnoisy = 1; %% filter the input for coarser scales before deblurring 0 or 1
opts.kernel_size = 27;  %% kernel size
opts.predeblur = 'L0';  %% deblurring method for coarser scales; Lp or L0
filename = './example.png'; 
lambda_grad = 4e-3; %% range(1e-3 - 2e-2)
opts.gamma_correct = 1.0; %% range (1.0-2.2)


dirPath = '/Users/apple/Downloads/outlier_public/impulsive_noise/blurry';
imageFiles = dir(fullfile(dirPath, '*.png'));
fileNames = {imageFiles.name};
% fileNames = sort(fileNames);
sorted_paths = fullfile(dirPath, fileNames);

dirPath = '/Users/apple/Downloads/outlier_public/impulsive_noise/gt';
gtFiles = dir(fullfile(dirPath, '*.jpg'));
gtfileNames = {gtFiles.name};
% gtfileNames = sort(gtfileNames);
gt_sorted_paths = fullfile(dirPath, gtfileNames);

ssd_error_list = zeros(1, 40);
oid_psnr_value_list = zeros(1, 40);

noise_density = 0.15:0.05:0.5;
noise_density_list = (noise_density - 0.1) / 0.9;

for density = noise_density_list
    fprintf('Noise density: %.3f\n', density);
    for i = 1:length(sorted_paths)
        disp(sorted_paths{i});
        y = imread(sorted_paths{i});
        if size(y,3)==3
            yg = im2double(rgb2gray(y));
        else
            yg = im2double(y);
        end
        yg = imnoise(yg, 'salt & pepper', density);

        tic;
        [kernel, interim_latent] = blind_deconv(yg, lambda_grad, opts);
        toc
        %% Algorithm is done!
        [filepath, name, ext] = fileparts(sorted_paths{i});
        underscore_index = strfind(name, '_');
        if ~isempty(underscore_index)
            number = name(1:underscore_index(1)-1);
        else
            number = name;
        end
        gt_path = fullfile(filepath, '..', 'gt', [number '.jpg']);


        y = im2double(y);
        Latent = image_estimate(y, kernel, 0.003,0);
        
        y_gt = imread(gt_path);
        y_gt = im2double(y_gt);
        disp(gt_path);

        %% Compute PSNR
        mse = sum((Latent(:) - y_gt(:)).^2) / numel(y);
    %     max_possible_value = max(y_gt(:));
        max_possible_value = 255;
        psnr_value = 10 * log10((max_possible_value^2) / mse);
        oid_psnr_value_list(i) = psnr_value;

        k = kernel - min(kernel(:));
        k = k./max(k(:));
        saveKernelPath = fullfile(saveDir, [fileName, '_kernel.png']);
        saveLatentPath = fullfile(saveDir, [fileName, '_latent.png']);
        imwrite(k,saveKernelPath);
        imwrite(Latent,saveLatentPath);
    end
    psnr_filename = sprintf('oid_psnr_value_list_density_%.3f.csv', density);
    writematrix(ssd_error_list, 'oid_ssd_errors.csv');
    writematrix(oid_psnr_value_list, psnr_filename);
end
