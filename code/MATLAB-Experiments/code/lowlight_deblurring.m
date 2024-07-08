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


dirPath = '/Users/apple/Downloads/outlier_public/lowlight/OID';
imageFiles = dir(fullfile(dirPath, '*.png'));
fileNames = {imageFiles.name};
% fileNames = sort(fileNames);
sorted_paths = fullfile(dirPath, fileNames);

ssd_error_list = zeros(1, 120);
oid_psnr_value_list = zeros(1, 48);

for i = 1:length(sorted_paths)
    disp(sorted_paths{i});
    y = imread(sorted_paths{i});
%     y = im2double(y);
    [filepath, name, ext] = fileparts(sorted_paths{i});
    underscore_index = strfind(name, '_');
    if ~isempty(underscore_index)
        number = name(1:underscore_index(1)-1);
    else
        number = name;
    end
    gt_path = fullfile(filepath, '..', 'img', [number '.png']);
    
    y_gt = imread(gt_path);
%     y_gt = im2double(y_gt);
    disp(gt_path);

    %% Compute PSNR
    mse = sum((y(:) - y_gt(:)).^2) / numel(y);
%     max_possible_value = 1;
    max_possible_value = 255;
    psnr_value = 10 * log10((max_possible_value^2) / mse);
    oid_psnr_value_list(i) = psnr_value;
    disp(psnr_value);

    saveKernelPath = fullfile(saveDir, [fileName, '_kernel.png']);
    saveLatentPath = fullfile(saveDir, [fileName, '_latent.png']);
    imwrite(k,saveKernelPath);
    imwrite(Latent,saveLatentPath);
end

writematrix(oid_psnr_value_list, 'goa_lowlight_psnr_value_list.csv');
