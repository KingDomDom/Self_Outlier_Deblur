% 定义文件夹路径
img_folder = 'lowlight/img';
kernel_folder = 'lowlight/kernel';
output_folder = 'lowlight/output';

if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

img_files = dir(fullfile(img_folder, '*.png'));
kernel_files = dir(fullfile(kernel_folder, '*.png'));

for i = 1:length(img_files)
    img_path = fullfile(img_folder, img_files(i).name);
    img = imread(img_path);
    
    if size(img, 3) == 3
%         img = rgb2gray(img);
        img = im2double(img);
    end
    
    for j = 1:length(kernel_files)
        kernel_path = fullfile(kernel_folder, kernel_files(j).name);
        kernel = imread(kernel_path);
        
        if size(kernel, 3) == 3
            kernel = rgb2gray(kernel);
        end
        
        kernel = double(kernel) / sum(kernel(:));
        blurred_img = imfilter(double(img), double(kernel), 'conv', 'replicate');
        
        noisy_img = imnoise(blurred_img, 'salt & pepper', 0.01);
        noisy_img = im2uint8(noisy_img);
        output_filename = sprintf('%d_%d.png', i, j);
        output_path = fullfile(output_folder, output_filename);
        imwrite(noisy_img, output_path);
    end
end
