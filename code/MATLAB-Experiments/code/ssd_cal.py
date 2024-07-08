import os
import numpy as np
import pandas as pd
from PIL import Image
import re

def custom_sort(file_name):
    match = re.match(r'(\d+)_(\d+)', file_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return float('inf'), float('inf')

all_dirs = [
            "/Users/apple/Downloads/outlier_public/impulsive_noise/OID",
            "/Users/apple/Downloads/outlier_public/impulsive_noise/chen_results",
            "/Users/apple/Downloads/outlier_public/impulsive_noise/cho_results",
            "/Users/apple/Downloads/outlier_public/impulsive_noise/dong_results",
            "/Users/apple/Downloads/outlier_public/impulsive_noise/pan_results",
            "/Users/apple/Downloads/outlier_public/impulsive_noise/xul0_results",
            "/Users/apple/Downloads/outlier_public/impulsive_noise/xutwo_results",
            "/Users/apple/Downloads/outlier_public/impulsive_noise/zhong_results",
            "/Users/apple/Downloads/outlier_public/impulsive_noise/blurry_orig"
            ]

gt_dir = '/Users/apple/Downloads/outlier_public/impulsive_noise/gt'

for results_dir in all_dirs:
    results_df = pd.DataFrame()
    dir = sorted(os.listdir(results_dir), key=custom_sort)

    for file_name in dir:
        if file_name.endswith('.png'):
            gt_file_name = file_name.split('_')[0] + '.jpg'
            
            cho_result_path = os.path.join(results_dir, file_name)
            gt_path = os.path.join(gt_dir, gt_file_name)
            
            cho_result_img = Image.open(cho_result_path).convert('RGB')
            gt_img = Image.open(gt_path).convert('RGB')
            
            cho_result_arr = np.array(cho_result_img, dtype=np.float32) / 255.0
            gt_arr = np.array(gt_img, dtype=np.float32) / 255.0
            
            ssd_error = np.sum((cho_result_arr - gt_arr) ** 2)
            
            results_df[file_name.replace('_out.png', '')] = [ssd_error]


    results_df.index = [f'{results_dir.split("/")[-1].split("/")[0]}']
    results_df.to_csv(f'{results_dir.split("/")[-1].split("/")[0]}_ssd_errors.csv')




import os
import pandas as pd

csv_dir = '/Users/apple/Downloads/outlier_public/all_ssd_results'
csv_files = [file for file in os.listdir(csv_dir) if file.endswith('.csv')]
combined_df = pd.DataFrame()

for file in csv_files:
    file_path = os.path.join(csv_dir, file)
    df = pd.read_csv(file_path)
    combined_df = pd.concat([combined_df, df], ignore_index=True)

combined_csv_path = os.path.join(csv_dir, 'combined_ssd_results.csv')
combined_df.to_csv(combined_csv_path, index=False)



import pandas as pd
df = pd.read_csv('/Users/apple/Downloads/outlier_public/combined_ssd_results.csv')

blurry_orig_values = df.loc[df['Unnamed: 0'] == 'blurry_orig'].iloc[:, 1:].values.flatten()
df.iloc[0:, 1:] = df.iloc[0:, 1:].div(blurry_orig_values, axis=1)
df.to_csv('ssd_ratio.csv', index=False)

import pandas as pd
import numpy as np

df = pd.read_csv('/Users/apple/Downloads/outlier_public/combined_ssd_ratio.csv')
threshold_list = np.arange(0.1, 1.1, 0.1)
result_df = pd.DataFrame(index=df.index)

for threshold in threshold_list:
    column_name = f'{threshold:.1f}' 
    threshold_count = (df.iloc[0:, 1:] < threshold).sum(axis=1)
    threshold_ratio = threshold_count / 120 * 100
    result_df[column_name] = threshold_ratio
# result_df.iloc[:, 0] = df.iloc[:, 0]
result_df.to_csv('threshold_ratios.csv')