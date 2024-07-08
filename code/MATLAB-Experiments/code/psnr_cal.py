import os
import numpy as np
import pandas as pd
from PIL import Image
import re
from skimage.metrics import peak_signal_noise_ratio as psnr

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
            
            cho_result_arr = np.array(cho_result_img)
            gt_arr = np.array(gt_img)
            
            mse = np.mean((cho_result_arr - gt_arr) ** 2)

            psnr_value = 10 * np.log10(255.0**2 / mse)
            results_df[file_name.replace('_out.png', '')] = [psnr_value]


    results_df.index = [f'{results_dir.split("/")[-1].split("/")[0]}']
    results_df.to_csv(f'{results_dir.split("/")[-1].split("/")[0]}_psnr.csv')


import os
import pandas as pd

csv_dir = '/Users/apple/Downloads/outlier_public/all_psnr_results'
csv_files = [file for file in os.listdir(csv_dir) if file.endswith('.csv')]
combined_df = pd.DataFrame()

for file in csv_files:
    file_path = os.path.join(csv_dir, file)
    df = pd.read_csv(file_path)
    combined_df = pd.concat([combined_df, df], ignore_index=True)

combined_csv_path = os.path.join(csv_dir, 'combined_psnr_results.csv')
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

import plotly.graph_objects as go

images = ['im1', 'im2', 'im3', 'im4', 'Avg.']
l1 = [30.34,25.58,30.75,27.41,28.52]
l2 = [31.16,26.08,31.75,27.75,29.17]
l3 = [29,21.75,27.67,22,25.105]
l4 = [25.08,18.08,25.91,19.67,22.16]
l5 = [32.25,25.22,33.31,28.26,29.76]
fig = go.Figure()

fig.add_trace(go.Bar(
    x=images,
    y=l1,
    name='Cho',
    marker=dict(opacity=0.8)
))

fig.add_trace(go.Bar(
    x=images,
    y=l2,
    name='Xu',
    marker=dict(opacity=0.8)
))

fig.add_trace(go.Bar(
    x=images,
    y=l3,
    name='Shan',
    marker=dict(opacity=0.8)
))

fig.add_trace(go.Bar(
    x=images,
    y=l4,
    name='Fergus',
    marker=dict(opacity=0.8)
))

fig.add_trace(go.Bar(
    x=images,
    y=l5,
    name='OID',
    marker=dict(opacity=0.8)
))

y_min = 10
y_max = 35
fig.update_yaxes(range=[y_min, y_max])

fig.update_layout(
    # title='Blurry Image vs OID',
    # xaxis_title='Image',
    yaxis_title='Average PSNR',
    barmode='group',
    legend=dict(
        x=0.01,
        y=0.99,
        traceorder='normal',
        font=dict(size=16),
        bgcolor='rgba(255, 255, 255, 0.5)'
    ),
    font=dict(
        size=18
    )
)

output_path = '/Users/apple/Downloads/outlier_public/no_outlier_psnr.png'
fig.write_image(output_path)
