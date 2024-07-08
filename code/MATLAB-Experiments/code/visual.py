import os
import pandas as pd
import plotly.graph_objects as go

def extract_number(file_name):
    return float(file_name.split('_')[-1].replace('.csv', ''))

csv_files = [file for file in os.listdir() if file.startswith('oid_psnr_value_list_density_') and file.endswith('.csv')]
csv_files = sorted(csv_files, key=extract_number)
combined_df = pd.DataFrame()
new_columns = [f"{i}-{j}" for i in [1, 4, 7, 10, 13] for j in range(1, 9)]
for file in csv_files:
    df = pd.read_csv(file, header=None)
    density = float(file.split('_')[-1].replace('.csv', ''))
    df_reordered = pd.concat([df.iloc[:, 16:], df.iloc[:, :16]], axis=1)
    df_reordered.columns = new_columns
    df_reordered.insert(0, 'density', density)
    combined_df = pd.concat([combined_df, df_reordered], ignore_index=True)
combined_df['average'] = combined_df.iloc[:, 2:].mean(axis=1) / 2
combined_df.insert(1, 'avg', combined_df.pop('average'))
combined_df.to_csv('combined_oid_psnr_values.csv', index=False)



df = pd.read_csv('/Users/apple/Downloads/outlier_public/temp/combined_oid_psnr_values.csv')
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df['density'], 
    y=df['avg'], 
    mode='lines+markers',
    marker=dict(size=10),
    line=dict(width=2),
))

fig.update_layout(
    # title='Average PSNR vs Noise Density',
    xaxis_title='Noise Density',
    yaxis_title='Average PSNR',
    font=dict(
        size=18
    ),
)

y_min = 25
y_max = 35
fig.update_yaxes(range=[y_min, y_max])

fig.write_image('average_psnr_vs_noise_density.png')




import plotly.graph_objects as go
import pandas as pd

file_path = '/Users/apple/Downloads/outlier_public/temp/data_fidelity.csv'
data = pd.read_csv(file_path)


data_filtered = data[data['res'] <= 0.08]
data_filtered = data[(data['res'] <= 0.08) & 
                     ~((data['res'] >= 0.01) & (data['res'] <= 0.055) & (data['energy'] <= 0.0001))]
data_sorted = data_filtered.sort_values(by='res')
sampled_data = data_sorted.iloc[::100]

fig = go.Figure(data=go.Scatter(
    x=sampled_data['res'],
    y=sampled_data['energy'],
    mode='markers',
    marker=dict(
        size=3,
        color=sampled_data['energy'],
        colorscale='Viridis',
        showscale=True
    )
))

fig.update_layout(
    title='',
    xaxis_title='Residual',
    yaxis_title='Energy of the data fidelity term',
    hovermode='closest',
    font=dict(
        size=18
    ),
)

output_path = '/Users/apple/Downloads/outlier_public/scatter_plot_sampled_filtered.png'
fig.write_image(output_path)




file_path = '/Users/apple/Downloads/outlier_public/oid_psnr_value_list.csv'
data = pd.read_csv(file_path, header=None)
values = list(data.iloc[0])
kernel_list = []

for start in range(8):
    subset = [values[start + i * 8] for i in range(15)]
    kernel_list.append(sum(subset) / 30)

x_values = [i + 1 for i in range(len(kernel_list))]
fig = go.Figure(data=go.Bar(
    x=x_values,
    y=kernel_list,
    # marker=dict(
    #     color='rgb(49,130,189)'
    # )
))

y_min = 25
y_max = 40
fig.update_yaxes(range=[y_min, y_max])

fig.update_layout(
    font=dict(
        size=18
    ),
    xaxis_title='Kernel Index',
    yaxis_title='Average PSNR',
    xaxis=dict(
        tickmode='linear',
        tick0=1,
        dtick=1
    )
    # hovermode='closest'
)


output_path = '/Users/apple/Downloads/outlier_public/kernel_psnr.png'
fig.write_image(output_path)



file_path = '/Users/apple/Downloads/outlier_public/oid_psnr_value_list.csv'
data = pd.read_csv(file_path, header=None)
values = list(data.iloc[0])
kernel_list = []

for start in range(15):
    start = start * 8
    subset = [values[start + i] for i in range(8)]
    kernel_list.append(sum(subset) / 16)

x_values = [i + 1 for i in range(len(kernel_list))]
fig = go.Figure(data=go.Bar(
    x=x_values,
    y=kernel_list,
    # marker=dict(
    #     color='rgb(49,130,189)'
    # )
))

y_min = 25
y_max = 40
fig.update_yaxes(range=[y_min, y_max])

fig.update_layout(
    # title='Average PSNR vs Kernel Index',
    font=dict(
        size=18  # 设置全局字体大小
    ),
    xaxis_title='Image Index',
    yaxis_title='Average PSNR',
    xaxis=dict(
        tickmode='linear',
        tick0=1,
        dtick=1
    )
    # hovermode='closest'
)
output_path = '/Users/apple/Downloads/outlier_public/kernel_psnr.png'
fig.write_image(output_path)


import pandas as pd
import plotly.graph_objects as go


x_values = list(df.columns)[1:]
y_values = df.iloc[0:, 1:].values

legend_names = df.iloc[0:, 0].values
fig = go.Figure()

for i in range(len(y_values)):
    fig.add_trace(go.Scatter(x=x_values, y=y_values[i], mode='lines+markers', name=legend_names[i]))

fig.update_layout(
    # title="Error Ratio vs. Success Rate",
    xaxis_title="Error Ratio",
    yaxis_title="Success Rate",
    font = dict(
        size=18
    ),
    legend=dict(
        x=0,
        y=1,
        xanchor='left',
        yanchor='top',
        bgcolor='rgba(255, 255, 255, 0.5)'
    )
)

output_path = '/Users/apple/Downloads/outlier_public/error_ratio.png'
fig.write_image(output_path)

