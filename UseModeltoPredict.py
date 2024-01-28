import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import mean_absolute_error
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from matplotlib.ticker import StrMethodFormatter
import pandas_ta as ta

model = tf.keras.models.load_model('/home/grouptan/Documents/yudian/data/linebyline1092singlepitchtraining/tf_model')

import os
df_actual = pd.DataFrame([])
df_pred = pd.DataFrame([])
R_prediction = round(1.6, 1)
# Specify the path for the new folder
folder_path = '/home/grouptan/Documents/yudian/data/linebyline1092prediction'+str(R_prediction)  # Replace with the desired path

# Check if the folder already exists or not
if not os.path.exists(folder_path):
    # Create the folder if it doesn't exist
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created successfully.")
else:
    print(f"Folder '{folder_path}' already exists.")
    
    
train_test_ape = []
head_size_list = []
num_head_list = []
ff_dim_list = []
num_transformer_blocks_list = []
ape_label = []
time_list = []
sequence_length_list = []
n_list = []

backcandles = 1

url = "https://raw.githubusercontent.com/yd145763/DifferentPitchML/main/pitch"+str(R_prediction)+"um.csv"
df1 = pd.read_csv(url)
df1 = df1.iloc[:, 1:]
N = np.arange(0,388,1)
for n in N:
    n_list.append(n)
    start = time.time()
    #prediction dataset
    df = pd.DataFrame([])
    e = df1.iloc[n,:].to_list()
    x = (df1.columns).to_list()
    x = [float(i) for i in x]
    x = [i+15 for i in x]
    df['x'] = x
    df['e'] = e 
    df['pitch'] = int(round(R_prediction,1)*1000)
    df['RSI'] = ta.rsi(df['e'], length = 10)
    df['EMAF'] = ta.ema(df['e'], length = 20)
    df['EMAM'] = ta.ema(df['e'], length = 30)
    df['EMAS'] = ta.ema(df['e'], length = 40)
    df['TargetNextClose'] = df['e'].shift(-3)
    
    data_full_filtered_sorted_shortened2 = df.sort_values(by='x', axis=0)

    data2 = data_full_filtered_sorted_shortened2[['x','e', 'RSI', 'EMAF', 'EMAM', 'EMAS', 'TargetNextClose']]

    data2.dropna(inplace = True)
    data2.reset_index(inplace = True)
    data2.drop(['index'], axis=1, inplace = True)
    data_set2 = data2
    pd.set_option('display.max_columns', None)


    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range=(0,1))
    data_set_scaled2 = sc.fit_transform(data_set2)
    
    print(data_set_scaled2)
    

    X2 = []

    for j in range(data_set_scaled2.shape[1]-1):
        X2.append([])
        for i in range(backcandles, data_set_scaled2.shape[0]):
            X2[j].append(data_set_scaled2[i-backcandles:i, j])
            print(data_set_scaled2[i-backcandles:i, j])
            print(" ")
    X2 = np.moveaxis(X2, [0], [2])
    X_test2 = np.array(X2)
    yi2 = np.array(data_set_scaled2[backcandles:,-1])
    y_test2 = np.reshape(yi2, (len(yi2), 1))
    
    y_pred = model.predict(X_test2)
    
    nextclose = np.array(data2['TargetNextClose'])
    nextclose = nextclose.reshape(-1, 1)

    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(nextclose)
    denormalized_data = scaler.inverse_transform(normalized_data)
    y_pred_ori = scaler.inverse_transform(y_pred)
    y_test_ori = scaler.inverse_transform(y_test2)

    x_plot = df['x']
    x_plot = [i for i in x_plot]
    y_pred_ori = y_pred_ori[:,0]
    y_test_ori = y_test_ori[:,0]

    x_plot1 = x_plot[(len(x_plot)-len(y_test_ori)):]
    x_plot1 = np.array(x_plot1, dtype=np.float64)



    diff = (pd.Series(y_test_ori.flatten()) - pd.Series(y_pred_ori.flatten())).abs()
    rel_error = diff / pd.Series(y_test_ori.flatten())
    pct_error = rel_error * 100
    ape = pct_error.mean()
    ape_label.append(ape)
    
    df_actual['e'+str(n)] = y_test_ori
    df_pred['e'+str(n)] = y_pred_ori

    fig = plt.figure(figsize=(20, 13))
    ax = plt.axes()
    ax.scatter(x_plot1,[i*1000 for i in y_test_ori], s=50, facecolor='blue', edgecolor='blue')
    ax.plot(x_plot1,[i*1000 for i in y_pred_ori], color = "red", linewidth = 5)
    #graph formatting     
    ax.tick_params(which='major', width=5.00)
    ax.tick_params(which='minor', width=5.00)
    ax.xaxis.label.set_fontsize(35)
    ax.xaxis.label.set_weight("bold")
    ax.yaxis.label.set_fontsize(35)
    ax.yaxis.label.set_weight("bold")
    ax.tick_params(axis='both', which='major', labelsize=35)
    ax.set_yticklabels(ax.get_yticks(), weight='bold')
    ax.set_xticklabels(ax.get_xticks(), weight='bold')
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines['bottom'].set_linewidth(5)
    ax.spines['left'].set_linewidth(5)
    plt.xlabel("x (µm)")
    plt.ylabel("E-field (meV)")
    plt.legend(["Original Data", "Predicted Data"], prop={'weight': 'bold','size': 35}, loc = "best")
    plt.show()
    plt.close()
    
    stop = time.time()
    time_list.append(stop-start)

df_pred['x'] = x_plot1
df_actual['x'] = x_plot1

df_result = pd.DataFrame()

df_result['ape_label'] = ape_label
df_result['time_list'] = time_list

df_result['n_list'] = n_list


df_result.to_csv(folder_path+'/df_result.csv')
df_pred.to_csv(folder_path+'/df_pred.csv')
df_actual.to_csv(folder_path+'/df_actual.csv')

df_actual.set_index('x', inplace=True)
df_actual = df_actual.transpose()

df_pred.set_index('x', inplace=True)
df_pred = df_pred.transpose()

y1 = np.linspace(0, 60, len(df1.index.tolist()))
x1 = df_actual.columns.values

colorbarmax1 = df_actual.max().max()
X,Y = np.meshgrid(x1,y1)
fig = plt.figure(figsize=(18, 4))
ax = plt.axes()
cp=ax.contourf(X,Y,df_actual, 200, zdir='z', offset=-100, cmap='viridis')
clb1=fig.colorbar(cp, ticks=(np.around(np.linspace(0.0, colorbarmax1, num=6), decimals=3)).tolist())
clb1.ax.set_title('Electric Field (eV)', fontweight="bold")
for l in clb1.ax.yaxis.get_ticklabels():
    l.set_weight("bold")
    l.set_fontsize(15)
ax.set_xlabel('x-position (µm)', fontsize=20, fontweight="bold", labelpad=1)
ax.set_ylabel('z-position (µm)', fontsize=20, fontweight="bold", labelpad=1)
ax.xaxis.label.set_fontsize(20)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(20)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
ax.set_xticklabels(ax.get_xticks(), weight='bold')
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
plt.savefig(folder_path+'/Actual.jpg', format='jpg')
plt.show()
plt.close()

colorbarmax2 = df_pred.max().max()
X,Y = np.meshgrid(x1,y1)
fig = plt.figure(figsize=(18, 4))
ax = plt.axes()
cp=ax.contourf(X,Y,df_pred, 200, zdir='z', offset=-100, cmap='viridis')
clb2=fig.colorbar(cp, ticks=(np.around(np.linspace(0.0, colorbarmax2, num=6), decimals=3)).tolist())
clb2.ax.set_title('Electric Field (eV)', fontweight="bold")
for l in clb2.ax.yaxis.get_ticklabels():
    l.set_weight("bold")
    l.set_fontsize(15)
ax.set_xlabel('x-position (µm)', fontsize=20, fontweight="bold", labelpad=1)
ax.set_ylabel('z-position (µm)', fontsize=20, fontweight="bold", labelpad=1)
ax.xaxis.label.set_fontsize(20)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(20)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
ax.set_xticklabels(ax.get_xticks(), weight='bold')
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
plt.savefig(folder_path+'/Predicted.jpg', format='jpg')
plt.show()
plt.close()

