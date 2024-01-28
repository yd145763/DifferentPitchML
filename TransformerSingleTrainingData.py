# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 11:14:56 2024

@author: limyu
"""



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

R_training = round(0.6, 1)
R_prediction = round(1.6, 1)

df_actual = pd.DataFrame([])
df_pred = pd.DataFrame([])
df_training = pd.DataFrame([])
df_validation = pd.DataFrame([])

import os
# Specify the path for the new folder
folder_path = '/home/grouptan/Documents/yudian/data/linebyline1092singlepitchtraining'  # Replace with the desired path

# Check if the folder already exists or not
if not os.path.exists(folder_path):
    # Create the folder if it doesn't exist
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created successfully.")
else:
    print(f"Folder '{folder_path}' already exists.")

backcandlesS = 5,10,20

head_sizeS=16,32,64
num_headsS=2,3,4
ff_dimS=2,3,4
num_transformer_blocksS=2,3,4

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
num_transformer_blocks=1
num_heads = 2
ff_dim = 2
head_size = 16

n = 380

#setup the model first
url = "https://raw.githubusercontent.com/yd145763/DifferentPitchML/main/pitch"+str(R_training)+"um.csv"
df1 = pd.read_csv(url)
df1 = df1.iloc[:, 1:]
x1 = np.linspace(0, 80, len(df1.columns))
y1 = np.linspace(0, 60, len(df1.index.tolist()))

colorbarmax = df1.max().max()

X,Y = np.meshgrid(x1,y1)
fig = plt.figure(figsize=(18, 4))
ax = plt.axes()
cp=ax.contourf(X,Y,df1, 200, zdir='z', offset=-100, cmap='viridis')
clb=fig.colorbar(cp, ticks=(np.around(np.linspace(0.0, colorbarmax, num=6), decimals=3)).tolist())
clb.ax.set_title('Electric Field (eV)', fontweight="bold")
for l in clb.ax.yaxis.get_ticklabels():
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
plt.show()
plt.close()

df = pd.DataFrame([])
e = df1.iloc[n,:].to_list()
x = (df1.columns).to_list()
x = [float(i) for i in x]
x = [i+15 for i in x]


df['x'] = x
df['e'] = e 
df['pitch'] = int(round(R_training,1)*1000)
df['RSI'] = ta.rsi(df['e'], length = 10)
df['EMAF'] = ta.ema(df['e'], length = 20)
df['EMAM'] = ta.ema(df['e'], length = 30)
df['EMAS'] = ta.ema(df['e'], length = 40)
df['TargetNextClose'] = df['e'].shift(-1)



#setup the model first

data_full_original = df
data_full = pd.DataFrame()

data_full_filtered_sorted = data_full_original.sort_values(by='x', axis=0)
data_full_filtered_sorted_shortened = data_full_filtered_sorted.iloc[:int(len(data_full_filtered_sorted['x'])*0.6),:]
data = data_full_filtered_sorted_shortened[['x','e', 'RSI', 'EMAF', 'EMAM', 'EMAS', 'TargetNextClose']]

data.dropna(inplace = True)
data.reset_index(inplace = True)
data.drop(['index'], axis=1, inplace = True)
data_set = data
pd.set_option('display.max_columns', None)
print(data_set.head(5))

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
data_set_scaled = sc.fit_transform(data_set)

print(data_set_scaled)

X = []

for j in range(data_set_scaled.shape[1]-1):
    X.append([])
    for i in range(backcandles, data_set_scaled.shape[0]):
        X[j].append(data_set_scaled[i-backcandles:i, j])
        print(data_set_scaled[i-backcandles:i, j])
        print(" ")
X = np.moveaxis(X, [0], [2])
X_train = np.array(X)
yi = np.array(data_set_scaled[backcandles:,-1])
y_train = np.reshape(yi, (len(yi), 1))


#functions to define transformer model

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1)(x)
    return keras.Model(inputs, outputs)

input_shape = X_train.shape[1:]

model = build_model(
    input_shape,
    head_size=head_size,
    num_heads=num_heads,
    ff_dim=ff_dim,
    num_transformer_blocks=num_transformer_blocks,
    mlp_units=[128],
    mlp_dropout=0.4,
    dropout=0.25,
)

model.compile(
    loss="mean_squared_error",
    optimizer=keras.optimizers.Adam(learning_rate=1e-4)
)

N = np.arange(0,388,1)


for n in N:
    n_list.append(n)

    url = "https://raw.githubusercontent.com/yd145763/DifferentPitchML/main/pitch"+str(R_training)+"um.csv"
    df1 = pd.read_csv(url)
    df1 = df1.iloc[:, 1:]
    df = pd.DataFrame([])
    e = df1.iloc[n,:].to_list()
    x = (df1.columns).to_list()
    x = [float(i) for i in x]
    x = [i+15 for i in x]
    df['x'] = x
    df['e'] = e 
    df['pitch'] = int(round(R_training,1)*1000)
    df['RSI'] = ta.rsi(df['e'], length = 10)
    df['EMAF'] = ta.ema(df['e'], length = 20)
    df['EMAM'] = ta.ema(df['e'], length = 30)
    df['EMAS'] = ta.ema(df['e'], length = 40)
    df['TargetNextClose'] = df['e'].shift(-1)


    #set training data
    
    data_full_filtered_sorted = df.sort_values(by='x', axis=0)
    data_full_filtered_sorted_shortened = data_full_filtered_sorted.iloc[:int(len(data_full_filtered_sorted['x'])*0.6),:]
    data = data_full_filtered_sorted_shortened[['x','e', 'RSI', 'EMAF', 'EMAM', 'EMAS', 'TargetNextClose']]

    data.dropna(inplace = True)
    data.reset_index(inplace = True)
    data.drop(['index'], axis=1, inplace = True)
    data_set = data
    pd.set_option('display.max_columns', None)
    print(data_set.head(5))
    
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range=(0,1))
    data_set_scaled = sc.fit_transform(data_set)
    
    print(data_set_scaled)
    
    X = []
    
    for j in range(data_set_scaled.shape[1]-1):
        X.append([])
        for i in range(backcandles, data_set_scaled.shape[0]):
            X[j].append(data_set_scaled[i-backcandles:i, j])
            print(data_set_scaled[i-backcandles:i, j])
            print(" ")
    X = np.moveaxis(X, [0], [2])
    X_train = np.array(X)
    yi = np.array(data_set_scaled[backcandles:,-1])
    y_train = np.reshape(yi, (len(yi), 1))


        
    start = time.time()
    #add record transformer model parameters
    head_size_list.append(head_size)
    num_head_list.append(num_heads)
    ff_dim_list.append(ff_dim)
    num_transformer_blocks_list.append(num_transformer_blocks)
    sequence_length_list.append(backcandles)


   #set validation data
    data_full_filtered_sorted1 = df.sort_values(by='x', axis=0)
    data_full_filtered_sorted_shortened1 = data_full_filtered_sorted1.iloc[int(len(data_full_filtered_sorted1['x'])*0.6):,:]
    data1 = data_full_filtered_sorted_shortened1[['x','e', 'RSI', 'EMAF', 'EMAM', 'EMAS', 'TargetNextClose']]
    
    data1.dropna(inplace = True)
    data1.reset_index(inplace = True)
    data1.drop(['index'], axis=1, inplace = True)
    data_set1 = data1
    pd.set_option('display.max_columns', None)
    print(data_set1.head(5))

    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range=(0,1))
    data_set_scaled1 = sc.fit_transform(data_set1)
    
    print(data_set_scaled1)
    
    X1 = []
    
    for j in range(data_set_scaled1.shape[1]-1):
        X1.append([])
        for i in range(backcandles, data_set_scaled1.shape[0]):
            X1[j].append(data_set_scaled1[i-backcandles:i, j])
            print(data_set_scaled1[i-backcandles:i, j])
            print(" ")
    X1 = np.moveaxis(X1, [0], [2])
    X_test = np.array(X1)
    yi1 = np.array(data_set_scaled1[backcandles:,-1])
    y_test = np.reshape(yi1, (len(yi1), 1))



    #prediction dataset
    url = "https://raw.githubusercontent.com/yd145763/DifferentPitchML/main/pitch"+str(R_prediction)+"um.csv"
    df1 = pd.read_csv(url)
    df1 = df1.iloc[:, 1:]
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





    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=64,
        #callbacks=callbacks,
    )

    training_loss = pd.Series(history.history['loss'])
    validation_loss = pd.Series(history.history['val_loss'])
    
    df_training['loss'+str(n)] = training_loss
    df_validation['loss'+str(n)] = validation_loss
    
    diff = (validation_loss[50:] - training_loss[50:])
    ape = sum(diff)/len(diff)
    train_test_ape.append(ape)

    epochs = range(1, 100 + 1)

    fig = plt.figure(figsize=(20, 13))
    ax = plt.axes()
    ax.plot(epochs, training_loss, color = "blue", linewidth = 5)
    ax.plot(epochs, validation_loss, color = "red", linewidth = 5)
    #graph formatting     
    ax.tick_params(which='major', width=2.00)
    ax.tick_params(which='minor', width=2.00)
    ax.xaxis.label.set_fontsize(35)
    ax.xaxis.label.set_weight("bold")
    ax.yaxis.label.set_fontsize(35)
    ax.yaxis.label.set_weight("bold")
    ax.tick_params(axis='both', which='major', labelsize=35)
    ax.set_yticklabels(ax.get_yticks(), weight='bold')
    ax.set_xticklabels(ax.get_xticks(), weight='bold')
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines['bottom'].set_linewidth(5)
    ax.spines['left'].set_linewidth(5)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Training loss", "Validation Loss"], prop={'weight': 'bold','size': 35}, loc = "best")
    plt.savefig(folder_path+'/Training_Validation'+str(n)+'.jpg', format='jpg')
    plt.show()
    plt.close()



    y_pred = model.predict(X_test2)
    #y_pred=np.where(y_pred > 0.43, 1,0)



    nextclose = np.array(data['TargetNextClose'])
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
    plt.savefig(folder_path+'/Prediction_Actual'+str(n)+'.jpg', format='jpg')
    plt.show()
    plt.close()





    end = time.time()

    time_list.append(end-start)





df_pred['x'] = x_plot1
df_actual['x'] = x_plot1

df_result = pd.DataFrame()
df_result['train_test_ape'] = train_test_ape
df_result['head_size_list'] = head_size_list
df_result['num_head_list'] = num_head_list
df_result['ff_dim_list'] = ff_dim_list
df_result['num_transformer_blocks_list'] = num_transformer_blocks_list
df_result['ape_label'] = ape_label
df_result['time_list'] = time_list
df_result['sequence_length_list'] = sequence_length_list
df_result['n_list'] = n_list


df_result.to_csv(folder_path+'/df_result.csv')
df_pred.to_csv(folder_path+'/df_pred.csv')
df_actual.to_csv(folder_path+'/df_actual.csv')
df_training.to_csv(folder_path+'/df_training.csv')
df_validation.to_csv(folder_path+'/df_validation.csv')

# Save the model to a file
model.save(folder_path+'/tf_model')

