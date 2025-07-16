import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import SimpleRNN,Dense
df=pd.read_csv('C:/Users/nppra/PycharmProjects/PythonProject7/1_Daily_minimum_temps.csv')
df['Temp']=pd.to_numeric(df['Temp'],errors='coerce')
df=df.dropna(subset=['Temp'])
print(df['Temp'].head())
series=df['Temp'].values
scaler=MinMaxScaler()
series=scaler.fit_transform(series.reshape(-1,1)).flatten()
def sequences(data,seq_len):
    X,Y=[],[]
    for i in range (len(data)-seq_len):
        X.append(data[i:i+seq_len])
        Y.append(data[i+seq_len])
    return np.array(X),np.array(Y)
seq_len=50
X,Y=sequences(series,seq_len)
X=X.reshape((X.shape[0],X.shape[1],1))
model=Sequential(
    [
        SimpleRNN(seq_len,activation='relu',input_shape=(seq_len,1)),
        Dense(1)
    ]
)
model.compile(optimizer='adam',loss='mse')
model.summary()
model.fit(X,Y,epochs=30,batch_size=35)
predicted=model.predict(X)
plt.figure(figsize=(12,6))
plt.plot(Y,label="Actual")
plt.plot(predicted,label="Prediction")
plt.legend()
plt.title("RNN Prediction")
plt.show()