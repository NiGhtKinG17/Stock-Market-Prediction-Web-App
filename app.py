from cProfile import label
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import datetime

st.title('Stock Market Prediction')

stocks = st.text_input('Enter Stock Ticker', 'AAPL')

start = st.date_input('Enter the starting date', datetime.date(2011,1,1))
end = st.date_input('Enter the end date', datetime.date(2021,12,31))
# start = '2011-01-01'
# end = '2021-12-31'

df = data.DataReader(stocks, 'yahoo', start, end)

#Data describing

st.subheader(f'Data from {start} to {end}')
st.write(df.describe())

#data visualizations
st.subheader('Closing price vs Time Plot')
fig = plt.figure(figsize=(12,8))
plt.plot(df.Close)
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

# 100MA Graph
st.subheader('100 Mean Avg Graph')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,8))
plt.plot(df.Close, label='Closing Price')
plt.plot(ma100, label='MA100')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

# 200MA Graph
st.subheader('200 Mean Avg Graph')
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,8))
plt.plot(df.Close, label='Closing Price')
plt.plot(ma100, label='MA100')
plt.plot(ma200, label='MA200')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

data_train = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
data_test = pd.DataFrame(df['Close'][int(len(df)*0.7):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_train_scaled = scaler.fit_transform(data_train)

model = load_model('LSTM_model.h5')

prev_100_days = data_train.tail(100)

test_df = prev_100_days.append(data_test, ignore_index=True)

test_data_scaled = scaler.fit_transform(test_df)

x_test = []
y_test = []

for i in range(100, test_data_scaled.shape[0]):
    x_test.append(test_data_scaled[i-100:i])
    y_test.append(test_data_scaled[i,0])
    
x_test, y_test = np.array(x_test), np.array(y_test)

y_predictions = model.predict(x_test)
scale = scaler.scale_
scf = 1/scale[0]
y_predictions = y_predictions*scf
y_test = y_test*scf

st.subheader('Original vs Predictions plot')
fig = plt.figure(figsize=(12,8))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predictions, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)