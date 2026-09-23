import streamlit as st
import pandas
from pandas_datareader import data as pdr
import yfinance as yfin
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from datetime import date
from datetime import datetime
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential, load_model
from plotly import graph_objs as go
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load Model

yfin.pdr_override()

start = "2013-01-01"
end = "2023-03-04"

with st.sidebar:
    option = st.selectbox(
    'Navigator',
    ('Home',"Analyse",'LSTM',"BI-LSTM","RNN"))

if option == "Home":
    st.subheader("Stock Market Prediction")
    st.write('''Stock market prediction is the act of trying to determine the future value of a company stock or other financial instrument traded on an exchange. 
    The successful prediction of a stock's future price could yield significant profit.''')
    st.subheader("What is the use of stock prediction?")
    st.write('''Stock Price Prediction using machine learning helps you discover the future value of company stock and other financial assets traded on an exchange. 
    The entire idea of predicting stock prices is to gain significant profits. 
    Predicting how the stock market will perform is a hard task to do''')


elif option == "BI-LSTM":
    model = load_model("bi-lstm_model.h5")
elif option == "LSTM":
    model = load_model("LSTM_model.h5")
elif option == "RNN":
    model = load_model("RNN_model.h5")

st.title("Stock Trend Prediction BI-LSTM")
user_input = st.text_input("Enter Stock Ticker", "AAPL")
df = pdr.get_data_yahoo(user_input, start, end)

n_days = st.slider('Forecast Days', 0, 45, 15) 
submit = st.button("Submit")

# ---------------------------------

ds = df['Close']

normalizer = MinMaxScaler(feature_range=(0,1))
ds_scaled = normalizer.fit_transform(np.array(ds).reshape(-1,1))

train_size = int(len(ds_scaled)*0.70)
test_size = len(ds_scaled) - train_size

ds_train, ds_test = ds_scaled[0:train_size,:], ds_scaled[train_size:len(ds_scaled),:1]

def create_ds(dataset,step):
    Xtrain, Ytrain = [], []
    for i in range(len(dataset)-step-1):
        a = dataset[i:(i+step), 0]
        Xtrain.append(a)
        Ytrain.append(dataset[i + step, 0])
    return np.array(Xtrain), np.array(Ytrain)

time_stamp = 100
X_train, y_train = create_ds(ds_train,time_stamp)
X_test, y_test = create_ds(ds_test,time_stamp)

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
y_predicted = model.predict(X_test)

train_predict = normalizer.inverse_transform(train_predict)
test_predict = normalizer.inverse_transform(test_predict)

if submit:
    # fig1 = plt.figure(figsize = (12,6))
    # plt.plot(normalizer.inverse_transform(ds_scaled),'r')
    # plt.plot(train_predict,'g')
    # plt.plot(test_predict,'b')
    # st.pyplot(fig1)

    test = np.vstack((train_predict,test_predict))

    st.subheader("Scaling the data")
    fig2 = plt.figure(figsize = (12,6))
    plt.plot(normalizer.inverse_transform(ds_scaled))
    plt.plot(test)
    st.pyplot(fig2)

    # ---------------------------------

    # Final Graph
    st.subheader("Predictions vs Original")
    fig2 = plt.figure(figsize = (12,6))
    plt.plot(y_test, 'b', label = 'Original Price')
    plt.plot(y_predicted, 'r', label = 'Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)


    # # --------------------------------

    fut_inp = ds_test[len(ds_test)-100:]
    fut_inp = fut_inp.reshape(1,-1)
    tmp_inp = list(fut_inp)
    tmp_inp = tmp_inp[0].tolist()

    lst_output=[]
    n_steps=100
    i=0

    while(i < n_days):
        
        if(len(tmp_inp)>100):
            fut_inp = np.array(tmp_inp[1:])
            fut_inp=fut_inp.reshape(1,-1)
            fut_inp = fut_inp.reshape((1, n_steps, 1))
            yhat = model.predict(fut_inp, verbose=0)
            tmp_inp.extend(yhat[0].tolist())
            tmp_inp = tmp_inp[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            fut_inp = fut_inp.reshape((1, n_steps,1))
            yhat = model.predict(fut_inp, verbose=0)
            tmp_inp.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1

    fig2 = plt.figure(figsize = (12,6))
    plot_new=np.arange(1,101)
    plot_pred=np.arange(101,101+n_days)
    plt.plot(plot_new, normalizer.inverse_transform(ds_scaled[len(ds_scaled)-100:]))
    plt.plot(plot_pred, normalizer.inverse_transform(lst_output))
    st.pyplot(fig2)

    ds_new = ds_scaled.tolist()
    ds_new.extend(lst_output)

    final_graph = normalizer.inverse_transform(ds_new).tolist()

    st.subheader(f"Prediction of the next {n_days} days")
    fig2 = plt.figure(figsize = (12,6))
    plt.plot(final_graph,)
    plt.ylabel("Price")
    plt.xlabel("Time")
    plt.title(f"Prediction of Next {n_days} days")
    plt.axhline(y=final_graph[len(final_graph)-1], color = 'red', linestyle = ':', label = 'NEXT {0} DAY : {1}'.format(n_days,round(float(*final_graph[len(final_graph)-1]),2)))
    plt.legend()
    st.pyplot(fig2)

# -------------------------------------------------------------------------

