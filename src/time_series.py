import streamlit as st
import pandas
from pandas_datareader import data as pdr
import yfinance as yfin
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from prophet import Prophet
from prophet.plot import plot_plotly

start = "2013-01-01"
end = "2023-03-04"

# startDate = "2013-01-01"
# todayDate = datetime.date.today().strftime("%Y-%m-%d")

st.title("Time Series Forecasting using FBProphet")
user_input = st.text_input("Enter Stock Ticker", "AAPL")

years = st.slider("Number of Future Years to Forcast", 1, 5)
period = years * 365

yfin.pdr_override()
df = pdr.get_data_yahoo(user_input, start, end)
df_time = df.reset_index()
data = pd.DataFrame(df_time)
DFTrain = data[['Date', 'Close']]
DFTrain = DFTrain.rename(columns = {"Date": "ds", "Close": "y"})

submit = st.button("Submit")

if submit :
    
    model = Prophet()
    model.fit(DFTrain)
    future = model.make_future_dataframe(periods = period)
    forecast = model.predict(future)

    st.subheader('Forecasted Prices')
    forecastFig = model.plot(forecast)
    ax = forecastFig.gca() 
    ax.set_xlabel("Time", size=10)
    ax.set_ylabel("Stock Price ($)", size=10)
    ax.set_facecolor((0.08, 0.08, 0.1))
    forecastFig.patch.set_facecolor((0.13, 0.18, 0.25))
    ax.xaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(axis='y', colors='white')
    st.pyplot(forecastFig)