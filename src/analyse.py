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

yfin.pdr_override()


st.title("Stock Chart")
user_input = st.text_input("Enter Stock Ticker", "AAPL")

start = st.date_input("Enter Start Date", datetime.date(2023, 1, 1))
end = st.date_input("Enter End Date")

df = pdr.get_data_yahoo(user_input, start, end)

df_time = df.reset_index()
data = pd.DataFrame(df_time)

submit = st.button("Submit")

if submit:

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, specs=[[{"type": "candlestick"}], [{"type": "bar"}]])
    fig.add_trace(go.Candlestick(x=data['Date'], open=data['Open'], high=data['High'],
                                low=data['Low'], close=data['Close'], name='AAPL'), row=1, col=1)
    fig.add_trace(go.Bar(x=data["Date"],y=data["Volume"],name="Volume"),row=2,col=1)
    fig.layout.update(title_text=f'Analysing Price and Volume of {user_input}', xaxis_rangeslider_visible=True)
    fig.update_layout(
        autosize=False,
        width=800,
        height=800)
    st.plotly_chart(fig)