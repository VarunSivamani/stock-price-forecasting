import streamlit as st
import pandas
from pandas_datareader import data as pdr
import yfinance as yfin
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

yfin.pdr_override()

start = "2023-01-01"
end = "2023-04-01"

st.title("Stock Trend Prediction")
user_input = st.text_input("Enter Stock Ticker", "AAPL")
df = pdr.get_data_yahoo(user_input, start, end)

df_time = df.reset_index()
data = pd.DataFrame(df_time)

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, specs=[[{"type": "candlestick"}], [{"type": "bar"}]])
# fig.add_trace(go.Scatter(x=data["Date"],y=data["Close"]),row=1,col=1)
fig.add_trace(go.Candlestick(x=data['Date'], open=data['Open'], high=data['High'],
                             low=data['Low'], close=data['Close'], name='AAPL'), row=1, col=1)
fig.add_trace(go.Bar(x=data["Date"],y=data["Volume"],name="Volume"),row=2,col=1)
fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
fig.update_layout(
    autosize=False,
    width=800,
    height=800)
st.plotly_chart(fig)

# ----------------------------------------

# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import pandas as pd

# # Load sample data
# df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')

# # Create subplots with shared x-axis and rangeslider
# fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
#                     row_heights=[0.7, 0.3], specs=[[{"type": "candlestick"}], [{"type": "bar"}]])
# fig.add_trace(go.Candlestick(x=df['Date'], open=df['AAPL.Open'], high=df['AAPL.High'],
#                              low=df['AAPL.Low'], close=df['AAPL.Close'], name='AAPL'), row=1, col=1)
# fig.add_trace(go.Bar(x=df['Date'], y=df['AAPL.Volume'], name='AAPL Volume'), row=2, col=1)

# # Update layout to move range slider to the end of the subplots
# fig.update_layout(xaxis=dict(domain=[0, 1], rangeslider=dict(visible=True, thickness=0.1, bgcolor='white'),
#                              tickfont=dict(size=10), tickangle=0, showgrid=True, gridwidth=1, gridcolor='gray'),
#                   yaxis=dict(domain=[0.3, 1]), yaxis2=dict(domain=[0, 0.2]), height=600,
#                   margin=dict(l=50, r=50, t=10, b=50, pad=0), hovermode='x')

# # Show figure
# st.plotly_chart(fig)
