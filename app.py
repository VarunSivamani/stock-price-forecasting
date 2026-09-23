"""Stock Price Prediction and Forecasting — Streamlit app.

Pages: Home | Analyse | Predict (LSTM / BI-LSTM / RNN) | FBPROPHET
Original logic preserved from `Testing deployment full stack/final.py`
(verbatim Mini Project-II, SKCET 2023). Refactored only for structure:
models live in `models/`, no duplicates in repo root.
"""

from pathlib import Path

import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pandas_datareader import data as pdr
import yfinance as yfin
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import streamlit as st


# ---------------------------------------------------------------- constants
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
ALGO_TO_FILE = {
    "LSTM": "LSTM.h5",
    "BI-LSTM": "BI-LSTM.h5",
    "RNN": "RNN.h5",
}

TIME_STEP = 100
TRAIN_SPLIT = 0.70
PREDICT_START = "2013-01-01"
PREDICT_END = "2023-03-04"

st.set_page_config(page_title="Stock Price Prediction", layout="wide")
yfin.pdr_override()


# ---------------------------------------------------------------- helpers
@st.cache_resource(show_spinner=False)
def get_keras_model(algo: str):
    """Load a pretrained .h5 from models/ (cached across reruns)."""
    path = MODEL_DIR / ALGO_TO_FILE[algo]
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Expected pretrained weights in models/ "
            f"({', '.join(ALGO_TO_FILE.values())})."
        )
    return load_model(str(path))


def fetch_yahoo(ticker: str, start, end) -> pd.DataFrame:
    """Fetch OHLCV data; raises a friendly error on empty frames."""
    df = pdr.get_data_yahoo(ticker, start, end)
    if df is None or df.empty:
        raise ValueError(f"No data for ticker '{ticker}' in the given range.")
    return df


def create_sequences(dataset: np.ndarray, step: int):
    """Sliding-window sequences: X = past `step` points, y = next point."""
    xs, ys = [], []
    for i in range(len(dataset) - step - 1):
        xs.append(dataset[i:(i + step), 0])
        ys.append(dataset[i + step, 0])
    return np.array(xs), np.array(ys)


def forecast_future(model, tail_scaled: np.ndarray, n_days: int,
                    n_steps: int = TIME_STEP):
    """Autoregressive rollout: feed each prediction back as input."""
    tmp = tail_scaled.reshape(1, -1)[0].tolist()
    out = []
    for _ in range(n_days):
        window = np.array(tmp[1:] if len(tmp) > n_steps else tmp).reshape(1, -1)
        window = window.reshape((1, n_steps, 1))
        yhat = model.predict(window, verbose=0)
        tmp.extend(yhat[0].tolist())
        if len(tmp) > n_steps:
            tmp = tmp[1:]
        out.extend(yhat.tolist())
    return out


# ---------------------------------------------------------------- pages
def show_home():
    st.header("Stock Price Prediction and Forecasting Using Machine Learning")
    st.markdown("---")
    st.write("- Stock Price Prediction using machine learning helps you discover the future value of company stock and other financial assets traded on an exchange.")
    st.write("- The entire idea of predicting stock prices is to gain significant profits. Predicting how the stock market will perform is a hard task to do.")
    st.write("- There are other factors involved in the prediction, such as physical and psychological factors, rational and irrational behavior, and so on.")
    st.write("- All these factors combine to make share prices dynamic and volatile.")
    st.write("- This makes it very difficult to predict stock prices with high accuracy.")


def show_analyse():
    st.title("Stock Chart")
    ticker = st.text_input("Enter Stock Ticker", "AAPL")
    start = st.date_input("Enter Start Date", datetime.date(2023, 1, 1))
    end = st.date_input("Enter End Date")
    if st.button("Submit"):
        df = fetch_yahoo(ticker, start, end)
        data = pd.DataFrame(df.reset_index())
        st.subheader("Stock Summary")
        st.table(data.describe())
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            specs=[[{"type": "candlestick"}], [{"type": "bar"}]])
        fig.add_trace(go.Candlestick(x=data["Date"], open=data["Open"],
                                     high=data["High"], low=data["Low"],
                                     close=data["Close"], name=ticker),
                      row=1, col=1)
        fig.add_trace(go.Bar(x=data["Date"], y=data["Volume"], name="Volume"),
                      row=2, col=1)
        fig.layout.update(title_text=f"Analysing Price and Volume of {ticker}",
                          xaxis_rangeslider_visible=True)
        fig.update_layout(autosize=False, width=800, height=800)
        st.plotly_chart(fig, use_container_width=True)


def show_prophet():
    st.title("Time Series Forecasting using FBProphet")
    start = st.date_input("Enter Start Date", datetime.date(2020, 1, 1))
    end = st.date_input("Enter End Date")
    ticker = st.text_input("Enter Stock Ticker", "AAPL")
    years = st.slider("Number of Future Years to Forecast", 1, 5)
    period = years * 365
    if st.button("Submit"):
        df = fetch_yahoo(ticker, start, end)
        data = pd.DataFrame(df.reset_index())
        train = data[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
        model = Prophet()
        model.fit(train)
        forecast = model.predict(model.make_future_dataframe(periods=period))
        st.subheader("Forecasted Prices")
        fig = model.plot(forecast)
        ax = fig.gca()
        ax.set_xlabel("Time", size=10)
        ax.set_ylabel("Stock Price ($)", size=10)
        ax.set_facecolor((0.08, 0.08, 0.1))
        fig.patch.set_facecolor((0.13, 0.18, 0.25))
        ax.xaxis.label.set_color("white")
        ax.tick_params(axis="x", colors="white")
        ax.yaxis.label.set_color("white")
        ax.tick_params(axis="y", colors="white")
        st.pyplot(fig)


def show_predict():
    st.title("Stock Trend Forecasting")
    algo = st.selectbox("Select the algorithm", tuple(ALGO_TO_FILE))
    model = get_keras_model(algo)
    ticker = st.text_input("Enter Stock Ticker", "AAPL")
    n_days = st.slider("Forecast Days", 0, 45, 15)
    if not st.button("Submit"):
        return

    df = fetch_yahoo(ticker, PREDICT_START, PREDICT_END)
    series = df["Close"]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(np.array(series).reshape(-1, 1))

    split = int(len(scaled) * TRAIN_SPLIT)
    train, test = scaled[:split], scaled[split:]

    x_train, _ = create_sequences(train, TIME_STEP)
    x_test, y_test = create_sequences(test, TIME_STEP)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    train_pred = scaler.inverse_transform(model.predict(x_train))
    test_pred = scaler.inverse_transform(model.predict(x_test))
    y_hat = model.predict(x_test)

    st.subheader("Scaling the data")
    fig = plt.figure(figsize=(12, 6))
    plt.plot(scaler.inverse_transform(scaled))
    plt.plot(np.vstack((train_pred, test_pred)))
    st.pyplot(fig)

    st.subheader("Predictions vs Original")
    fig = plt.figure(figsize=(12, 6))
    plt.plot(y_test, "b", label="Original Price")
    plt.plot(y_hat, "r", label="Predicted Price")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(fig)

    future = forecast_future(model, test[len(test) - TIME_STEP:], n_days)
    if n_days:
        fig = plt.figure(figsize=(12, 6))
        plt.plot(np.arange(1, TIME_STEP + 1),
                 scaler.inverse_transform(scaled[len(scaled) - TIME_STEP:]))
        plt.plot(np.arange(TIME_STEP + 1, TIME_STEP + 1 + n_days),
                 scaler.inverse_transform(future))
        st.pyplot(fig)

    extended = scaled.tolist()
    extended.extend(future)
    full = scaler.inverse_transform(extended).tolist()

    fig = plt.figure(figsize=(12, 6))
    plt.plot(full)
    plt.ylabel("Price")
    plt.xlabel("Time")
    plt.title(f"Prediction of Next {n_days} days")
    plt.axhline(y=full[-1], color="red", linestyle=":",
                label="NEXT {0} DAY : {1}".format(
                    n_days, round(float(*full[-1]), 2)))
    plt.legend()
    st.pyplot(fig)

    if n_days:
        tail = [p for sub in full[len(full) - n_days:] for p in sub]
        df_future = pd.DataFrame({"Date": list(range(n_days)), "Price": tail})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_future["Date"], y=df_future["Price"],
                                 name="forecast"))
        fig.layout.update(title_text="Time Series data with Rangeslider",
                          xaxis_rangeslider_visible=True)
        st.plotly_chart(fig, use_container_width=True)

    flat = [p for sub in full for p in sub]
    df_all = pd.DataFrame({"Date": list(range(len(extended))), "Price": flat})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_all["Date"], y=df_all["Price"],
                             name="forecast", marker=dict(color="blue")))
    fig.layout.update(title_text="Time Series data with Rangeslider",
                      xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------- entry
with st.sidebar:
    page = st.selectbox("Select the page",
                        ("Home", "Analyse", "Predict", "FBPROPHET"))

{"Home": show_home, "Analyse": show_analyse,
 "FBPROPHET": show_prophet, "Predict": show_predict}[page]()
