# Stock Price Prediction and Forecasting Using Machine Learning

> **Mini Project-II**   
> B.Tech Artificial Intelligence and Data Science   
> Sri Krishna College of Engineering and Technology, Coimbatore (Anna University)   
> April 2023

**Authors**

- Varun S (20EUAI051)
- Tharun S (20EUAI048)
- Vikass Aravind C V (20EUAI054)
- Sathish Kumar M (20EUAI034)

**Supervisor**

- Mr. G. S. Pugalendhi, Assistant Professor, AI & DS

---

### Project Description

Short-term stock-price forecasting with LSTM, Bi-LSTM, RNN, and Facebook Prophet,
served as an interactive Streamlit app over live Yahoo Finance data.

---

## Contents

- [1. Overview](#1-overview)
- [2. Results](#2-results-report-ch-53-table-51)
- [3. Repository Structure](#3-repository-structure)
- [4. Installation](#4-installation)
- [5. Usage](#5-usage)
- [6. Method](#6-method)
- [7. Testing](#7-testing)
- [8. Limitations & Future Work](#8-limitations--future-work)
- [9. References](#9-references)
- [10. License & Disclaimer](#10-license--disclaimer)

## 1. Overview

Stock markets are volatile, non-linear, and driven by economics, policy, and investor
psychology. This project tests whether deep sequence models can learn
context-specific dependence of prices on their own history and produce useful
short-horizon forecasts for entry/exit decisions.

**What is demonstrated:**

- Live OHLCV fetching (no static CSV) + MinMax scaling + 100-step sliding windows
- Three Keras sequence models (RNN / LSTM / Bi-LSTM, 50 epochs) + Prophet baseline
- A 4-page Streamlit app: Home, Analyse (candlestick + volume), Predict (deep models), FBPROPHET (multi-year trend)
- Training notebooks, pretrained `.h5` weights, report, slides, and base paper archived verbatim

## 2. Results (report Ch. 5.3, Table 5.1)

**Setup:**

| Setting     | Value            |
|-------------|------------------|
| Feature     | `Close`          |
| Scaler      | `MinMaxScaler(0,1)` |
| Split       | 70/30            |
| Timestep    | 100              |
| Optimizer / Loss | `adam` + `mse` |
| Epochs      | 50               |
| Ticker      | AAPL             |

| Model   | Architecture                                                     | Loss % |
|---------|------------------------------------------------------------------|-------:|
| RNN     | 4× SimpleRNN(50, tanh) + Dropout(0.2) + Dense(1), batch 32       | 0.285  |
| LSTM    | 3× LSTM(50) + Dense(1, linear), batch 64                         | 0.063  |
| Bi-LSTM | Bi-LSTM(128, return_seq) + Bi-LSTM(64) + Dense(1), batch 32      | 0.088  |

#### Loss curves

- Fig 5.1–5.3 in `docs/mini-report.pdf` (pp. 36-37).
- Bi-LSTM judged best for trend capture.
- Base-paper MLS-LSTM reference (Samsung 2016-2021) reports 95.9% train / 98.1% test, MAPE 2.18%, Norm. RMSE 0.019–0.028 - see `docs/base-paper-MLS-LSTM.pdf`.

## 3. Repository Structure

```text
├── app.py                  # Streamlit entrypoint (Home/Analyse/Predict/FBPROPHET)
├── src/
│   ├── analyse.py          # Standalone Analyse page (reference)
│   └── time_series.py      # Standalone Prophet page (reference)
├── models/
│   ├── LSTM.h5             # 664 KB
│   ├── BI-LSTM.h5          # 3.6 MB
│   └── RNN.h5              # 280 KB
├── notebooks/
│   ├── LSTM/               # LSTM.ipynb + demo.py + plotly_demo.py
│   ├── RNN/                # RNN.ipynb + demo.py
│   ├── BI-LSTM/            # BI-LSTM.ipynb + demo.py
│   ├── Final/              # Full-app-era demos
│   └── legacy/             # Superseded predict-only app (traceability)
├── docs/
│   ├── mini-report.docx
│   ├── mini-report.pdf
│   ├── TEAM-1.pdf
│   ├── TEAM-1.pptx
│   ├── base-paper-MLS-LSTM.pdf
│   └── snippets/rangeslider_visual.txt
├── assets/stock.jpg
├── requirements.txt
├── LICENSE (MIT)
└── .gitignore
```

> **Design rule enforced:**
>
> - All runtime weights live *only* in `models/`.
> - `app.py` resolves them via `MODEL_DIR = Path(__file__).parent / "models"`.
> - There are no `.h5` files in the repo root.

## 4. Installation

```powershell
cd repo
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> See [Prophet docs](https://facebook.github.io/prophet/docs/installation.html) if `pip install prophet` fails.

## 5. Usage

```powershell
streamlit run app.py
```

1. Sidebar → **Home**: problem framing.
2. **Analyse**: ticker (e.g. `AAPL`, `RELIANCE.NS`, `BTC-USD`) + date range → summary table + candlestick/volume.
3. **Predict**: algorithm + ticker + forecast-days (0–45) → scaling plot, predicted-vs-actual, 100-day context forecast, full-history forecast, Plotly rangeslider views.
4. **FBPROPHET**: ticker + range + horizon (1–5 yrs) → Prophet trend with uncertainty.

> Predict uses fixed training window `2013-01-01`–`2023-03-04` (as in the report);
> Analyse/Prophet use user-selected ranges.

## 6. Method

- **Collection:** `pandas_datareader.get_data_yahoo` + `yfinance.pdr_override()`, OHLCV.
- **Preprocessing:** NumPy/pandas arrays → MinMax(0,1) → 70/30 split → reshape to `[N, 100, 1]`.
- **Training:** Appendix-I (`docs/mini-report.docx`) and `notebooks/*/*.ipynb`; AAPL `2010-01-01`–`2020-01-01` (`Open` for training, `Close` for app inference).
- **Inference:** one-step test prediction + autoregressive rollout for `n_days`.
- **Evaluation:** train/val loss curves, loss % table, visual predicted-vs-actual.

## 7. Testing

Report Ch. 6:

- White-box, conditional, data-flow, loop testing.
- Manual cases 15-day and 30-day forecasts render correctly.
- No automated suite is shipped.

## 8. Limitations & Future Work

### Limitations

| # | Limitation | Details |
|---|------------|---------|
| 1 | Overfitting sensitivity | Deep sequence models (RNN / LSTM / Bi-LSTM) can overfit short, noisy price histories without careful regularization and early stopping |
| 2 | Basic null handling | Missing values are handled with simple forward-fill, which can propagate stale prices across gaps |
| 3 | Single-feature input | Models use only `Close` (app inference) / `Open` (training) - OHLCV volume, volatility, and market context are ignored |
| 4 | Static predict window | `Predict` page uses a fixed training window `2013-01-01` – `2023-03-04` (as in the report), so it does not adapt to new regimes without retraining |

### Future Work

- **Richer features:** add sentiment signals (news / social) + fundamental features and financial ratios
- **Better imputation:** median / running-average imputation instead of forward-fill
- **Architecture search:** automated search (e.g. genetic algorithms) for layers, units, lookback, and dropout
- **Hybrid models:** fuzzy-logic hybrids with deep sequence models for uncertainty handling
- **Training-time optimization:** faster training / inference tuning for live use

## 9. References

1. Md et al., MLS-LSTM, *Applied Soft Computing* 134 (2023) 109830 - `docs/base-paper-MLS-LSTM.pdf`
2. Teng et al., MLCA-LSTM, *Neurocomputing* 505 (2022)
3. Gajamannage et al., Dual-LSTMs, *Expert Syst. Appl.* 223 (2023)
4. Khodaee et al., CNN-LSTM-RESNET, *Eng. Appl. AI* 116 (2022)

Full [1]–[20] list: report p. 45 / `docs/mini-report.pdf`.

## 10. License & Disclaimer

MIT - see `LICENSE`. Academic mini-project (2023); forecasts are experimental and
**not financial advice**. Past performance does not predict future returns.
