# models — pretrained weights (verbatim, not retrained)

* `LSTM.h5` (664 KB), `BI-LSTM.h5` (3.6 MB), `RNN.h5` (280 KB)
* Source: `Testing deployment full stack/*.h5`
* `app.py` loads via `load_model(f"{choice}.h5")` where choice ∈ {LSTM, BI-LSTM, RNN},
  so runnable copies are also kept in repo root alongside `app.py`.
  `models/` holds the canonical archive copies (identical bytes).
