# notebooks — training & demo archive (verbatim, not re-run)

Copied from `test/sample/` + `test/Final/`:

* `LSTM/LSTM.ipynb` + `demo.py` + `plotly_demo.py`
* `RNN/RNN.ipynb` + `demo.py`
* `BI-LSTM/BI-LSTM.ipynb` + `demo.py`
* `Final/demo_full.py` (was `test/Final/demo.py`) + `app_variant.py` (was `test/Final/app.py`)

Training setup from `docs/mini-report.docx` Appendix-I:
AAPL `2010-01-01`–`2020-01-01`, `Open` column, MinMax(0,1), 70/30 split,
window 100, RNN/LSTM/Bi-LSTM architectures as in main README, 50 epochs,
saved as `*_model.h5`. App inference uses `Close` instead — see main README.
