# Streamlit E-Commerce Profit Dashboard

## 1. Files needed
Put these 3 files in the same folder:

- `app.py`
- `requirements.txt`
- `Sample - Superstore.csv`

## 2. Install packages

```bash
pip install -r requirements.txt
```

## 3. Run

```bash
streamlit run app.py
```

If `streamlit` is not recognized on Windows:

```bash
python -m streamlit run app.py
```

## 4. Dashboard structure

1. Overview
2. Discount Analysis
3. Customer Segmentation
4. Profit Simulator
5. Model Performance

The Random Forest model is retrained automatically when the app starts. This avoids
having to export a separate `.pkl` file and keeps the feature encoding identical
to the notebook logic (`pd.get_dummies(..., drop_first=True)`).

## 5. Important methodological note

The simulator is intentionally presented as a **Profit Prediction / Scenario Analysis**
tool rather than an "optimal discount calculator". Sales is an input feature in the
current Random Forest model, so the simulator should not be interpreted as a causal
estimate of what changing discount will do to profit before a transaction occurs.
