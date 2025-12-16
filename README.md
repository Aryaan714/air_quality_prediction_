# air_quality_prediction_
# Air Quality Prediction (PM2.5 Forecasting) 

**Course:** CS 470: Machine Learning

**Presented by:** Syed Mukarrum Ali and Muhammad Aryaan Kasi

## 1. Abstract
We developed a time-series forecasting system to predict PM2.5 air pollution levels in Beijing. We compared a classical Random Forest Regressor against a Deep Learning LSTM model. Our results show that [Insert which model was better] achieved the lowest RMSE of [Insert Score].

## 2. Methodology
### Data Preprocessing
* **Source:** Beijing PM2.5 Data (UCI Repository).
* **Cleaning:** Handled missing values via Linear Interpolation.
* **Feature Engineering:** Created 24 "Lag" features (using past 24h to predict next hour).
* **Filtering:** Removed non-numeric text columns (e.g., Wind Direction) to ensure compatibility with regression models.

### Models
1.  **Classical ML:** Random Forest Regressor (n_estimators=100, max_depth=10).
2.  **Deep Learning:** Long Short-Term Memory (LSTM) Network with PyTorch (Hidden Dim=64, Adam Optimizer).

## 3. Results Comparison

| Model | RMSE (Error) | Notes |
| :--- | :--- | :--- |
| Random Forest | [INSERT YOUR RF_RMSE HERE] | Faster training, good baseline. |
| LSTM (Deep Learning) | [INSERT YOUR LSTM_RMSE HERE] | Captures sequential trends better. |

## 4. Conclusion
[Write one sentence about which model you would recommend for real-world use.]
