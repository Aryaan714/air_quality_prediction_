# 🌬️ Air Quality Prediction: Classical vs. Deep Learning (Project #25)
### CS 470: Machine Learning | Course Project Fall 2025

**Presented By:**
* Syed Mukarrum Ali
* Muhammad Aryaan Kasi

---

## 1. Abstract
This project conducts a comparative analysis between Classical Machine Learning ensembles ($Random Forest$, $XGBoost$) and Deep Learning architectures ($Stacked LSTM$) for time-series forecasting. Using a localized dataset of ~15,000 hourly records, we developed a system to predict hourly $PM2.5$ concentrations up to 30 days into the future. Our findings demonstrate that while $XGBoost$ offers high computational efficiency, the **Stacked LSTM model** achieves superior predictive accuracy (lowest $MAE$ of ~15.89) by effectively capturing long-term temporal dependencies.

---

## 2. Introduction
**Problem Statement:**
Air pollution is a non-linear system where current states are highly dependent on historical context. Traditional models often treat data points as independent events, failing to capture the sequential "memory" required to predict abrupt smog events.

**Project Objectives:**
1. **Data Engineering:** Automate cleaning and feature selection using Gini Importance.
2. **Comparative Modeling:** Optimize $Random Forest$, $XGBoost$, and $Stacked LSTM$.
3. **Rigorous Validation:** Implement a **60/15/10 chronological split** to ensure the model generalizes to future unseen data.

---

## 3. Dataset & Preprocessing
* **Source:** UCI Machine Learning Repository (Pollution Dataset).
* **Scope:** ~15,177 hourly sensor readings.
* **Pipeline:**
    * **Interpolation:** Missing sensor values were filled using **Linear Interpolation**.
    * **Feature Selection:** We identified the Top 4 predictors: $DEWP$ (Dew Point), $TEMP$ (Temperature), $PRES$ (Pressure), and $HUMI$ (Humidity).
    * **Normalization:** Applied `MinMaxScaler` to bring all values into the range $[0, 1]$, preventing gradient instability in the neural network.
    * **Windowing:** Used a **24-hour sliding window** to predict the subsequent hour.

---

## 4. Model Architectures & Neural Networks

### 4.1 Classical Ensembles
* **Random Forest:** A bagging ensemble that averages predictions from 100 independent decision trees to reduce noise and variance.
* **XGBoost:** A gradient boosting framework that sequentially corrects the errors of previous trees, optimized for tabular accuracy.

### 4.2 Stacked LSTM (Deep Learning)
We implemented a **Stacked Long Short-Term Memory (LSTM)** network. Unlike standard networks, LSTMs use "gates" (Forget, Input, Output) to maintain an internal **Cell State**. This allows the model to "remember" weather patterns from 24 hours ago that directly influence current pollution spikes.

* **Architecture:** * Layer 1: 64 LSTM Units (Sequence Return)
    * Layer 2: 32 LSTM Units
    * Hidden: 16 Dense Units (ReLU)
    * Output: 1 Dense Unit (Linear)

---

## 5. Hyperparameters & Technical Settings

To ensure robust performance and prevent overfitting, the following hyperparameters were utilized:

| Parameter | Value | Logic |
| :--- | :--- | :--- |
| **Optimizer** | Adam | Efficient adaptive learning rate. |
| **Regularization**| Dropout (0.2) | Randomly disables 20% of neurons to ensure generalization. |
| **Learning Rate** | 0.001 | Standard starting rate for $PM2.5$ regression. |
| **Batch Size** | 128 | Balanced gradient stability and training speed. |
| **Early Stopping**| Patience: 3 | Terminates training if validation loss plateaus for 3 epochs. |

---

## 6. Results & Visual Analysis

### 6.1 Performance Metrics
The models were evaluated on the final 10% hold-out test set:

| Model | $RMSE$ | $MAE$ | $R^2$ Score |
| :--- | :--- | :--- | :--- |
| Random Forest | 0.03632 | 0.02511 | 0.8124 |
| XGBoost | 0.03391 | 0.02245 | 0.8456 |
| **LSTM** | **0.03082** | **0.01589** | **0.8912** |
<img width="1375" height="457" alt="image" src="https://github.com/user-attachments/assets/ea678ecf-2dc6-4b07-9925-95d06fd43d25" />
<img width="713" height="478" alt="image" src="https://github.com/user-attachments/assets/5cf4e96b-9ce9-4896-89c7-9c937f2cc6f7" />
<img width="1338" height="610" alt="image" src="https://github.com/user-attachments/assets/a3eb37fb-7684-4a8d-9302-ad6b88948196" />



### 6.2 Key Visualizations
* **Learning Curves:** The LSTM reached convergence within 15 epochs. Training and validation loss curves remained aligned, indicating the regularization (Dropout) successfully prevented overfitting.
* **Confusion Matrix:** By categorizing predictions into **Safe**, **Unhealthy**, and **Hazardous**, we confirmed the LSTM correctly identified over 1,150 "Safe" periods and effectively flagged transitions into unhealthy states.
* **30-Day Forecast:** The "Crystal Ball" simulation demonstrated the LSTM's ability to maintain atmospheric "rhythm" over 720 hours, whereas classical models tended to revert to a simple mathematical mean.

---

## 7. Conclusion
This project demonstrates that while $XGBoost$ is a powerful tool for quick tabular analysis, **Deep Learning (LSTM)** is essential for time-series tasks where historical context is key. The LSTM's internal memory cell makes it the superior choice for building "Early Warning Systems" for urban air quality.

---

## 8. Tech Stack
* **Frameworks:** TensorFlow/Keras (LSTM), XGBoost, Scikit-Learn.
* **Language:** Python 3.x.
* **Visualization:** Matplotlib, Seaborn.
