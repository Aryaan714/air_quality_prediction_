#  Air Quality Forecasting: Classical ML vs. Deep Learning
### CS 470: Machine Learning | Course Project Fall 2025

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

##  Project Overview
This project focuses on the **comparative analysis of Classical Machine Learning algorithms versus Deep Learning architectures** for Time Series Forecasting. Specifically, we aim to predict **PM2.5 Air Quality Indices** for major Chinese cities (Beijing, Shanghai, Guangzhou, Chengdu, Shenyang) up to 30 days into the future.

We implement a rigorous pipeline including intelligent feature selection, hyperparameter tuning, and a 3-way data split (Train/Validation/Test) to ensure robust performance evaluation.

---

##  Course Information
* **University:** National University of Sciences and Technology (NUST), Pakistan
* **Department:** Electrical and Computer Engineering, SEECS
* **Course:** CS 470 - Machine Learning
* **Instructor:** Dr. Sajjad Hussain

---

##  Dataset
The dataset consists of hourly air quality and meteorological data from five major cities in China (2010–2015).
* **Source:** UCI Machine Learning Repository (Five Citie PM2.5 Data).
* **Target Variable:** PM2.5 concentration (ug/m^3).
* **Input Features:** Meteorological factors (Dew Point, Temperature, Pressure, Humidity, Wind Direction, etc.).

**Preprocessing Steps:**
1.  **Handling Missing Data:** Linear interpolation for continuous gaps.
2.  **Feature Selection:** Used Random Forest Regressor to calculate "Feature Importance" and selected the **Top 4** most predictive variables (reducing dimensionality).
3.  **Scaling:** MinMax Scaling (0, 1) for neural network stability.
4.  **Sequencing:** Created 24-hour lookback windows for time-series input.

---

##  Methodology & Models

We implemented three distinct models to cover both Classical and Deep Learning approaches:

### 1. Random Forest Regressor (Classical Ensemble)
* **Type:** Bagging Ensemble.
* **Why:** Handles non-linear relationships well and is robust to overfitting.
* **Tuning:** Optimized `n_estimators`, `max_depth`, and `min_samples_split` using `RandomizedSearchCV`.

### 2. XGBoost Regressor (Gradient Boosting)
* **Type:** Boosting Ensemble.
* **Why:** Known for state-of-the-art performance on tabular data and faster training speed than SVR.
* **Tuning:** Optimized `learning_rate`, `max_depth`, and `n_estimators`.

### 3. Long Short-Term Memory (LSTM) Network (Deep Learning)
* **Type:** Recurrent Neural Network (RNN).
* **Why:** Specifically designed for sequence prediction; capable of learning long-term dependencies in weather patterns.
* **Architecture:**
    * Input Layer (24h Sequence)
    * LSTM Layer (128 units, Return Sequences) + Dropout (0.3)
    * LSTM Layer (64 units) + Dropout (0.3)
    * Dense Output Layer
* **Optimization:** Adam Optimizer with ReduceLROnPlateau and EarlyStopping.

---

##  Evaluation & Results

We employed a **3-Way Split** strategy to prevent data leakage:
* **Training:** 70%
* **Validation:** 15% (Used for Tuning)
* **Testing:** 15% (Used for Final Evaluation)

### Performance Metrics
| Model | RMSE (Root Mean Squared Error) | MAE (Mean Absolute Error) | R² Score |
| :--- | :--- | :--- | :--- |
| **Random Forest** | *[Insert Value]* | *[Insert Value]* | *[Insert Value]* |
| **XGBoost** | *[Insert Value]* | *[Insert Value]* | *[Insert Value]* |
| **LSTM (Deep Learning)** | *[Insert Value]* | *[Insert Value]* | *[Insert Value]* |

> *Note: LSTM generally captures temporal trends better in long sequences, while XGBoost offers competitive accuracy with faster training times.*

---

##  Installation & Usage

### Prerequisites
* Python 3.8+
* Jupyter Notebook / Google Colab

### 1. Clone the Repo
```bash
git clone [https://github.com/yourusername/AirQuality-ML-Forecasting.git](https://github.com/yourusername/AirQuality-ML-Forecasting.git)
cd AirQuality-ML-Forecasting
