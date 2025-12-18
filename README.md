# Air Quality Prediction: Classical vs. Deep Learning (Project #25)
### CS 470: Machine Learning | Course Project Fall 2025

**Presented By:**
* Syed Mukarrum Ali
* Muhammad Aryaan Kasi

---

## 1. Abstract
* **Overview:** This project conducts a rigorous comparative analysis between Classical Machine Learning ensembles (Random Forest, XGBoost) and Deep Learning architectures (Long Short-Term Memory Networks) for the task of time-series forecasting.
* **Goal:** To predict hourly **PM2.5** (Particulate Matter < 2.5µm) concentrations using a localized dataset of ~15,000 records, forecasting up to 30 days into the future.
* **Key Findings:** Our experiments demonstrate that while Gradient Boosting (XGBoost) offers high computational efficiency, the **LSTM Deep Learning model** achieves superior predictive accuracy (lowest MAE of ~15.89) by effectively capturing long-term temporal dependencies and seasonal weather patterns.
* **Significance:** Accurate forecasting of PM2.5 is crucial for developing "Early Warning Systems" that allow governments to mitigate smog-related health crises before they occur.

---

## 2. Introduction
**Problem Statement:**
Air pollution is a dynamic, non-linear system influenced by complex interactions between meteorological factors (humidity, pressure, temperature). Traditional statistical models often fail to capture the abrupt spikes in pollution (smog events) because they do not account for the sequential "memory" of the atmosphere.

**Project Objectives:**
1. **Data Engineering:** To construct a robust pipeline that handles missing values via interpolation, normalizes diverse sensor data, and selects the most relevant features using mathematical importance scores.
2. **Model Implementation:** To implement and optimize three distinct classes of algorithms:
    * *Bagging Ensemble:* Random Forest
    * *Boosting Ensemble:* XGBoost
    * *Recurrent Neural Network:* Stacked LSTM
3. **Evaluation:** To validate model performance using a **60/15/10 Data Split** (Train/Validation/Test) to ensure the models are tested on entirely unseen future data.

---

## 3. Dataset Description
* **Source:** UCI Machine Learning Repository (Pollution Dataset).
* **Scope:** Hourly sensor readings (~15,177 records) including PM2.5 concentrations, Dew Point, Temperature, and Pressure.

### 3.1 Preprocessing Pipeline
1. **Missing Value Imputation:**
    * *Problem:* Sensor failure leads to `NA` values.
    * *Solution:* We applied **Linear Interpolation** with a 12-hour limit. This assumes weather changes gradually, filling gaps by drawing a straight line between known data points.
2. **Feature Selection (Dimensionality Reduction):**
    * We trained a preliminary Random Forest to calculate **Gini Importance**.
    * We retained only the **Top 4 Features** ($DEWP$, $TEMP$, $PRES$, $HUMI$) that contributed most to reducing prediction error.
3. **Scaling:**
    * Applied `MinMaxScaler` to transform features to the range `[0, 1]`. This is critical for LSTM convergence to prevent "exploding gradients" during training.
4. **Sequence Generation (Sliding Window):**
    * Transformed tabular data into a 3D Time-Series format: `(Samples, Time Steps, Features)`.
    * **Window Size:** 24 Hours. The model looks at the previous 24 hours ($T_{-24} \dots T_{-1}$) to predict the next hour ($T_{0}$).

---

## 4. Code Structure & Implementation Details

### **Cell 1: Intelligent Data Loader & Splitter**
This module automates the ingestion and splitting of the CSV.
* **Leakage Prevention:** Uses `train_test_split` with `shuffle=False`. In time-series data, shuffling is prohibited as the future cannot predict the past.
* **Split Logic:**
    * **Train (60%):** Historical baseline for learning.
    * **Validation (15%):** Used for real-time tuning and Early Stopping.
    * **Test (10%):** Final "Exam" on unseen data to report final accuracy.

### **Cell 2: Hyperparameter Tuning & Training**
* **Classical Models:** Random Forest and XGBoost are trained on flattened data to establish baseline regression metrics.
* **Deep Learning (LSTM):** * Implemented a **Stacked LSTM** architecture (64 units → 32 units).
    * Includes **Dropout (0.2)** layers after each LSTM block to prevent overfitting.
    * Uses an **EarlyStopping** callback to stop training automatically if the validation loss plateaus for 3 consecutive epochs.

### **Cell 3: Evaluation Suite**
Generates performance metrics and visualizes the **Confusion Matrix**.
* **Metrics:** RMSE, MAE, and R² Score comparison across all three models.
* **Classification:** Since this is regression, we "binned" results into categories (**Safe, Unhealthy, Hazardous**) to create a Confusion Matrix, assessing how well the model predicts health-critical events.

### **Cell 4: The "Crystal Ball" (Recursive Forecasting)**
Implements a feedback loop for long-term prediction.
1. Model predicts hour $T+1$.
2. This prediction is appended to the input window.
3. The window "slides" forward, and the model uses its own prediction to forecast $T+2$.
4. Repeated for 720 hours (30 Days).

---

## 5. Theoretical Framework

### 5.1 Ensemble Methods (RF & XGBoost)
* **Random Forest:** A **Bagging** ensemble that averages predictions from multiple independent decision trees to reduce noise.
* **XGBoost:** A **Boosting** ensemble that builds trees sequentially, with each new tree correcting the residual errors of the previous ones.

### 5.2 Stacked LSTM (The "Memory" Model)

* **Theory:** A specialized Recurrent Neural Network (RNN) designed for long-range dependencies.
* **Mechanism:** It uses a "Cell State" and three gates (Forget, Input, Output) to decide what information from the past (e.g., a pressure drop 12 hours ago) should influence the current prediction.

---

## 6. Experimental Settings

| Model | Parameter | Value | Description |
| :--- | :--- | :--- | :--- |
| **Random Forest** | `n_estimators` | 100 | Total decision trees. |
| **XGBoost** | `learning_rate` | 0.05 | Weight of each correction tree. |
| **LSTM** | `Neurons` | 64, 32 | Capacity of temporal memory. |
| **LSTM** | `Dropout` | 0.2 | Probability of disabling neurons. |

---

## 7. Results & Analysis

### 7.1 Performance Metrics
| Model | RMSE | MAE | R² Score |
| :--- | :--- | :--- | :--- |
| **Random Forest** | 0.03632 | 0.02511 | 0.8124 |
| **XGBoost** | 0.03391 | 0.02245 | 0.8456 |
| **LSTM** | **0.03082** | **0.01589** | **0.8912** |

### 7.2 Interpretation
* **Loss Convergence:** The LSTM successfully converged within 15 epochs. The training and validation loss stayed closely aligned, proving the Dropout layers effectively prevented overfitting.
* **Classification Accuracy:** The Confusion Matrix confirmed that the LSTM is highly reliable at identifying "Safe" air quality periods, with significant success in flagging "Unhealthy" shifts before they occur.

---

## 8. Conclusion
We successfully demonstrated that while Classical ML models like XGBoost are powerful and efficient, **Deep Learning (LSTM)** is the superior choice for Air Quality Forecasting due to its ability to model the delayed effects of weather systems. 

---

## 9. References
* **Data Source:** UCI Machine Learning Repository, "PM2.5 Data".
* **Frameworks:** TensorFlow, Keras, Scikit-Learn.
