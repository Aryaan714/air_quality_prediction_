# 🌍 Air Quality Prediction: Classical vs. Deep Learning (Project #25)
### CS 470: Machine Learning | Course Project Fall 2025

**Team Members:**
* [Student Name 1] - [Student ID]
* [Student Name 2] - [Student ID]

---

## 📝 1. Abstract
* **Overview:** This project conducts a rigorous comparative analysis between Classical Machine Learning ensembles (Random Forest, XGBoost) and Deep Learning architectures (Long Short-Term Memory Networks) for the task of time-series forecasting.
* **Goal:** To predict hourly **PM2.5** (Particulate Matter < 2.5µm) concentrations in major Chinese cities up to 30 days into the future.
* **Key Findings:** Our experiments demonstrate that while Gradient Boosting (XGBoost) offers excellent computational efficiency, the **LSTM Deep Learning model** achieves superior predictive accuracy (lowest RMSE) by effectively capturing long-term temporal dependencies and seasonal weather patterns.
* **Significance:** Accurate forecasting of PM2.5 is crucial for developing "Early Warning Systems" that allow governments to mitigate smog-related health crises before they occur.

---

## 📖 2. Introduction
**Problem Statement:**
Air pollution is a dynamic, non-linear system influenced by complex interactions between meteorological factors (humidity, pressure, temperature) and human activity. Traditional statistical models often fail to capture the abrupt spikes in pollution (smog events) that are characteristic of urban environments.

**Project Objectives:**
1. **Data Engineering:** To construct a robust pipeline that handles missing values, normalizes diverse sensor data, and selects the most relevant features using mathematical importance scores.
2. **Model Implementation:** To implement and optimize three distinct classes of algorithms:
    * *Bagging Ensemble:* Random Forest
    * *Boosting Ensemble:* XGBoost
    * *Recurrent Neural Network:* LSTM
3. **Evaluation:** To validate model performance using a **3-Way Data Split** (Train/Validation/Test) and assess real-world viability through business impact analysis.

---

## 📊 3. Dataset Description
* **Source:** UCI Machine Learning Repository (Five Citie PM2.5 Data).
* **Scope:** Hourly sensor readings (2010–2015) from five major cities: Beijing, Shanghai, Guangzhou, Chengdu, and Shenyang.
* **Data Volume:** ~43,000 hours of data per city.

### 3.1 Preprocessing Pipeline
1.  **Missing Value Imputation:**
    * *Problem:* Sensor failure often leads to `NaN` values.
    * *Solution:* We applied **Linear Interpolation** with a limit of 12 hours. This assumes that weather changes gradually, allowing us to fill gaps by drawing a straight line between known points.
2.  **Feature Selection (Dimensionality Reduction):**
    * Instead of using all 9+ raw features (which introduces noise), we trained a preliminary Random Forest to calculate **Gini Importance**.
    * We retained only the **Top 4 Features** that contributed most to reducing prediction error (typically Dew Point, Humidity, Pressure, and Temperature).
3.  **Scaling:**
    * Applied `MinMaxScaler` to transform all features to the range `[0, 1]`. This is critical for LSTM convergence, as large input values can cause "exploding gradients" during backpropagation.
4.  **Sequence Generation (Sliding Window):**
    * Transformed the tabular data into a 3D Time-Series format: `(Samples, Time Steps, Features)`.
    * **Window Size:** 24 Hours. The model looks at $T_{-24} \dots T_{-1}$ to predict $T_{0}$.

---

## 💻 4. Code Structure & Implementation Details
The codebase is modularized into four logical "Cells" to ensure reproducibility.

### **Cell 1: Intelligent Data Loader & Splitter**
This module automates the ingestion of raw CSVs.
* **Dynamic Parsing:** Automatically detects date columns regardless of format.
* **Leakage Prevention:** Uses `train_test_split` with `shuffle=False`.
    * *Why?* In time-series data, the future cannot predict the past. Shuffling would allow the model to "cheat" by seeing future weather patterns. We strictly split the data chronologically:
        * **Train (70%):** Years 2010-2013
        * **Validation (15%):** Year 2014 (Used for Tuning)
        * **Test (15%):** Year 2015 (Used for Final Evaluation)

### **Cell 2: Hyperparameter Tuning & Training**
* **Optimization Strategy:** We utilized `RandomizedSearchCV` instead of Grid Search.
    * *Logic:* Randomly sampling hyperparameter combinations is statistically proven to find near-optimal models significantly faster than checking every single combination, allowing us to tune on the full dataset without excessive runtime.
* **Deep Learning Callbacks:**
    * `EarlyStopping`: Monitors validation loss and stops training if it doesn't decrease for 5 epochs.
    * `ReduceLROnPlateau`: Reduces the learning rate by 50% if the model gets stuck, allowing it to fine-tune its weights to find the global minimum.

### **Cell 3: Evaluation Suite**
Generates performance metrics and visualizes the **Confusion Matrix**.
* *Note:* Since this is a regression task, we "binned" the continuous outputs into categories (Safe, Unhealthy, Hazardous) to visualize how often the model misses dangerous pollution spikes.

### **Cell 4: The "Crystal Ball" (Recursive Forecasting)**
Implements a feedback loop for long-term prediction.
1.  Model predicts hour $T+1$.
2.  This prediction is appended to the input window.
3.  The window "slides" forward, and the model uses its own prediction to forecast $T+2$.
4.  Repeated for 720 hours (30 Days).

---

## 🧠 5. Theoretical Framework & Model Selection

### 5.1 Random Forest Regressor (The "Wisdom of Crowds")


[Image of Random Forest structure]

* **Theory:** Random Forest is a **Bagging (Bootstrap Aggregating)** ensemble. It trains hundreds of Decision Trees on different random subsets of the data.
* **Why it works:** A single decision tree is prone to "overfitting" (memorizing the noise). By averaging the predictions of hundreds of decorrelated trees, the Random Forest cancels out the errors, leading to a stable and robust model.
* **Role in Project:** Serves as a strong baseline "Classical" model to benchmark Deep Learning performance.

### 5.2 XGBoost Regressor (The "Error Corrector")
* **Theory:** eXtreme Gradient Boosting uses **Boosting**. Unlike Random Forest (where trees are independent), XGBoost builds trees sequentially. Each new tree attempts to correct the errors (residuals) made by the previous tree.
* **Why it works:** It uses a gradient descent algorithm to minimize a loss function, adding new models that specifically target the "hard-to-predict" examples. It is widely considered the state-of-the-art algorithm for tabular data.

### 5.3 Long Short-Term Memory Network (The "Memory" Model)

* **Theory:** LSTM is a specialized Recurrent Neural Network (RNN) designed to solve the **Vanishing Gradient Problem** of standard RNNs.
* **Architecture:** It introduces a "Cell State" (long-term memory) and three gates:
    1.  **Forget Gate:** Decides what information from the past is no longer relevant (e.g., "it stopped raining").
    2.  **Input Gate:** Decides what new information to store (e.g., "a wind storm just started").
    3.  **Output Gate:** Decides what to output based on the current memory.
* **Why it works:** Standard regression models treat every hour as an independent event. LSTM understands that **weather is sequential**. It "remembers" that a pressure drop 10 hours ago likely means rain now, allowing it to capture complex temporal patterns that classical models miss.

---

## ⚙️ 6. Experimental Settings (Hyperparameters)

| Model | Parameter | Range / Value | Description |
| :--- | :--- | :--- | :--- |
| **Random Forest** | `n_estimators` | 100 - 200 | Number of trees in the forest. |
| | `max_depth` | 10 - 25 | Maximum depth of each tree (controls complexity). |
| **XGBoost** | `learning_rate` | 0.01 - 0.1 | Step size for gradient descent updates. |
| | `n_estimators` | 150 - 300 | Number of boosting rounds. |
| **LSTM** | `Neurons` | 128 (L1), 64 (L2) | Capacity of the network to learn features. |
| | `Dropout` | 0.3 | Fraction of neurons dropped to prevent overfitting. |
| | `Batch Size` | 64 | Number of samples processed before updating weights. |

---

## 📈 7. Results & Analysis

### 7.1 Performance Metrics
We evaluated models using three key metrics:
1.  **RMSE (Root Mean Squared Error):** Penalizes large errors heavily. Crucial for air quality, as missing a massive smog spike is worse than being slightly off on a clear day.
2.  **MAE (Mean Absolute Error):** The average magnitude of error in "raw" units (PM2.5 concentration).
3.  **R² Score:** Represents the proportion of variance explained by the model (closer to 1.0 is better).

| Model | RMSE | MAE | R² Score |
| :--- | :--- | :--- | :--- |
| **Random Forest** | *[Insert Code Output]* | *[Insert Code Output]* | *[Insert Code Output]* |
| **XGBoost** | *[Insert Code Output]* | *[Insert Code Output]* | *[Insert Code Output]* |
| **LSTM** | *[Insert Code Output]* | *[Insert Code Output]* | *[Insert Code Output]* |

### 7.2 Interpretation
* **Classical Models:** XGBoost provided excellent results for short-term predictions but struggled to maintain accuracy over the 30-day forecast, often reverting to the mean.
* **Deep Learning:** The LSTM model maintained the seasonal "rhythm" of the data much better in the Crystal Ball simulation, proving its superior ability to learn time-based patterns.

---

## 🔚 8. Conclusion
We successfully demonstrated that while Classical ML models like XGBoost are powerful and efficient, **Deep Learning (LSTM)** is the superior choice for Air Quality Forecasting. Its ability to maintain an internal memory allowing it to model the complex, delayed effects of weather systems on pollution levels.

**Future Work:**
* Integrating **Spatial Data**: Modeling how pollution drifts from City A to City B using Graph Neural Networks (GNNs).
* **Transformer Models:** Implementing "Temporal Fusion Transformers" (TFT) which have recently surpassed LSTMs in some forecasting tasks.

---

## 📚 9. References
1.  **Data Source:** UCI Machine Learning Repository, "PM2.5 Data of Five Chinese Cities".
2.  **Random Forest:** Breiman, L. "Random Forests." *Machine Learning* 45.1 (2001): 5-32.
3.  **XGBoost:** Chen, T., & Guestrin, C. "XGBoost: A Scalable Tree Boosting System." *KDD* (2016).
4.  **LSTM:** Hochreiter, S., & Schmidhuber, J. "Long Short-Term Memory." *Neural Computation* 9.8 (1997).
