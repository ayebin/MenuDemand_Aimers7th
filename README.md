# Predicting Weekly Demand for Each Menu Item at Resort Restaurants and Bars
- This project was developed for the **LG AI Research** Institute's menu demand prediction hackathon, utilizing a time-series AI model based on sales data collected from actual food and beverage (F&B) establishments.

<br>

### Project Overview

- The goal is to predict 7-day ahead menu-level demand, thereby enabling optimized inventory planning, workforce scheduling, and enhanced customer satisfaction.
- To address the demand forecasting task, we developed a HurdleLSTM model trained on 193 unique ‘restaurant_menu’ series—spanning 9 restaurants and 171 menu items—with 532 days of sales history each.
- The model achieved a ranking within the **top 11%** among 818 participating teams.

<br>

### Data Preprocessing

#### 1. Handling Duplicate Rows
No duplicate rows were found, so this preprocessing step was not required.

#### 2. Handling Negative Sales Values
As sales quantities cannot be negative, all negative values were replaced with 0.

#### 3. Pivoting the Dataset
The dataset was reshaped into a pivot table with date as the index, restaurant_menu as columns, and sales as values.

#### 4. Handling Missing Values
Since no missing values were detected, NaN handling was unnecessary.

#### 5. Feature Engineering
Since the original dataset contained limited features for training, we created additional features based on insights from EDA and statistical analysis. The engineered features include:
- Lag features 
- Rolling mean and rolling standard deviation
- Weekend and holiday indicators
- Seasonality encodings
- New menu launch indicator
- Zero-sales ratio features

<br>

### Model Components
#### LSTM Backbone
- Captures temporal dependencies in demand.

#### Embedding Layers
- Store, item, day-of-week, month, and season IDs embedded into dense vectors.
- Captures categorical patterns across restaurants and menus.

#### Lightweight Attention Module
- Allows model to focus on relevant days when forecasting.

#### Two-ㄴtage prediction
- out_zero: logits for purchase probability.
- out_qty: regression for positive demand.

<br>

### Model Training Strategy
#### Competition's Evaluation Rule: Weighted sMAPE
$$
Score = \sum_{s} w_s \cdot \Bigg( \frac{1}{|I_s|} \sum_{i \in I_s} \Big( \frac{1}{T_i} \sum_{t=1}^{T_i} \frac{2|A_{t,i} - P_{t,i}|}{|A_{t,i}| + |P_{t,i}|} \Big) \Bigg)
$$
- Restaurant-specific weights: Each restaurant has a different importance weight, but "담하" and "미라시아" especially have higher weights. 
- Zero-sales exclusion: Days with actual sales = 0 are excluded from the calculation.

#### Loss Function
- To align with the competition’s evaluation metric, we designed a custom loss function.
- Weighted per sample, with higher importance given to specific restaurants (e.g., 담하, 미라시아).
- Total Loss = λ_zero * FocalBCE(p, target>0) + λ_qty * SmoothL1(q, log1p(y))

#### Validation & Early Stopping
- For validation, we split the last *n* days from the training set.
- We used early stopping technique monitored by weighted sMAPE.

#### Hyperparameter Tuning
- The best hyperparameter combination was identified through experiments.