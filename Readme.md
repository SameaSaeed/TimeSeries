##### **Time Series Data**



Time series data is a sequence of data points collected over successive, equally spaced points in time. It's characterized by its chronological order, allowing for the analysis of trends, patterns, and forecasting over a period. This type of data is crucial for understanding how a variable changes over time and is used in various fields like finance, weather forecasting, and inventory management.



###### Key characteristics of time series data:



a. Chronologically Sequential order: Data points are recorded in a specific order based on time.

b. Regular intervals/Constant frequency: Continuous Measurements are typically taken at consistent intervals (e.g., daily, monthly, hourly) without any missing valuea

c. Temporal components: Trend, seasonality, and randomness: Time series data often exhibits trends (long-term direction), seasonality (repeating patterns within a year), and random fluctuations.

d. Dynamic nature: Affected by external factors

###### 

###### Examples of time series data:



Stock prices: Daily closing prices of a stock over a period of time.

Weather data: Temperature, rainfall, and wind speed recorded at regular intervals.

Sales figures: Monthly sales data for a product or service.

Website traffic: Number of visitors to a website over time.



###### Common applications of time series data:



Forecasting: Predicting future values based on historical data.

Trend analysis: Identifying patterns and changes over time.

Anomaly detection: Identifying unusual data points that deviate from the norm.

Business intelligence: Supporting informed decision-making and strategic planning.

###### 

###### Distinguishing time series data from other data types:



Cross-sectional data: Measures multiple variables at a single point in time.

Pooled data: Combines both cross-sectional and time series data.



##### **Time Series Analysis**



###### Statistical methods for Time-series Analysis



1. Classical Decomposition: Assumes a fixed seasonal pattern: Easily influenced by outliers

2\. STL Decomposition using LOESS: Shows actual seasonal pattern: Handles outliers: Only handles additive models



##### **Time Series Decomposition**



###### Components of Decomposition



Trends: Long-term direction (upwards/downwards)
Seasonality: Repeating pattern at fixed intervals

Cyclic: Repeating pattern but not at fixed intervals (Business, Economy)

Residuals/Noise: Sudden/Random fluctuation in time-series data



###### Types of decomposition models



In time series analysis, additive and multiplicative models are used to decompose a time series into its constituent components (trend, seasonality, and residuals). The key difference lies in how these components are combined: additive models assume they are added together, while multiplicative models assume they are multiplied.



1. **Additive Model:**



In an additive model, the time series is represented as the sum of its components: y(t) = Trend(t) + Seasonality(t) + Residual(t).

This model is appropriate when the seasonal fluctuations or the variation around the trend do not change with the level of the time series.



For example, if you have a time series of monthly sales data for a product, and the seasonal peaks and troughs remain relatively consistent in magnitude over time, even as the overall sales trend increases, an additive model might be suitable.



An additive trend indicates a linear trend, and an additive seasonality indicates the same frequency (width) and amplitude (height) of seasonal cycles.



**2. Multiplicative Model:**



In a multiplicative model, the time series is represented as the product of its components: y(t) = Trend(t) \* Seasonality(t) \* Residual(t).



This model is appropriate when the seasonal fluctuations or the variation around the trend are proportional to the level of the time series.



For example, if you have a time series of website traffic, where the peaks during the holiday season become much larger as the overall traffic grows, a multiplicative model might be more appropriate.



A multiplicative trend indicates a non-linear trend (curved trend line), and a multiplicative seasonality indicates increasing/decreasing frequency (width) and/or amplitude (height) of seasonal cycles, according to a blog post on Towards Data Science.



**3. Choosing between Additive and Multiplicative Models:**



The choice between additive and multiplicative models depends on the characteristics of the time series data.

If the magnitude of seasonal variations remains constant regardless of the level of the time series (linear), an additive model is suitable.



If the magnitude of seasonal variations changes proportionally with the level of the time series, a multiplicative model is more appropriate (exponential).



In some cases, data transformation (e.g., log transformation) can be used to convert a multiplicative time series into an additive one, allowing the use of an additive model.



###### Stationarity



1. **Weak Stationarity (Shorter period of data)**



Make predictions easier by assuming the same statistical properties over time i.e. Mean (no trend), Variance, and Auto-correlation constant over time.



*a. ADF Test*



The Augmented Dickey-Fuller (ADF) test is a statistical test used to determine if a time series is stationary or not. It is a unit root test, meaning it tests the null hypothesis that a time series is non-stationary and has a unit root.



What it tests:

Null Hypothesis: The time series is non-stationary (has a unit root).

Alternative Hypothesis: The time series is stationary.



Interpretation:

Low p-value (typically less than 0.05): Reject the null hypothesis, indicating the series is likely stationary.

High p-value: Fail to reject the null hypothesis, suggesting the series may not be stationary.



b. KPSS Test



The KPSS test can be used to check for both trend stationarity (where the series fluctuates around a trend) and level stationarity (where the series fluctuates around a constant level).



**2. Strict Stationarity (Longer period of data)**



Weak Stationarity + Joint distribution remains unchanged when shifted along any time period.



a. KS Test



The Kolmogorov-Smirnov (K-S) test is a nonparametric statistical test used to determine if two datasets come from the same distribution or if a sample comes from a specific distribution. It's particularly useful when you don't want to assume a particular distribution for your data.



###### Making Timeseries Data Stationary



1. Differentiation: 2nd order
2. Transformation: Stabilizes the variance of time series data
   Logarithmic
   Power
   Box-cox
3. Detrending: Removing trend component
   Linear
   Moving average
4. Seasonal Adjustment: Removing seasonal component
   STL Decomposition



###### White Noise and Random Walk



White Noise: Completely random and unpredictable with no pattern, trend, or seasonality, along with constant mean and variance and no auto-correlation, for eg. coin tossing



Random Walk: A Cumulative pattern with mean and variance changing over time and a stationary 1st order difference leading to no predictable pattern.



Identifying Whit Noise and Random Walk:

1. Visually
2. ACF and PACF

   For stationary data, the ACF plot decays exponentially or sinusoidally

3. Ljung-box test

   

   ###### Smoothing

   

   Noise Reduction to forecast high-fluctuating time-series data and trend identification.

   

1. Simple Moving Average
2. Weighted Moving Average
3. Exponential

   

   ##### **Time Series Data Pre-processing**

   

1. Handling Missing Values:
   Imputation with mean, median, mode, forward fill, backward fill
   Interpolation with linear, spline, and polynomial
   Predictive Modelling with Machine Learning
2. Making Data Stationary
3. Handling Outliers
   Imputation
   Interpolation
   Transformation
   Smoothing
4. Resampling (frequency= hourly, daily, monthly, quarterly, yearly); Interpolation fills the gaps

      Down-sampling

      Up-sampling

   

   ##### **Time Series Forecasting**

   

   ARIMA, Prophet, and LSTM each offer unique strengths for time series forecasting, making the choice dependent on the specific characteristics of the data and the forecasting goals. ARIMA is suitable for simpler, stationary time series, while Prophet excels with business-related data featuring strong seasonality. LSTM, a deep learning model, can capture complex, non-linear patterns and long-term dependencies, making it ideal for intricate time series.

   

   ###### 1\. ARIMA (Autoregressive Integrated Moving Average):

   

   ARMA (Combination of AR and MA, and chosen when both ACF and PACF plots decay)

   ARIMA, SARIMA (univariate)

   VARMA (multi-variate)

   

   Strengths:

   Well-established statistical method, performs well with **stationary** **time** **series** (where statistical properties don't change over time), good for short-term forecasting.

   

   Limitations:

   Can be challenging to apply to non-stationary data, requires careful hyperparameter tuning, may not capture complex non-linear patterns effectively.

   

   Use Cases:

   Financial forecasting, inventory management, and other scenarios where the data exhibits a clear trend and seasonality that can be modeled with linear relationships.

   

   ###### 2\. Prophet:

   

   Strengths:

   Designed for business time series with strong seasonality and holiday effects, user-friendly, handles missing data and outliers well, requires less hyperparameter tuning than ARIMA, according to Meta Open Source.

   

   Limitations:

   Prophet is particularly effective for univariate time series with clear trends and seasonality. May not be suitable for time series without clear seasonal patterns or where the concept of calendar date is not relevant, can be outperformed by ARIMA and LSTM in certain scenarios.

   

   Use Cases:

   Sales forecasting, demand planning, and other applications where seasonal trends are important.

   

   ###### 3\. LSTM (Long Short-Term Memory):

   

   Strengths:

   Powerful deep learning model capable of capturing complex, non-linear patterns and long-term dependencies, suitable for large datasets with intricate relationships, making it well-suited for multivariate time series.

   

   Limitations:

   Computationally intensive, requires a large amount of data for training, can be prone to overfitting, requires more expertise to implement and tune.

   

   Use Cases:

   Financial time series with complex patterns, weather forecasting, healthcare data analysis, and other scenarios where traditional methods may not be sufficient, according to Alibaba Cloud.

   

   ###### 4\. Choosing between models

   

   Choose ARIMA

   when dealing with relatively simple, stationary time series where you need a baseline model for comparison or when dealing with shorter forecasting horizons.

   

   Choose Prophet

   when you have business-related time series with strong seasonality and you need a user-friendly tool that can

   handle missing data and outliers.

   

   Choose LSTM

   when you have complex, non-linear data, long-term dependencies, and access to a large dataset, and you are willing to invest the time and resources to train and tune the model.

   ###### 

   ###### 5\. Performance Measures

   

   MAE, MSE, RMSE, MAPE

