 ---

# Stock Price Prediction with Sentiment Analysis

This project aims to predict stock prices by integrating sentiment analysis of financial news and social media data with machine learning models. By leveraging both historical price data and market sentiment, this solution provides a comprehensive approach to stock price prediction.

## Features

- **Data Collection**: 
  - Historical stock price data is sourced from Yahoo Finance using the `yfinance` library.
  - Sentiment data is extracted from financial news and social media using Natural Language Processing (NLP) techniques.
  
- **Sentiment Analysis**: 
  - Used `VADER Sentiment Analyzer` and `TextBlob` to analyze the polarity and sentiment of financial news and tweets, transforming textual data into numerical sentiment scores.

- **Time-Series and Predictive Models**:
  - **Linear Regression**: A baseline predictive model for stock price trends.
  - **ARIMA**: Time-series analysis to model and predict stock price movements based on historical data.
  - **LSTM (Long Short-Term Memory)**: A deep learning model designed to capture long-term dependencies in time-series data, offering advanced predictive capabilities.

- **Feature Engineering**:
  - Combined features derived from historical stock data (e.g., opening/closing prices, volume) with sentiment scores to create a robust dataset for training models.

- **Evaluation**: 
  - Models are assessed using metrics like Mean Squared Error (MSE) and R² Score to ensure performance accuracy and reliability.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/AyaanHimani/Stock_Price_Prediction_With_Sentiment_Analysis.git
    ```
2. Navigate to the project directory:
    ```bash
    cd stock-price-prediction
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Change the ticker symbol in the configuration file. Ticker symbols can be found in the provided `Yahoo-Finance-Ticker-Symbols.csv` file.

## Usage

1. Run the stock-price-prediction_ml.ipynb file.
  
2. The results, including sentiment analysis and predicted stock prices, will be displayed in the console and visualized as plots.

## Key Libraries and Tools

- **`yfinance`**: Fetch historical stock price data from Yahoo Finance.
- **`TextBlob`**: Analyze textual sentiment for financial news.
- **`VADER Sentiment Analyzer`**: Provide robust sentiment scoring tailored for financial language.
- **`Linear Regression`**: A simple, interpretable predictive model.
- **`ARIMA`**: Time-series forecasting model to analyze and predict stock trends.
- **`LSTM`**: A deep learning model designed to handle sequential data, providing advanced predictive power for stock price forecasting.
- **`XGBoost`**: A gradient boosting framework used to optimize predictive accuracy with engineered features.

## Workflow Overview

1. **Data Collection**:
    - Historical stock data: `yfinance`.
    - Sentiment scores: `VADER` and `TextBlob`.

2. **Data Preprocessing**:
    - Cleaning and scaling numerical data using `StandardScaler`.
    - Transforming text data into sentiment scores.

3. **Model Training**:
    - Train `Linear Regression`, `ARIMA`, and `LSTM` models using preprocessed features.

4. **Prediction and Evaluation**:
    - Predict stock prices.
    - Evaluate model performance using metrics like MSE and R² Score.

5. **Visualization**:
    - Generate plots for stock price trends, actual vs. predicted prices, and sentiment trends.

## Contribution Guidelines

Contributions are welcome! If you'd like to enhance this project or fix bugs, please open an issue or submit a pull request. Let's improve together!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Yahoo Finance for stock data.
- Financial news APIs and platforms for sentiment data.
- Developers of the following libraries: 
  - `yfinance`, `VADER`, `TextBlob`, `scikit-learn`, `XGBoost`, `TensorFlow/Keras`.

---