# StockTrendSVM: Stock Predictor Enhanced with Transformer Sentiment

This project implements a classical Support Vector Machine (SVM) model to predict stock price direction, augmented with sentiment features derived from financial news headlines using a transformer-based model.

## Overview

The notebook integrates traditional technical indicators and macroeconomic data with modern NLP sentiment analysis to improve stock price movement prediction. The model is trained and tested on historical Apple (AAPL) stock data.

## Features

- Technical indicators: Moving averages, volatility, returns.
- Macroeconomic feature: 10-Year Treasury Rate from FRED API.
- Sentiment analysis: Transformer-based sentiment scoring of financial news headlines.
- Classical SVM classifier for price direction prediction.
- Visualisation of actual vs predicted price movements.

## Requirements

- Python 3.7+
- yfinance
- fredapi
- nltk
- transformers
- datasets
- scikit-learn
- matplotlib
- pandas
- numpy

## Usage

1. Clone this repository or download the notebook.
2. Replace the `FRED_API_KEY` in the notebook with your own API key from [FRED](https://fred.stlouisfed.org/docs/api/api_key.html).
3. Run the notebook in Google Colab or any Python environment.
4. The notebook downloads historical stock data, fetches macroeconomic indicators, computes sentiment scores from sample news headlines, trains an SVM model, and evaluates its predictions.

## Notes

- The sentiment analysis uses a pre-trained DistilBERT model fine-tuned on SST-2.
- The example financial news data is limited; for better performance, replace with a larger, real dataset.
- The FRED API key is required to fetch macroeconomic data; without it, the model uses a fallback value.

## Licence

This project is licensed under the Apache License 2.0
