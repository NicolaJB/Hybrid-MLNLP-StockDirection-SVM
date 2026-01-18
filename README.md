# Hybrid ML/NLP Stock Direction Predictor – Classical Prototype

## Overview
This project demonstrates a classical Support Vector Machine (SVM) model for predicting next-day stock price direction, enhanced with sentiment features derived from financial news using a transformer-based model. 

The pipeline combines:
- **Technical Indicators** (moving averages, volatility, returns)
- **Macroeconomic Data** (10-Year Treasury Rate from FRED API)
- **Transformer-Based NLP Sentiment Analysis** (financial news headlines)
- **Classical SVM Classifier** for predicting next-day stock movement

The notebook provides an end-to-end workflow, including data fetching, feature engineering, model training, evaluation, and simple backtesting. It is fully compatible with Google Colab and suitable for educational or exploratory use in hybrid ML/quant projects.

## Features
| Capability                     | Description                                                        |
|--------------------------------|--------------------------------------------------------------------|
| Technical Indicators            | Moving averages (MA10, MA50), daily returns, and volatility       |
| Macroeconomic Feature           | 10-Year Treasury Rate from FRED API                                 |
| Transformer Sentiment Analysis  | Converts financial news headlines into numerical sentiment scores |
| Classical SVM Classifier        | Predicts next-day stock price direction (Up/Down)                 |
| Visualisation                   | Cumulative returns chart with prediction accuracy markers         |

## Skills & Tools
- **Programming & Libraries:** Python, Pandas, NumPy, Matplotlib  
- **Machine Learning:** SVM, Scikit-learn  
- **NLP & Transformers:** Hugging Face Transformers  
- **Financial Data:** yfinance, FRED API  
- **Time Series & Feature Engineering:** Rolling averages, volatility, macroeconomic features  

## Repository Structure
```
stocktrendsvm/
├─ README.md               # This file
├─ Hybrid-MLNLP-StockDirection-SVM.ipynb     # End-to-end notebook
```
## Quick Start
Clone the repository:
```bash
git clone https://github.com/YourUsername/Hybrid-MLNLP-StockDirection-SVM.git
cd Hybrid-MLNLP-StockDirection-SVM
```
Open the notebook in Google Colab or Jupyter:
- Colab: https://colab.research.google.com/github/YourUsername/Hybrid-MLNLP-StockDirection-SVM/blob/main/Hybrid-MLNLP-StockDirectionPredictor–ClassicalPrototype.ipynb
- Jupyter: Open `Hybrid-MLNLP-StockDirectionPredictor–ClassicalPrototype.ipynb` locally after cloning.

Install dependencies directly in the notebook (already included in the first cell):
```bash
!pip install yfinance fredapi nltk transformers datasets scikit-learn matplotlib pandas numpy --quiet
```

Configure API Keys: 
- Replace FRED_API_KEY in the notebook with your own key to fetch macroeconomic data.

Run the Notebook:
- The notebook downloads historical stock data, computes technical and sentiment features, trains an SVM model, evaluates predictions, and visualises cumulative strategy vs market returns.

**Example Strategy vs Market Cumulative Returns**  

![Strategy vs Market Chart](Hybrid-MLNLP-StockDirectionPredictor-ChartExample.png)

## Notes
- Sentiment analysis uses a pre-trained DistilBERT model fine-tuned on SST-2.
- The sample news dataset is limited; using a larger real dataset will improve results.
- If FRED API is unavailable, the notebook uses a fallback value of 0 for interest rates.

## Educational Goals

This project demonstrates:

- Combining classical ML models with modern NLP sentiment features
- Integration of technical, macroeconomic, and sentiment indicators for predictive modeling
- End-to-end, reproducible ML pipelines for financial time-series analysis

## License
This project is licensed under the Apache License 2.0 – see the [LICENSE](LICENSE) file for details.
