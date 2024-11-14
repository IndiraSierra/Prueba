import ccxt
import pandas as pd
import numpy as np
import onnx
import onnxruntime as ort
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
import requests
from datetime import datetime, timedelta
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from newsapi import NewsApiClient

nltk.download('vader_lexicon')

def get_news_sentiment(symbol, api_key, date):
    try:
        newsapi = NewsApiClient(api_key=api_key)
        
        # Obtener noticias relacionadas con el símbolo para la fecha específica
        end_date = date + timedelta(days=1)
        articles = newsapi.get_everything(q=symbol,
                                          from_param=date.strftime('%Y-%m-%d'),
                                          to=end_date.strftime('%Y-%m-%d'),
                                          language='en',
                                          sort_by='relevancy',
                                          page_size=10)
        
        sia = SentimentIntensityAnalyzer()
        
        sentiments = []
        for article in articles['articles']:
            text = article.get('title', '')
            if article.get('description'):
                text += ' ' + article['description']
            
            if text:
                sentiment = sia.polarity_scores(text)
                sentiments.append(sentiment['compound'])
        
        avg_sentiment = np.mean(sentiments) if sentiments else 0
        return avg_sentiment
    except Exception as e:
        print(f"Error al obtener el sentimiento para {symbol} en la fecha {date}: {e}")
        return 0

investment_df = comparison_df.copy()
investment_df['price_direction'] = np.where(investment_df['prediction'].shift(-1) > investment_df['prediction'], 1, -1)
investment_df['sentiment_direction'] = np.where(investment_df['sentiment'] > 0, 1, -1)
investment_df['position'] = np.where(investment_df['price_direction'] == investment_df['sentiment_direction'], investment_df['price_direction'], 0)
investment_df['strategy_returns'] = investment_df['position'] * (investment_df['actual'].shift(-1) - investment_df['actual']) / investment_df['actual']
investment_df['buy_and_hold_returns'] = (investment_df['actual'].shift(-1) - investment_df['actual']) / investment_df['actual']

Datos normalizados guardados en 'binance_data_normalized.csv'
Sentimientos diarios guardados en 'daily_sentiments.csv'
Predicciones y sentimiento guardados en 'predicted_data_with_sentiment.csv'
Mean Absolute Error (MAE): 30.66908467315391
Root Mean Squared Error (RMSE): 36.99641752814565
R-squared (R2): 0.9257591918098058
Mean Absolute Percentage Error (MAPE): 0.00870572230484879
Gráfica guardada como 'ETH_USDT_price_prediction.png'
Gráfica de residuales guardada como 'ETH_USDT_residuals.png'
Correlation between actual and predicted prices: 0.9752007459642241
Gráfica de estrategia de inversión guardada como 'ETH_USDT_investment_strategy.png'
Gráfica de drawdown guardada como 'ETH_USDT_drawdown.png'
Sharpe Ratio: 9.41431958149606
Sortino Ratio: 11800588386323879936.0000
Número de rendimientos totales: 28
Número de rendimientos en exceso: 28
Número de rendimientos negativos: 19
Media de rendimientos en exceso: 0.005037
Desviación estándar de rendimientos negativos: 0.000000
Sortino Ratio: nan
Beta: 0.33875104783408166
Alpha: 0.006981197358213854
Cross-Validation MAE: 1270.7809910146143 ± 527.5746657573876
SMA Mean Absolute Error (MAE): 344.3737716856061
SMA Mean Absolute Error (MAE): 344.3737716856061
SMA Root Mean Squared Error (RMSE): 483.0396130996611
SMA R-squared (R2): 0.5813550203375846
Gráfica de predicción SMA guardada como 'ETH_USDT_sma_price_prediction.png'
Gráfica de precio, predicción y sentimiento guardada como 'ETH_USDT_price_prediction_sentiment.png'
Gráfica de drawdown guardada como 'ETH_USDT_drawdown.png'
Maximum Drawdown: 0.00%

"""
chart_modified
"""