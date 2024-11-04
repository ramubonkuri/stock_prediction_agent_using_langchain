
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime, timedelta

class StockPredictor:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.model = LinearRegression()
        self.current_price = 0.0
        self.metrics = {}
        
    def get_historical_data(self):
        stock = yf.Ticker(self.symbol)
        data = stock.history(period='1y')
        return data
        
    def prepare_data(self, data):
        # Create features (X) using days from start
        data['Date'] = range(len(data))
        X = data[['Date']].values
        y = data['Close'].values
        return X, y
        
    def fit(self):
        data = self.get_historical_data()
        self.current_price = data['Close'].iloc[-1]
        X, y = self.prepare_data(data)
        
        # Fit the model
        self.model.fit(X, y)
        
        # Calculate metrics
        y_pred = self.model.predict(X)
        self.metrics['mse'] = np.mean((y - y_pred) ** 2)
        self.metrics['r2'] = self.model.score(X, y)
        
    def predict(self, days_ahead: int):
        # Generate future dates
        last_day = len(self.get_historical_data())
        future_days = np.array(range(last_day, last_day + days_ahead)).reshape(-1, 1)
        
        # Make predictions
        predictions = self.model.predict(future_days)
        
        # Calculate daily changes
        changes = np.diff(predictions, prepend=self.current_price)
        changes_pct = (changes / np.roll(predictions, 1)) * 100
        
        return predictions, changes_pct
        
    def get_model_metrics(self):
        return self.metrics
        
    def get_current_price(self):
        return self.current_price

    def get_financial_statements(self, symbol:str):
        # time.sleep(4) #To avoid rate limit error
        if "." in symbol:
            symbol=symbol.split(".")[0]
        else:
            symbol=symbol
        symbol=symbol+".NS"    
        company = yf.Ticker(symbol)
        balance_sheet = company.balance_sheet
        if balance_sheet.shape[1]>=3:
            balance_sheet=balance_sheet.iloc[:,:3]    # Remove 4th years data
        balance_sheet=balance_sheet.dropna(how="any")
        balance_sheet = balance_sheet.to_string()
        return balance_sheet