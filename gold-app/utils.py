import pandas as pd
import numpy as np
from textblob import TextBlob

def custom_sentiment(text):
    if pd.isna(text):
        return 0
    
    text = str(text).lower()
    score = TextBlob(text).sentiment.polarity
    
    if "rise" in text or "growth" in text:
        score += 0.3
    if "fall" in text or "decline" in text:
        score -= 0.3
    if "inflation" in text or "crisis" in text:
        score += 0.2
    
    return score

def prepare_features(df):
    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
        df["Month"] = df["Date"].dt.month
        df["Weekday"] = df["Date"].dt.weekday

    article_cols = ["Article1","Article2","Article3","Article4","Article5"]
    
    for col in article_cols:
        if col in df.columns:
            df[col+"_sent"] = df[col].apply(custom_sentiment)

    valid_article_cols = [col+"_sent" for col in article_cols if col in df.columns]
    if len(valid_article_cols) > 0:
        df["Final_Sentiment"] = df[valid_article_cols].mean(axis=1)
    else:
        df["Final_Sentiment"] = 0

    df["Lag1"] = df["Gold_Price"].shift(1)
    df["Lag2"] = df["Gold_Price"].shift(2)
    df["Lag3"] = df["Gold_Price"].shift(3)

    df["MA_5"] = df["Gold_Price"].rolling(5).mean()
    df["MA_10"] = df["Gold_Price"].rolling(10).mean()

    df["Momentum"] = df["Gold_Price"] - df["Gold_Price"].shift(5)
    df["Trend"] = df["MA_5"] - df["MA_10"]

    df["Deviation"] = df["Gold_Price"] - df["MA_10"]

    df["Volatility_5"] = df["Gold_Price"].rolling(5).std()
    df["Volatility_10"] = df["Gold_Price"].rolling(10).std()

    # Returns
    df["Return"] = np.log(df["Gold_Price"] / df["Gold_Price"].shift(1))
    df["Return_Lag1"] = df["Return"].shift(1)
    df["Return_Lag2"] = df["Return"].shift(2)

    # Target
    df["Target"] = np.log(df["Gold_Price"] / df["Gold_Price"].shift(3))

    df = df.dropna()
    return df

def extract_features(df):
    X = df[[
        "Return_Lag1",
        "Return_Lag2",
        "Momentum",
        "Trend",
        "Deviation",
        "Volatility_5",
        "Volatility_10",
        "Final_Sentiment",
        "Month",
        "Weekday"
    ]]
    y = df["Target"]
    return X, y

def predict_future(model_full, model_recent, df, days=10, model_type="Hybrid", custom_sentiment_value=None):
    future_preds = []

    last_price = df["Gold_Price"].iloc[-1]

    lag1 = df["Gold_Price"].iloc[-1]
    lag2 = df["Gold_Price"].iloc[-2]
    lag3 = df["Gold_Price"].iloc[-3]

    ret1 = df["Return"].iloc[-1]
    ret2 = df["Return"].iloc[-2]

    month = df["Month"].iloc[-1]
    weekday = df["Weekday"].iloc[-1]

    sentiment = df["Final_Sentiment"].iloc[-1]
    
    # Pre-calculate avg volatility for dynamic weighting logic
    avg_historical_vol = df['Volatility_10'].mean()

    for i in range(days):

        ma5 = df["Gold_Price"].tail(5).mean()
        ma10 = df["Gold_Price"].tail(10).mean()

        momentum = lag1 - df["Gold_Price"].iloc[-5]
        trend = ma5 - ma10
        deviation = lag1 - ma10

        vol5 = df["Gold_Price"].tail(5).std()
        vol10 = df["Gold_Price"].tail(10).std()
        
        # COMBINATION LOGIC (Requirement 4)
        if custom_sentiment_value is not None:
            w1_sent = 0.8
            w2_sent = 0.2
            sentiment_feature = (w1_sent * sentiment) + (w2_sent * custom_sentiment_value)
        else:
            sentiment_feature = sentiment

        # Assemble feature array
        features = np.array([[
            ret1, ret2,
            momentum,
            trend,
            deviation,
            vol5,
            vol10,
            sentiment_feature,
            month,
            weekday
        ]])

        # 1. Full Dataset Model Output
        full_model_output = model_full.predict(features)[0]
        
        # 2. Recent 365-Day Model Output
        sequential_model_output = model_recent.predict(features)[0]

        # Model type logic
        if model_type == "Full":
            w1 = 1.0
            w2 = 0.0
            w3 = 0.3 # Default mean reversion weight
        elif model_type == "Recent":
            w1 = 0.0
            w2 = 1.0
            w3 = 0.3
        else:
            # Dynamic Volatility Logic to modify weights
            vol_ratio = vol10 / avg_historical_vol if avg_historical_vol > 0 else 1.0
            
            if vol_ratio > 1.2:
                # High volatility (Panicking/Strong trend) -> Trust Recent Model more
                w1 = 0.3
                w2 = 0.7
                w3 = 0.1 # Low mean reversion
            elif vol_ratio < 0.8:
                # Low volatility (Consolidation range) -> Trust historical stable Full Model more
                w1 = 0.7
                w2 = 0.3
                w3 = 0.4 # Higher mean reversion
            else:
                w1 = 0.5
                w2 = 0.5
                w3 = 0.3

        # Weighted Model Combination
        final_prediction = (w1 * full_model_output) + (w2 * sequential_model_output)

        # Mean reversion calculation
        reversion_return = - deviation / lag1

        # FINAL COMBINATION based on provided logic framework
        final_return = (1.0 - w3) * final_prediction + (w3 * reversion_return)

        # ====== FIXED IMPACT RULE (Requirements 1, 2, 3) ======
        if custom_sentiment_value is not None and custom_sentiment_value != 0:
            # Custom input contributes at least ±0.02 (2%) influence
            if custom_sentiment_value > 0:
                impact = max(0.02, custom_sentiment_value * 0.05)
            else:
                impact = min(-0.02, custom_sentiment_value * 0.05)
            final_return += impact

        # Safety clip (Requirement 5)
        # Bounded between ±0.06 to ensure the custom input physically shapes the curve without breaking market realism
        final_return = np.clip(final_return, -0.06, 0.06)

        # Convert to price
        next_price = last_price * np.exp(final_return)

        future_preds.append(next_price)

        # update lags
        lag3 = lag2
        lag2 = lag1
        lag1 = next_price

        ret2 = ret1
        ret1 = final_return

        last_price = next_price
        weekday = (weekday + 1) % 7

    return future_preds
