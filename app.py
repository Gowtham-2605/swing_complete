# app.py
"""
Enhanced Stock Scanner – Version 5.1 (Multi-Timeframe Rule-Based)
- Fixed yfinance data fetching issues
- Proper intraday data handling
- Multi-timeframe confirmation (HTF/MTF/LTF)
- Lightweight candle pattern recognition
- Score-based trade readiness
- Professional risk management
- Modern glassmorphic UI
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from logzero import logger
import time
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import hashlib
import math

# -------------------------------------------------
# Flask setup
# -------------------------------------------------
app = Flask(__name__)
CORS(app)

# -------------------------------------------------
# Configuration
# -------------------------------------------------
# Use symbols without .NS suffix
NIFTY_50 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "AXISBANK.NS", "WIPRO.NS", "SBIN.NS", "ITC.NS", "LT.NS",
    "MARUTI.NS", "ASIANPAINT.NS", "KOTAKBANK.NS", "SUNPHARMA.NS", "DRREDDY.NS",
    "BRITANNIA.NS", "HINDALCO.NS", "ULTRACEMCO.NS", "NTPC.NS", "POWERGRID.NS",
    "GRASIM.NS", "TECHM.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "EICHERMOT.NS",
    "HEROMOTOCO.NS", "ADANIENT.NS", "ADANIGREEN.NS", "TITAN.NS", "BAJAJFINSV.NS",
    "BAJAJ-AUTO.NS", "BOSCHLTD.NS", "NESTLEIND.NS", "APOLLOHOSP.NS", "M&M.NS",
    "TATACONSUM.NS", "ONGC.NS", "SAIL.NS", "HINDUNILVR.NS", "HINDPETRO.NS",
    "BEL.NS", "INDUSTOWER.NS", "SHRIRAMFIN.NS", "KPITTECH.NS", "PIDILITIND.NS",
    "COALINDIA.NS", "HCLTECH.NS"
]

CACHE_EXPIRY = 300
MAX_WORKERS = 8  # Reduced to avoid rate limiting
RATE_LIMIT_DELAY = 0.1  # Increased delay

# Timeframe hierarchy with adjusted periods for yfinance limitations
HTF_INTERVAL = "60m"    # 1-hour for trend permission
MTF_INTERVAL = "15m"    # 15-min for structure
LTF_INTERVAL = "5m"     # 5-min for entry timing

# Adjusted periods for yfinance intraday data limitations
# yfinance only provides intraday data for last 60 days
HTF_PERIOD = "30d"      # 30 days for 1H data
MTF_PERIOD = "15d"      # 15 days for 15M data  
LTF_PERIOD = "7d"       # 7 days for 5M data
MAIN_PERIOD = "15d"     # Main analysis period

# Global state for stats
SCAN_CACHE = {"data": None, "timestamp": None}
CURRENT_METHOD = "ATR"
CURRENT_TIMEFRAME = "15m"

# -------------------------------------------------
# Cache System
# -------------------------------------------------
class SmartCache:
    def __init__(self):
        self.memory_cache = {}
        self.hits = 0
        self.misses = 0

    def get(self, key):
        cached = self.memory_cache.get(key)
        if cached and cached['expiry'] > time.time():
            self.hits += 1
            return cached['data']
        self.misses += 1
        return None

    def set(self, key, value, expiry=CACHE_EXPIRY):
        self.memory_cache[key] = {'data': value, 'expiry': time.time() + expiry}

    def get_stats(self):
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {"hits": self.hits, "misses": self.misses, "hit_rate": round(hit_rate, 2)}

cache = SmartCache()

# -------------------------------------------------
# Data Fetching with yfinance error handling
# -------------------------------------------------
def get_cache_key(symbol, interval, period):
    time_bucket = datetime.now().strftime('%Y%m%d%H') + str(datetime.now().minute // 5)
    return f"stock:{symbol}:{interval}:{period}:{time_bucket}"

def fetch_stock_data(symbol: str, interval: str = "60m", period: str = "15d"):
    """
    Fetch stock data with yfinance error handling
    Note: yfinance has limitations:
    - Intraday data only available for last 60 days
    - Different intervals have different max periods
    """
    cache_key = get_cache_key(symbol, interval, period)
    cached_data = cache.get(cache_key)
    if cached_data:
        try:
            return pd.DataFrame(cached_data)
        except Exception:
            pass
    
    try:
        time.sleep(RATE_LIMIT_DELAY)
        
        # Adjust period based on interval for yfinance limitations
        if interval in ["5m", "15m", "30m"]:
            # For intraday intervals, limit to 60 days max
            if period in ["3mo", "60d"]:
                period = "60d"
            elif period in ["1mo", "30d"]:
                period = "30d"
            elif period in ["2wk", "15d"]:
                period = "15d"
            elif period in ["1wk", "7d"]:
                period = "7d"
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            # Try with shorter period if data not available
            if period != "7d":
                return fetch_stock_data(symbol, interval, "7d")
            return None
            
        if len(df) < 10:  # Need minimum data
            return None
            
        df.reset_index(inplace=True)
        date_col = "Datetime" if "Datetime" in df.columns else "Date"
        df = df[[date_col, "Open", "High", "Low", "Close", "Volume"]]
        df.columns = ["date", "open", "high", "low", "close", "volume"]
        
        # Ensure numeric columns
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        
        if df.empty:
            return None
            
        cache.set(cache_key, df.to_dict('records'), CACHE_EXPIRY)
        return df
        
    except Exception as e:
        logger.debug(f"Error fetching {symbol} ({interval}/{period}): {e}")
        return None

def get_live_price(symbol: str) -> float:
    cache_key = f"ltp:{symbol}:{datetime.now().strftime('%Y%m%d%H%M')}"
    cached = cache.get(cache_key)
    if cached:
        return cached
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d", interval="1m")
        if not data.empty:
            ltp = float(data["Close"].iloc[-1])
            cache.set(cache_key, ltp, 60)
            return ltp
    except Exception as e:
        logger.debug(f"Live price error for {symbol}: {e}")
    
    # Fallback: get last close from daily data
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="5d", interval="1d")
        if not data.empty:
            return float(data["Close"].iloc[-1])
    except Exception:
        pass
        
    return None

# -------------------------------------------------
# Technical Indicators
# -------------------------------------------------
def calculate_ema(series: pd.Series, length: int) -> float:
    if len(series) < length:
        return float(series.iloc[-1])
    try:
        return float(series.ewm(span=length, adjust=False).mean().iloc[-1])
    except Exception:
        return float(series.iloc[-1])

def calculate_rsi(prices, period=14):
    try:
        if len(prices) < period + 1:
            return 50.0
            
        prices = np.array(prices, dtype=float)
        deltas = np.diff(prices)
        
        # Initial calculation
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gain[:period])
        avg_loss = np.mean(loss[:period])
        
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Smoothing
        for i in range(period, len(prices)-1):
            avg_gain = (avg_gain * (period-1) + gain[i]) / period
            avg_loss = (avg_loss * (period-1) + loss[i]) / period
            
            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                
        return float(rsi)
    except Exception:
        return 50.0

def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    try:
        if len(df) < period + 1:
            return 0.0
            
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0
    except Exception:
        return 0.0

def calculate_adx(df: pd.DataFrame, period: int = 14):
    try:
        if len(df) < period * 2:
            return 20.0, 20.0, 20.0
            
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        # Calculate +DM and -DM
        up_move = high.diff()
        down_move = low.diff().abs() * -1
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Calculate True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Smooth the values
        def smooth(series, period):
            return series.rolling(window=period).sum()
        
        atr = smooth(tr, period)
        plus_di = 100 * smooth(pd.Series(plus_dm), period) / atr
        minus_di = 100 * smooth(pd.Series(minus_dm), period) / atr
        
        # Calculate DX and ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).abs()
        adx = dx.rolling(window=period).mean()
        
        last_idx = -1
        while pd.isna(adx.iloc[last_idx]) and abs(last_idx) < len(adx):
            last_idx -= 1
            
        if pd.isna(adx.iloc[last_idx]):
            return 20.0, 20.0, 20.0
            
        return (
            float(adx.iloc[last_idx]),
            float(plus_di.iloc[last_idx]) if not pd.isna(plus_di.iloc[last_idx]) else 20.0,
            float(minus_di.iloc[last_idx]) if not pd.isna(minus_di.iloc[last_idx]) else 20.0
        )
    except Exception:
        return 20.0, 20.0, 20.0

def calculate_vwap(df: pd.DataFrame) -> float:
    try:
        if df.empty:
            return 0.0
            
        typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
        vwap = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()
        
        last_valid = vwap.iloc[-1]
        if pd.isna(last_valid):
            # Find last non-NaN value
            for i in range(2, len(vwap) + 1):
                val = vwap.iloc[-i]
                if not pd.isna(val):
                    return float(val)
            return float(df["close"].iloc[-1])
            
        return float(last_valid)
    except Exception:
        return 0.0

def calculate_macd(close_series):
    try:
        if len(close_series) < 26:
            return 0.0, 0.0, 0.0
            
        ema12 = close_series.ewm(span=12, adjust=False).mean()
        ema26 = close_series.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        
        return (
            float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else 0.0,
            float(signal.iloc[-1]) if not pd.isna(signal.iloc[-1]) else 0.0,
            float(hist.iloc[-1]) if not pd.isna(hist.iloc[-1]) else 0.0
        )
    except Exception:
        return 0.0, 0.0, 0.0

# -------------------------------------------------
# Candle Pattern Detection
# -------------------------------------------------
def detect_candle_pattern(df):
    """Lightweight candle pattern detection"""
    patterns = []
    if df is None or len(df) < 3:
        return patterns
    
    try:
        # Get last 3 candles
        o = df['open'].iloc[-3:].values
        h = df['high'].iloc[-3:].values
        l = df['low'].iloc[-3:].values
        c = df['close'].iloc[-3:].values
        
        # Current candle data
        o1, h1, l1, c1 = o[-1], h[-1], l[-1], c[-1]
        o2, h2, l2, c2 = o[-2], h[-2], l[-2], c[-2]
        
        # Skip if any value is NaN
        if any(np.isnan([o1, h1, l1, c1, o2, h2, l2, c2])):
            return patterns
        
        # Body and range calculations
        body1 = abs(c1 - o1)
        range1 = h1 - l1 if h1 > l1 else 0.001
        body2 = abs(c2 - o2)
        range2 = h2 - l2 if h2 > l2 else 0.001
        
        # Bullish Engulfing
        if c2 < o2 and c1 > o1 and c1 > o2 and o1 < c2:
            patterns.append("BULLISH_ENGULFING")
        
        # Bearish Engulfing
        if c2 > o2 and c1 < o1 and c1 < o2 and o1 > c2:
            patterns.append("BEARISH_ENGULFING")
        
        # Hammer (bullish reversal)
        if range1 > 0 and body1 < 0.4 * range1:
            lower_wick = min(o1, c1) - l1
            upper_wick = h1 - max(o1, c1)
            if lower_wick > 2 * body1 and upper_wick < body1 * 0.5:
                patterns.append("HAMMER")
        
        # Shooting Star (bearish reversal)
        if range1 > 0 and body1 < 0.4 * range1:
            upper_wick = h1 - max(o1, c1)
            lower_wick = min(o1, c1) - l1
            if upper_wick > 2 * body1 and lower_wick < body1 * 0.5:
                patterns.append("SHOOTING_STAR")
                
    except Exception as e:
        logger.debug(f"Candle pattern error: {e}")
    
    return patterns

# -------------------------------------------------
# Multi-Timeframe Analysis
# -------------------------------------------------
def analyze_htf(symbol: str):
    """High Timeframe (1H) analysis for trend permission"""
    df = fetch_stock_data(symbol, interval=HTF_INTERVAL, period=HTF_PERIOD)
    if df is None or len(df) < 20:
        return "NEUTRAL", 0, 0, 0, 0
    
    close = df["close"]
    
    # Calculate indicators
    ema20 = calculate_ema(close, 20)
    ema50 = calculate_ema(close, 50)
    vwap = calculate_vwap(df)
    adx, plus_di, minus_di = calculate_adx(df, 14)
    price = float(close.iloc[-1])
    
    # Trend logic
    bias = "NEUTRAL"
    score = 0
    
    # Bullish conditions
    bullish_conditions = 0
    if price > ema20 and ema20 > ema50:
        bullish_conditions += 1
    if price > vwap:
        bullish_conditions += 1
    if adx >= 20 and plus_di > minus_di:
        bullish_conditions += 1
    
    # Bearish conditions
    bearish_conditions = 0
    if price < ema20 and ema20 < ema50:
        bearish_conditions += 1
    if price < vwap:
        bearish_conditions += 1
    if adx >= 20 and minus_di > plus_di:
        bearish_conditions += 1
    
    # Determine bias
    if bullish_conditions >= 2:
        bias = "BULLISH"
        score = bullish_conditions * 10
    elif bearish_conditions >= 2:
        bias = "BEARISH"
        score = bearish_conditions * 10
    
    return bias, score, ema20, ema50, adx

def analyze_mtf(symbol: str):
    """Medium Timeframe (15m) analysis for structure confirmation"""
    df = fetch_stock_data(symbol, interval=MTF_INTERVAL, period=MTF_PERIOD)
    if df is None or len(df) < 15:
        return "NEUTRAL", 0, 0, 0
    
    close = df["close"]
    vwap = calculate_vwap(df)
    
    ema9 = calculate_ema(close, 9)
    ema21 = calculate_ema(close, 21)
    price = float(close.iloc[-1])
    
    # Structure logic
    structure = "NEUTRAL"
    score = 0
    
    if ema9 > ema21 and price > vwap:
        structure = "BULLISH"
        score = 20
    elif ema9 < ema21 and price < vwap:
        structure = "BEARISH"
        score = 20
    
    return structure, score, ema9, ema21

def analyze_ltf(symbol: str):
    """Low Timeframe (5m) analysis for entry timing"""
    df = fetch_stock_data(symbol, interval=LTF_INTERVAL, period=LTF_PERIOD)
    if df is None or len(df) < 10:
        return "NEUTRAL", 0
    
    close = df["close"]
    
    ema9 = calculate_ema(close, 9)
    ema21 = calculate_ema(close, 21)
    price = float(close.iloc[-1])
    
    # Entry logic (only timing, not direction)
    entry_state = "NEUTRAL"
    if ema9 > ema21:
        entry_state = "BULLISH_CROSS"
    elif ema9 < ema21:
        entry_state = "BEARISH_CROSS"
    
    # Entry score based on recent crossover
    score = 0
    if len(close) >= 3:
        ema9_prev = calculate_ema(close.iloc[:-1], 9)
        ema21_prev = calculate_ema(close.iloc[:-1], 21)
        
        # Fresh crossover detection
        if (ema9 > ema21 and ema9_prev <= ema21_prev):
            score = 15  # Fresh bullish crossover
        elif (ema9 < ema21 and ema9_prev >= ema21_prev):
            score = 15  # Fresh bearish crossover
        elif entry_state != "NEUTRAL":
            score = 8   # Existing trend
    
    return entry_state, score

# -------------------------------------------------
# Score-Based Trade Readiness
# -------------------------------------------------
def calculate_trade_score(htf_bias, htf_score, mtf_structure, mtf_score, 
                         ltf_entry, ltf_score, candle_patterns, atr_ratio, 
                         rr_ratio, signal, rsi, adx):
    """Calculate confidence score (0-100) based on multiple factors"""
    score = 0
    
    # 1. HTF Alignment (MOST IMPORTANT) - +25
    if htf_bias == "BULLISH" and signal == "BUY":
        score += 25
    elif htf_bias == "BEARISH" and signal == "SELL":
        score += 25
    elif htf_bias == "NEUTRAL":
        score -= 30  # Hard penalty for neutral HTF
    
    # 2. MTF Structure Alignment - +20
    if mtf_structure == "BULLISH" and signal == "BUY":
        score += 20
    elif mtf_structure == "BEARISH" and signal == "SELL":
        score += 20
    
    # 3. LTF Entry Timing - +15
    if ltf_entry == "BULLISH_CROSS" and signal == "BUY":
        score += 15
    elif ltf_entry == "BEARISH_CROSS" and signal == "SELL":
        score += 15
    
    # 4. Candle Pattern Confirmation - +10 / -10
    if signal == "BUY":
        bullish_patterns = ["BULLISH_ENGULFING", "HAMMER"]
        bearish_patterns = ["BEARISH_ENGULFING", "SHOOTING_STAR"]
        
        for pattern in candle_patterns:
            if pattern in bullish_patterns:
                score += 10
            elif pattern in bearish_patterns:
                score -= 10
    elif signal == "SELL":
        bearish_patterns = ["BEARISH_ENGULFING", "SHOOTING_STAR"]
        bullish_patterns = ["BULLISH_ENGULFING", "HAMMER"]
        
        for pattern in candle_patterns:
            if pattern in bearish_patterns:
                score += 10
            elif pattern in bullish_patterns:
                score -= 10
    
    # 5. ADX Strength - +10
    if adx >= 25:
        score += 10
    elif adx >= 20:
        score += 5
    
    # 6. RSI confirmation
    if signal == "BUY" and rsi > 50:
        score += 5
    elif signal == "SELL" and rsi < 50:
        score += 5
    
    # 7. Risk:Reward Ratio - +10
    if rr_ratio >= 2.0:
        score += 10
    elif rr_ratio >= 1.5:
        score += 5
    
    # 8. Volatility Filter - penalty for low volatility
    if atr_ratio < 0.003:  # ATR/Price < 0.3%
        score -= 20
    
    # Ensure score is within bounds
    score = max(0, min(100, score))
    
    return int(round(score))

# -------------------------------------------------
# Rule-Based Signal Generation
# -------------------------------------------------
def generate_signal(htf_bias, mtf_structure):
    """Generate BUY/SELL/NEUTRAL signal based on timeframe alignment"""
    
    # Rule 1: HTF must provide trend permission
    if htf_bias == "NEUTRAL":
        return "NEUTRAL"
    
    # Rule 2: MTF must align with HTF
    if htf_bias == "BULLISH" and mtf_structure != "BULLISH":
        return "NEUTRAL"
    if htf_bias == "BEARISH" and mtf_structure != "BEARISH":
        return "NEUTRAL"
    
    # Determine final signal
    if htf_bias == "BULLISH" and mtf_structure == "BULLISH":
        return "BUY"
    elif htf_bias == "BEARISH" and mtf_structure == "BEARISH":
        return "SELL"
    
    return "NEUTRAL"

# -------------------------------------------------
# Main Scanner Function
# -------------------------------------------------
def scan_symbol(symbol: str, method: str = "ATR"):
    """Scan a single symbol with multi-timeframe confirmation"""
    try:
        # Fetch data for primary analysis (using MTF for main calculations)
        df = fetch_stock_data(symbol, interval=MTF_INTERVAL, period=MAIN_PERIOD)
        if df is None or len(df) < 20:
            return None
        
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]
        
        # Get live price
        ltp = get_live_price(symbol)
        if ltp is None or ltp <= 0:
            ltp = float(close.iloc[-1])
        
        # Calculate indicators on MTF
        rsi = calculate_rsi(close.values, 14)
        atr = calculate_atr(df, 14)
        adx, plus_di, minus_di = calculate_adx(df, 14)
        vwap = calculate_vwap(df)
        macd, macd_signal, macd_hist = calculate_macd(close)
        
        # Calculate EMA values
        ema9 = calculate_ema(close, 9)
        ema21 = calculate_ema(close, 21)
        ema100 = calculate_ema(close, 100) if len(close) >= 100 else ema21
        
        # Multi-Timeframe Analysis
        htf_bias, htf_score, htf_ema20, htf_ema50, htf_adx = analyze_htf(symbol)
        mtf_structure, mtf_score, mtf_ema9, mtf_ema21 = analyze_mtf(symbol)
        ltf_entry, ltf_score = analyze_ltf(symbol)
        
        # Candle Pattern Detection
        candle_patterns = detect_candle_pattern(df)
        
        # Generate signal based on timeframe alignment
        signal = generate_signal(htf_bias, mtf_structure)
        
        # Risk Management Calculations
        atr_ratio = atr / ltp if ltp > 0 else 0
        
        # Skip if volatility is too low
        if atr_ratio < 0.003:  # 0.3%
            signal = "NEUTRAL"
        
        # Calculate entry, stop loss, target based on method
        entry = round(ltp, 2)
        stop_loss = target = 0.0
        
        if signal == "BUY":
            if method == "ATR":
                stop_loss = round(ltp - 1.5 * atr, 2)
                target = round(ltp + 3.0 * atr, 2)
            else:  # Default to EMA-based
                stop_loss = round(min(ltp - 1.5 * atr, ema21), 2)
                target = round(ltp + (ltp - stop_loss) * 2.5, 2)
                
        elif signal == "SELL":
            if method == "ATR":
                stop_loss = round(ltp + 1.5 * atr, 2)
                target = round(ltp - 3.0 * atr, 2)
            else:  # Default to EMA-based
                stop_loss = round(max(ltp + 1.5 * atr, ema21), 2)
                target = round(ltp - (stop_loss - ltp) * 2.5, 2)
        
        # Calculate Risk:Reward Ratio
        rr_ratio = 0.0
        if signal in ["BUY", "SELL"] and stop_loss > 0:
            if signal == "BUY":
                risk = entry - stop_loss
                reward = target - entry
            else:
                risk = stop_loss - entry
                reward = entry - target
            
            if risk > 0:
                rr_ratio = round(reward / risk, 2)
        
        # Calculate confidence score
        confidence_score = calculate_trade_score(
            htf_bias, htf_score, mtf_structure, mtf_score,
            ltf_entry, ltf_score, candle_patterns, atr_ratio,
            rr_ratio, signal, rsi, adx
        )
        
        # Determine trade readiness
        trade_ready = (
            signal in ["BUY", "SELL"] and
            confidence_score >= 70 and
            rr_ratio >= 1.5 and
            atr_ratio >= 0.003 and
            htf_bias != "NEUTRAL"
        )
        
        # Build reason string
        reasons = []
        if htf_bias != "NEUTRAL":
            reasons.append(f"HTF:{htf_bias}")
        if mtf_structure != "NEUTRAL":
            reasons.append(f"MTF:{mtf_structure}")
        if ltf_entry != "NEUTRAL":
            reasons.append(f"LTF:{ltf_entry}")
        if candle_patterns:
            reasons.append(f"PAT:{len(candle_patterns)}")
        if rr_ratio > 0:
            reasons.append(f"RR:{rr_ratio}")
        
        reason = " | ".join(reasons) if reasons else "No strong signals"
        
        return {
            "symbol": symbol.replace(".NS", ""),
            "ltp": round(ltp, 2),
            "signal": signal,
            "entry": entry,
            "stop_loss": stop_loss,
            "target": target,
            "rr_ratio": rr_ratio,
            "confidence_score": confidence_score,
            "trade_ready": trade_ready,
            "rsi": round(rsi, 2),
            "atr": round(atr, 2),
            "adx": round(adx, 2),
            "vwap": round(vwap, 2),
            "ema9": round(ema9, 2),
            "ema21": round(ema21, 2),
            "ema100": round(ema100, 2),
            "macd_hist": round(macd_hist, 4),
            "htf_bias": htf_bias,
            "mtf_structure": mtf_structure,
            "ltf_entry": ltf_entry,
            "candle_pattern": candle_patterns,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Scan error for {symbol}: {e}")
        return None

def scan_symbol_wrapper(args):
    symbol, method = args
    return scan_symbol(symbol, method)

def scan_multiple_symbols(symbols, method="ATR"):
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_symbol = {
            executor.submit(scan_symbol_wrapper, (sym, method)): sym 
            for sym in symbols
        }
        for future in as_completed(future_to_symbol):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Scan error: {e}")
    return results

# -------------------------------------------------
# Flask Routes
# -------------------------------------------------
@app.route("/scan", methods=["POST"])
def scan():
    global CURRENT_METHOD, SCAN_CACHE
    try:
        data = request.get_json() or {}
        method = data.get("method", "ATR").upper()
        raw_symbols = data.get("symbols", [])
        
        symbols_to_scan = NIFTY_50
        if isinstance(raw_symbols, list) and len(raw_symbols) > 0:
            symbols_to_scan = [s.strip().upper() + ".NS" if not s.strip().upper().endswith(".NS") else s.strip().upper() for s in raw_symbols if s.strip()]
        
        if method not in ["ATR", "EMA"]:
            method = "ATR"
        
        CURRENT_METHOD = method
        
        logger.info(f"Scanning {len(symbols_to_scan)} symbols with {method} method...")
        start_time = time.time()
        results = scan_multiple_symbols(symbols_to_scan, method)
        scan_time = round(time.time() - start_time, 2)
        
        # Sort by trade readiness and confidence score
        results.sort(key=lambda x: (
            not x.get("trade_ready", False),
            -x.get("confidence_score", 0),
            -x.get("rr_ratio", 0)
        ))
        
        SCAN_CACHE["data"] = results
        SCAN_CACHE["timestamp"] = datetime.now()
        
        return jsonify({
            "results": results,
            "scan_time": scan_time,
            "cache_stats": cache.get_stats(),
            "total_scanned": len(symbols_to_scan),
            "successful_scans": len(results)
        })
    except Exception as e:
        logger.error(f"Scan error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/stats", methods=["GET"])
def stats():
    try:
        data = SCAN_CACHE.get("data") or []
        ts = SCAN_CACHE.get("timestamp")
        
        buy_signals = [s for s in data if s.get("signal") == "BUY"]
        sell_signals = [s for s in data if s.get("signal") == "SELL"]
        trade_ready = [s for s in data if s.get("trade_ready")]
        
        avg_score_buy = sum(s.get("confidence_score", 0) for s in buy_signals) / len(buy_signals) if buy_signals else 0
        avg_score_sell = sum(s.get("confidence_score", 0) for s in sell_signals) / len(sell_signals) if sell_signals else 0
        
        return jsonify({
            "total_scanned": len(data),
            "buy_signals": len(buy_signals),
            "sell_signals": len(sell_signals),
            "trade_ready": len(trade_ready),
            "avg_score_buy": round(avg_score_buy, 1),
            "avg_score_sell": round(avg_score_sell, 1),
            "last_scan": ts.isoformat() if ts else None,
            "method": CURRENT_METHOD,
            "cache_stats": cache.get_stats(),
            "version": "5.1 - Multi-Timeframe Rule-Based"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

# -------------------------------------------------
# HTML Template (Same as before - included for completeness)
# -------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 Advanced Stock Scanner V5.1 - Multi-Timeframe</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
            color: #e8eaf6;
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1800px;
            margin: 0 auto;
        }
        
        .header {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 25px 35px;
            margin-bottom: 25px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        h1 {
            font-size: 32px;
            font-weight: 700;
            background: linear-gradient(135deg, #00d4aa 0%, #00a8ff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }
        
        .subtitle {
            color: #9fa8da;
            font-size: 14px;
            margin-bottom: 5px;
        }
        
        .version-badge {
            display: inline-block;
            padding: 4px 12px;
            background: linear-gradient(135deg, #00d4aa 0%, #00a8ff 100%);
            color: #0a0e27;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 5px;
        }
        
        .timeframe-info {
            display: flex;
            gap: 15px;
            margin-top: 15px;
            flex-wrap: wrap;
        }
        
        .tf-badge {
            padding: 6px 15px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            font-size: 12px;
            color: #9fa8da;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .tf-badge .label {
            color: #00d4aa;
            font-weight: 600;
        }
        
        .controls {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 25px;
            margin-bottom: 25px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .control-row {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .control-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        
        label {
            font-size: 12px;
            color: #9fa8da;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        select, input {
            padding: 12px 16px;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 12px;
            color: #e8eaf6;
            font-size: 14px;
            outline: none;
            transition: all 0.3s;
            min-width: 200px;
        }
        
        select:focus, input:focus {
            background: grey;
            border-color: #00d4aa;
            box-shadow: 0 0 0 3px rgba(0, 212, 170, 0.1);
        }
        
        .btn {
            padding: 12px 28px;
            border: none;
            border-radius: 12px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #00d4aa 0%, #00a8ff 100%);
            color: #0a0e27;
            box-shadow: 0 4px 15px rgba(0, 212, 170, 0.3);
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 212, 170, 0.4);
        }
        
        .btn-secondary {
            background: rgba(255, 255, 255, 0.1);
            color: #e8eaf6;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.15);
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 15px;
            margin-bottom: 25px;
        }
        
        .stat-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s;
        }
        
        .stat-card:hover {
            background: rgba(255, 255, 255, 0.08);
            transform: translateY(-2px);
        }
        
        .stat-label {
            font-size: 12px;
            color: #9fa8da;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
            display: block;
        }
        
        .stat-value {
            font-size: 28px;
            font-weight: 700;
            color: #00d4aa;
        }
        
        .stat-value.buy {
            color: #4caf50;
        }
        
        .stat-value.sell {
            color: #f44336;
        }
        
        .stat-value.ready {
            color: #ffd700;
        }
        
        .filters {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .filter-row {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            align-items: center;
        }
        
        .filter-btn {
            padding: 8px 20px;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 20px;
            color: #e8eaf6;
            font-size: 13px;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .filter-btn.active {
            background: linear-gradient(135deg, #00d4aa 0%, #00a8ff 100%);
            color: #0a0e27;
            border-color: transparent;
        }
        
        .filter-btn:hover:not(.active) {
            background: rgba(255, 255, 255, 0.12);
        }
        
        .table-container {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 25px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            overflow-x: auto;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            min-width: 1200px;
        }
        
        thead tr {
            border-bottom: 2px solid rgba(255, 255, 255, 0.1);
        }
        
        th {
            padding: 15px 12px;
            text-align: left;
            font-size: 12px;
            color: #9fa8da;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
            white-space: nowrap;
        }
        
        td {
            padding: 15px 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            font-size: 13px;
            white-space: nowrap;
        }
        
        tbody tr {
            transition: all 0.2s;
        }
        
        tbody tr:hover {
            background: rgba(255, 255, 255, 0.05);
        }
        
        .signal-buy {
            color: #4caf50;
            font-weight: 700;
        }
        
        .signal-sell {
            color: #f44336;
            font-weight: 700;
        }
        
        .signal-neutral {
            color: #9fa8da;
        }
        
        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .badge-ready {
            background: rgba(255, 215, 0, 0.2);
            color: #ffd700;
            border: 1px solid rgba(255, 215, 0, 0.3);
        }
        
        .badge-high {
            background: rgba(76, 175, 80, 0.2);
            color: #4caf50;
            border: 1px solid rgba(76, 175, 80, 0.3);
        }
        
        .badge-medium {
            background: rgba(255, 152, 0, 0.2);
            color: #ff9800;
            border: 1px solid rgba(255, 152, 0, 0.3);
        }
        
        .badge-low {
            background: rgba(244, 67, 54, 0.2);
            color: #f44336;
            border: 1px solid rgba(244, 67, 54, 0.3);
        }
        
        .tf-badge-small {
            padding: 2px 8px;
            border-radius: 8px;
            font-size: 10px;
            font-weight: 600;
        }
        
        .tf-bullish {
            background: rgba(76, 175, 80, 0.15);
            color: #4caf50;
            border: 1px solid rgba(76, 175, 80, 0.3);
        }
        
        .tf-bearish {
            background: rgba(244, 67, 54, 0.15);
            color: #f44336;
            border: 1px solid rgba(244, 67, 54, 0.3);
        }
        
        .tf-neutral {
            background: rgba(159, 168, 218, 0.15);
            color: #9fa8da;
            border: 1px solid rgba(159, 168, 218, 0.3);
        }
        
        .pattern-badge {
            padding: 2px 8px;
            background: rgba(0, 212, 170, 0.15);
            color: #00d4aa;
            border-radius: 8px;
            font-size: 10px;
            margin: 1px;
            display: inline-block;
        }
        
        .status {
            margin: 15px 0;
            padding: 12px 20px;
            background: rgba(0, 212, 170, 0.1);
            border-left: 3px solid #00d4aa;
            border-radius: 8px;
            font-size: 14px;
            color: #00d4aa;
            display: none;
        }
        
        .loading {
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2px solid rgba(0, 212, 170, 0.3);
            border-top-color: #00d4aa;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            vertical-align: middle;
            margin-right: 8px;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        @media (max-width: 768px) {
            .control-row, .filter-row {
                flex-direction: column;
                align-items: stretch;
            }
            
            .stats-grid {
                grid-template-columns: 1fr;
            }
            
            select, input {
                min-width: 100%;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Advanced Stock Scanner V5.1</h1>
            <p class="subtitle">Multi-Timeframe Rule-Based Analysis | No ML | Professional Risk Management</p>
            <span class="version-badge">Fixed yfinance v5.1</span>
            
            <div class="timeframe-info">
                <div class="tf-badge">
                    <span class="label">HTF (1H):</span> Trend Permission
                </div>
                <div class="tf-badge">
                    <span class="label">MTF (15M):</span> Structure Confirmation
                </div>
                <div class="tf-badge">
                    <span class="label">LTF (5M):</span> Entry Timing
                </div>
            </div>
        </div>
        
        <div class="controls">
            <div class="control-row">
                <div class="control-group">
                    <label>Risk Method</label>
                    <select id="method">
                        <option value="ATR">ATR-Based (Recommended)</option>
                        <option value="EMA">EMA-Based</option>
                    </select>
                </div>
                
                <div class="control-group" style="flex: 1;">
                    <label>Custom Symbols (Optional, comma separated)</label>
                    <input type="text" id="customSymbols" placeholder="e.g., RELIANCE.NS, TCS.NS, INFY.NS (Leave empty for NIFTY 50)">
                </div>
            </div>
            
            <div class="control-row">
                <button class="btn btn-primary" onclick="startScan()">
                    <span id="scanBtn">🔍 Start Multi-Timeframe Scan</span>
                </button>
                <button class="btn btn-secondary" onclick="loadStats()">
                    📊 Refresh Statistics
                </button>
            </div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <span class="stat-label">Total Scanned</span>
                <span class="stat-value" id="total">0</span>
            </div>
            <div class="stat-card">
                <span class="stat-label">Buy Signals (HTF Bullish)</span>
                <span class="stat-value buy" id="buy">0</span>
            </div>
            <div class="stat-card">
                <span class="stat-label">Sell Signals (HTF Bearish)</span>
                <span class="stat-value sell" id="sell">0</span>
            </div>
            <div class="stat-card">
                <span class="stat-label">Trade Ready (Score ≥ 70)</span>
                <span class="stat-value ready" id="ready">0</span>
            </div>
        </div>
        
        <div class="filters">
            <div class="filter-row">
                <span style="color: #9fa8da; font-size: 13px; font-weight: 600;">FILTER SIGNALS:</span>
                <button class="filter-btn active" data-filter="all" onclick="filterResults('all')">All Signals</button>
                <button class="filter-btn" data-filter="buy" onclick="filterResults('buy')">Buy Only (HTF Bullish)</button>
                <button class="filter-btn" data-filter="sell" onclick="filterResults('sell')">Sell Only (HTF Bearish)</button>
                <button class="filter-btn" data-filter="ready" onclick="filterResults('ready')">Trade Ready Only</button>
                <button class="filter-btn" data-filter="high-score" onclick="filterResults('high-score')">High Score (≥80)</button>
            </div>
        </div>
        
        <div id="status" class="status">
            <span class="loading"></span> Scanning across multiple timeframes (HTF 1H, MTF 15M, LTF 5M)...
        </div>
        
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>LTP</th>
                        <th>Signal</th>
                        <th>Confidence</th>
                        <th>Entry</th>
                        <th>Stop Loss</th>
                        <th>Target</th>
                        <th>R:R</th>
                        <th>Timeframe Analysis</th>
                        <th>Candle Patterns</th>
                        <th>Indicators</th>
                        <th>Trade Ready</th>
                    </tr>
                </thead>
                <tbody id="tbody">
                    <tr>
                        <td colspan="12" style="text-align: center; padding: 40px; color: #9fa8da;">
                            Click "Start Multi-Timeframe Scan" to begin analysis
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
    
    <script>
        let allResults = [];
        let currentFilter = 'all';
        
        async function startScan() {
            const method = document.getElementById('method').value;
            const raw = document.getElementById('customSymbols').value.trim();
            const symbols = raw ? raw.split(/[\\s,]+/).map(s => s.trim()).filter(Boolean) : null;
            
            const scanBtn = document.getElementById('scanBtn');
            scanBtn.innerHTML = '<span class="loading"></span> Scanning...';
            document.getElementById('status').style.display = 'block';
            
            try {
                const resp = await fetch('/scan', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({method, symbols})
                });
                
                const data = await resp.json();
                allResults = data.results || [];
                
                scanBtn.textContent = '🔍 Start Multi-Timeframe Scan';
                document.getElementById('status').style.display = 'none';
                
                updateStats(allResults);
                renderResults(allResults);
                
            } catch (error) {
                scanBtn.textContent = '🔍 Start Multi-Timeframe Scan';
                document.getElementById('status').style.display = 'none';
                alert('Scan failed: ' + error.message);
            }
        }
        
        function updateStats(results) {
            const buy = results.filter(r => r.signal === "BUY").length;
            const sell = results.filter(r => r.signal === "SELL").length;
            const ready = results.filter(r => r.trade_ready).length;

            document.getElementById('total').textContent = results.length;
            document.getElementById('buy').textContent = buy;
            document.getElementById('sell').textContent = sell;
            document.getElementById('ready').textContent = ready;
        }
        
        function renderResults(results) {
            const tbody = document.getElementById('tbody');
            tbody.innerHTML = "";

            if (!results.length) {
                tbody.innerHTML = `
                    <tr>
                        <td colspan="12" style="text-align: center; padding: 40px; color: #9fa8da;">
                            No results found. Try changing your criteria.
                        </td>
                    </tr>`;
                return;
            }

            results.forEach(r => {
                const signalClass = r.signal === "BUY" ? "signal-buy"
                                  : r.signal === "SELL" ? "signal-sell"
                                  : "signal-neutral";
                
                const scoreBadgeClass = r.confidence_score >= 80 ? 'badge-high'
                                      : r.confidence_score >= 70 ? 'badge-medium'
                                      : 'badge-low';
                
                const htfClass = r.htf_bias === "BULLISH" ? "tf-bullish"
                               : r.htf_bias === "BEARISH" ? "tf-bearish"
                               : "tf-neutral";
                               
                const mtfClass = r.mtf_structure === "BULLISH" ? "tf-bullish"
                               : r.mtf_structure === "BEARISH" ? "tf-bearish"
                               : "tf-neutral";
                               
                const ltfClass = r.ltf_entry.includes("BULLISH") ? "tf-bullish"
                               : r.ltf_entry.includes("BEARISH") ? "tf-bearish"
                               : "tf-neutral";
                
                const patterns = r.candle_pattern && r.candle_pattern.length
                    ? r.candle_pattern.map(p => `<span class="pattern-badge">${p.replace('_', ' ')}</span>`).join(' ')
                    : "-";
                
                const indicators = `
                    RSI: ${r.rsi} | 
                    ADX: ${r.adx} | 
                    ATR: ${r.atr} |
                    MACD: ${r.macd_hist > 0 ? '+' : ''}${r.macd_hist}
                `;
                
                tbody.innerHTML += `
                    <tr>
                        <td><strong>${r.symbol}</strong></td>
                        <td>${r.ltp.toFixed(2)}</td>
                        <td class="${signalClass}">${r.signal}</td>
                        <td>
                            <span class="badge ${scoreBadgeClass}">${r.confidence_score}%</span>
                        </td>
                        <td>${r.entry.toFixed(2)}</td>
                        <td>${r.stop_loss.toFixed(2)}</td>
                        <td>${r.target.toFixed(2)}</td>
                        <td>${r.rr_ratio}</td>
                        <td>
                            <div style="display: flex; gap: 5px; flex-wrap: wrap;">
                                <span class="tf-badge-small ${htfClass}" title="High Timeframe (1H)">HTF: ${r.htf_bias}</span>
                                <span class="tf-badge-small ${mtfClass}" title="Medium Timeframe (15M)">MTF: ${r.mtf_structure}</span>
                                <span class="tf-badge-small ${ltfClass}" title="Low Timeframe (5M)">LTF: ${r.ltf_entry}</span>
                            </div>
                        </td>
                        <td>${patterns}</td>
                        <td style="font-size: 11px; color: #9fa8da;">${indicators}</td>
                        <td>
                            ${r.trade_ready 
                                ? '<span class="badge badge-ready">READY</span>' 
                                : '<span style="color: #9fa8da; font-size: 11px;">Not Ready</span>'}
                        </td>
                    </tr>
                `;
            });
        }
        
        function filterResults(type) {
            currentFilter = type;

            document.querySelectorAll(".filter-btn")
                .forEach(btn => btn.classList.remove("active"));

            document.querySelector(`.filter-btn[data-filter="${type}"]`)
                .classList.add("active");

            let filtered = [...allResults];

            if (type === "buy") {
                filtered = filtered.filter(r => r.signal === "BUY");
            }
            else if (type === "sell") {
                filtered = filtered.filter(r => r.signal === "SELL");
            }
            else if (type === "ready") {
                filtered = filtered.filter(r => r.trade_ready);
            }
            else if (type === "high-score") {
                filtered = filtered.filter(r => r.confidence_score >= 80);
            }

            renderResults(filtered);
            updateStats(filtered);
        }
        
        async function loadStats() {
            try {
                const resp = await fetch("/stats");
                const data = await resp.json();

                document.getElementById("total").textContent = data.total_scanned;
                document.getElementById("buy").textContent = data.buy_signals;
                document.getElementById("sell").textContent = data.sell_signals;
                document.getElementById("ready").textContent = data.trade_ready;
                
                // Show notification
                document.getElementById('status').style.display = 'block';
                document.getElementById('status').innerHTML = `
                    <span class="loading"></span> 
                    Stats updated: ${data.buy_signals} BUY, ${data.sell_signals} SELL, ${data.trade_ready} Trade Ready
                `;
                
                setTimeout(() => {
                    document.getElementById('status').style.display = 'none';
                }, 3000);
                
            } catch (err) {
                alert("Failed to load stats: " + err.message);
            }
        }
        
        // Load initial stats
        window.onload = loadStats;
        
    </script>
</body>
</html>
"""

# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🚀 Starting Multi-Timeframe Stock Scanner V5.1")
    logger.info("=" * 60)
    logger.info(f"Timeframes: HTF={HTF_INTERVAL}, MTF={MTF_INTERVAL}, LTF={LTF_INTERVAL}")
    logger.info(f"Periods adjusted for yfinance limitations")
    logger.info(f"Rules: HTF trend permission required")
    logger.info(f"Scoring: Confidence-based (70+ = Trade Ready)")
    logger.info(f"Parallel Workers: {MAX_WORKERS}")
    logger.info("=" * 60)
    
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
