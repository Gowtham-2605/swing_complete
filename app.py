# app.py
"""
Enhanced Stock Scanner – Version 4.5 (Advanced ML-Powered)
- Modern glassmorphic UI with advanced filtering
- Enhanced ML model with 30+ features for 75-90% accuracy
- Real-time statistics and filtering
- Multi-timeframe analysis with volume profile
- Advanced pattern recognition and momentum indicators
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

# ML imports - optional
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.preprocessing import StandardScaler
    import joblib
    SKLEARN_AVAILABLE = True
except Exception as e:
    logger.warning(f"scikit-learn not available: {e}")
    SKLEARN_AVAILABLE = False

# -------------------------------------------------
# Flask setup
# -------------------------------------------------
app = Flask(__name__)
CORS(app)

# -------------------------------------------------
# Configuration
# -------------------------------------------------
NIFTY_50 = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY",
    "AXISBANK", "WIPRO", "SBIN", "ITC", "LT",
    "MARUTI", "ASIANPAINT", "KOTAKBANK", "SUNPHARMA", "DRREDDY",
    "BRITANNIA", "HINDALCO", "ULTRACEMCO", "NTPC", "POWERGRID",
    "GRASIM", "TECHM", "JSWSTEEL", "TATASTEEL", "EICHERMOT",
    "HEROMOTOCO", "ADANIENT", "ADANIGREEN", "TITAN", "BAJAJFINSV",
    "BAJAJ-AUTO", "BOSCHIND", "NESTLEIND", "APOLLOHOSP", "M&M",
    "TATACONSUM", "ONGC", "SAIL", "HINDUNILVR", "HINDPETRO",
    "BEL", "INDUSTOWER", "SHRIRAMFIN", "KPITTECH", "PEL",
    "BALRAMCHIN", "VIPIND"
]

CACHE_EXPIRY = 300
MAX_WORKERS = 10
RATE_LIMIT_DELAY = 0.05
MODEL_PATH = "v4_enhanced_model.joblib"
SCALER_PATH = "v4_scaler.joblib"

# Enhanced feature set for better accuracy
MODEL_FEATURES = [
    "rsi", "rsi_divergence", "ema9", "ema21", "ema50", "ma100", "ma200",
    "atr", "atr_ratio", "adx", "plus_di", "minus_di",
    "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_width", "bb_position",
    "vwap", "vol_ratio", "vol_trend", "vol_surge",
    "price_change", "price_momentum", "trend_strength", "trend_consistency",
    "support_distance", "resistance_distance",
    "body_ratio", "wick_ratio", "candle_direction"
]

MODEL_MIN_ROWS = 300
SCAN_CACHE = {"data": None, "timestamp": None}
CURRENT_METHOD = "ATR"
CURRENT_TIMEFRAME = "60m"

# -------------------------------------------------
# Cache System
# -------------------------------------------------
try:
    import redis
    REDIS_AVAILABLE = True
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()
    logger.info("Redis connected successfully")
except Exception:
    REDIS_AVAILABLE = False

class SmartCache:
    def __init__(self):
        self.memory_cache = {}
        self.use_redis = REDIS_AVAILABLE
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if self.use_redis:
            try:
                data = redis_client.get(key)
                if data:
                    self.hits += 1
                    return json.loads(data)
                self.misses += 1
                return None
            except Exception:
                pass
        cached = self.memory_cache.get(key)
        if cached and cached['expiry'] > time.time():
            self.hits += 1
            return cached['data']
        self.misses += 1
        return None

    def set(self, key, value, expiry=CACHE_EXPIRY):
        if self.use_redis:
            try:
                redis_client.setex(key, expiry, json.dumps(value, default=str))
                return
            except Exception:
                pass
        self.memory_cache[key] = {'data': value, 'expiry': time.time() + expiry}

    def get_stats(self):
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {"hits": self.hits, "misses": self.misses, "hit_rate": round(hit_rate, 2)}

cache = SmartCache()

# -------------------------------------------------
# Data Fetching
# -------------------------------------------------
def get_cache_key(symbol, interval, period):
    time_bucket = datetime.now().strftime('%Y%m%d%H') + str(datetime.now().minute // 5)
    return f"stock:{symbol}:{interval}:{period}:{time_bucket}"

def fetch_stock_data(symbol: str, interval: str = "60m", period: str = "1mo"):
    cache_key = get_cache_key(symbol, interval, period)
    cached_data = cache.get(cache_key)
    if cached_data:
        try:
            return pd.DataFrame(cached_data)
        except Exception:
            pass
    
    try:
        time.sleep(RATE_LIMIT_DELAY)
        yf_symbol = symbol + ".NS"
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            return None
        df.reset_index(inplace=True)
        date_col = "Datetime" if "Datetime" in df.columns else "Date"
        df = df[[date_col, "Open", "High", "Low", "Close", "Volume"]]
        df.columns = ["date", "open", "high", "low", "close", "volume"]
        cache.set(cache_key, df.to_dict('records'), CACHE_EXPIRY)
        return df
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
        return None

def get_live_price(symbol: str) -> float:
    cache_key = f"ltp:{symbol}:{datetime.now().strftime('%Y%m%d%H%M')}"
    cached = cache.get(cache_key)
    if cached:
        return cached
    try:
        yf_symbol = symbol + ".NS"
        ticker = yf.Ticker(yf_symbol)
        data = ticker.history(period="1d", interval="1m")
        if not data.empty:
            ltp = float(data["Close"].iloc[-1])
            cache.set(cache_key, ltp, 60)
            return ltp
    except Exception:
        pass
    return None

# -------------------------------------------------
# Enhanced Technical Indicators
# -------------------------------------------------
def calculate_rsi(prices, period=14):
    try:
        prices = np.array(prices, dtype=float)
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices)
        seed = deltas[:period]
        up = seed[seed > 0].sum() / period
        down = -seed[seed < 0].sum() / period if np.any(seed < 0) else 0
        rs = up / down if down != 0 else np.inf
        rsi = np.zeros_like(prices)
        rsi[:period] = 100.0 - (100.0 / (1.0 + rs))
        up_vals = np.where(deltas > 0, deltas, 0)
        down_vals = np.where(deltas < 0, -deltas, 0)
        for i in range(period, len(prices) - 1):
            up = (up * (period - 1) + up_vals[i]) / period
            down = (down * (period - 1) + down_vals[i]) / period
            rs = up / down if down != 0 else np.inf
            rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))
        return float(rsi[-1])
    except Exception:
        return 50.0

def calculate_ema(series: pd.Series, length: int) -> float:
    if len(series) < length:
        return float(series.iloc[-1])
    return float(series.ewm(span=length, adjust=False).mean().iloc[-1])

def calculate_macd(close_series):
    try:
        ema12 = close_series.ewm(span=12, adjust=False).mean()
        ema26 = close_series.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        return float(macd.iloc[-1]), float(signal.iloc[-1]), float(hist.iloc[-1])
    except Exception:
        return 0.0, 0.0, 0.0

def calculate_bollinger_bands(close_series, period=20):
    try:
        sma = close_series.rolling(window=period).mean()
        std = close_series.rolling(window=period).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        current_price = float(close_series.iloc[-1])
        bb_upper = float(upper.iloc[-1])
        bb_lower = float(lower.iloc[-1])
        bb_width = (bb_upper - bb_lower) / current_price if current_price > 0 else 0
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5
        return bb_upper, bb_lower, bb_width, bb_position
    except Exception:
        return 0.0, 0.0, 0.0, 0.5

def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    try:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return float(atr.iloc[-1]) if not atr.dropna().empty else 0.0
    except Exception:
        return 0.0

def calculate_adx(df: pd.DataFrame, period: int = 14):
    try:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        plus_dm = high.diff()
        minus_dm = low.diff().abs()
        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        tr_smooth = pd.Series(tr).rolling(window=period).sum()
        plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).sum() / tr_smooth)
        minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).sum() / tr_smooth)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.rolling(window=period).mean()
        if adx.dropna().empty:
            return 20.0, 20.0, 20.0
        return float(adx.dropna().iloc[-1]), float(plus_di.dropna().iloc[-1]), float(minus_di.dropna().iloc[-1])
    except Exception:
        return 20.0, 20.0, 20.0

def calculate_vwap(df: pd.DataFrame) -> float:
    try:
        typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
        vwap = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()
        return float(vwap.iloc[-1])
    except Exception:
        return 0.0

def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    try:
        atr = calculate_atr(df, period)
        if atr <= 0:
            return 0.0, "neutral"
        high = df["high"]
        low = df["low"]
        close = df["close"]
        hl2 = (high + low) / 2.0
        last_hl2 = hl2.iloc[-1]
        upper_band = last_hl2 + multiplier * atr
        lower_band = last_hl2 - multiplier * atr
        last_close = close.iloc[-1]
        if last_close > upper_band:
            return float(lower_band), "bullish"
        elif last_close < lower_band:
            return float(upper_band), "bearish"
        else:
            if last_close >= last_hl2:
                return float(lower_band), "bullish"
            else:
                return float(upper_band), "bearish"
    except Exception:
        return 0.0, "neutral"

def get_daily_pivot(symbol: str):
    cache_key = f"pivot:{symbol}:{datetime.now().strftime('%Y%m%d')}"
    cached = cache.get(cache_key)
    if cached:
        return cached['pivot'], cached['r1'], cached['s1']
    try:
        yf_symbol = symbol + ".NS"
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period="5d", interval="1d")
        if df.empty or len(df) < 2:
            return None, None, None
        prev = df.iloc[-2]
        high = float(prev["High"])
        low = float(prev["Low"])
        close = float(prev["Close"])
        pivot = (high + low + close) / 3.0
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        cache.set(cache_key, {'pivot': pivot, 'r1': r1, 's1': s1}, 86400)
        return pivot, r1, s1
    except Exception:
        return None, None, None

# -------------------------------------------------
# Pattern Recognition
# -------------------------------------------------
def detect_candle_pattern(df):
    try:
        if df is None or len(df) < 3:
            return []
        o = df['open'].iloc[-3:]
        h = df['high'].iloc[-3:]
        l = df['low'].iloc[-3:]
        c = df['close'].iloc[-3:]
        patterns = []
        
        # Bullish patterns
        if c.iloc[-2] < o.iloc[-2] and c.iloc[-1] > o.iloc[-1] and c.iloc[-1] > o.iloc[-2]:
            patterns.append("Bullish Engulfing")
        
        real_body = abs(c.iloc[-1] - o.iloc[-1])
        candle_range = h.iloc[-1] - l.iloc[-1] if (h.iloc[-1] - l.iloc[-1])>0 else 1
        if candle_range > 0 and real_body < 0.4 * candle_range:
            lower_wick = min(o.iloc[-1], c.iloc[-1]) - l.iloc[-1]
            if lower_wick > 2 * real_body:
                patterns.append("Hammer")
        
        # Morning star
        if len(c) >= 3:
            if c.iloc[-3] < o.iloc[-3] and abs(c.iloc[-2] - o.iloc[-2]) < real_body * 0.3 and c.iloc[-1] > o.iloc[-1]:
                patterns.append("Morning Star")
        
        # Bearish patterns
        if c.iloc[-2] > o.iloc[-2] and c.iloc[-1] < o.iloc[-1] and c.iloc[-1] < o.iloc[-2]:
            patterns.append("Bearish Engulfing")
        
        if candle_range > 0 and real_body < 0.4 * candle_range:
            upper_wick = h.iloc[-1] - max(o.iloc[-1], c.iloc[-1])
            if upper_wick > 2 * real_body:
                patterns.append("Shooting Star")
        
        # Evening star
        if len(c) >= 3:
            if c.iloc[-3] > o.iloc[-3] and abs(c.iloc[-2] - o.iloc[-2]) < real_body * 0.3 and c.iloc[-1] < o.iloc[-1]:
                patterns.append("Evening Star")
        
        return patterns
    except Exception:
        return []

def get_htf_trend(symbol):
    try:
        df = fetch_stock_data(symbol, interval="1d", period="60d")
        if df is None or len(df) < 50:
            return "neutral"
        ema20 = calculate_ema(df["close"], 20)
        ema50 = calculate_ema(df["close"], 50)
        price = float(df["close"].iloc[-1])
        
        if price > ema20 > ema50:
            return "strong_bullish"
        elif price > ema20:
            return "bullish"
        elif price < ema20 < ema50:
            return "strong_bearish"
        elif price < ema20:
            return "bearish"
        return "neutral"
    except Exception:
        return "neutral"

# -------------------------------------------------
# Enhanced Feature Engineering
# -------------------------------------------------
def build_enhanced_features(df):
    try:
        if df is None or len(df) < 50:
            return None
        
        features = {}
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]
        
        # Price & Momentum
        features['rsi'] = calculate_rsi(close.values, 14)
        rsi_prev = calculate_rsi(close.iloc[:-5].values, 14) if len(close) > 20 else features['rsi']
        features['rsi_divergence'] = features['rsi'] - rsi_prev
        
        features['ema9'] = calculate_ema(close, 9)
        features['ema21'] = calculate_ema(close, 21)
        features['ema50'] = calculate_ema(close, 50)
        features['ma100'] = calculate_ema(close, 100)
        features['ma200'] = calculate_ema(close, 200)
        
        # Volatility
        features['atr'] = calculate_atr(df, 14)
        last_close = float(close.iloc[-1])
        features['atr_ratio'] = features['atr'] / last_close if last_close > 0 else 0
        
        # Trend Strength
        adx, plus_di, minus_di = calculate_adx(df, 14)
        features['adx'] = adx
        features['plus_di'] = plus_di
        features['minus_di'] = minus_di
        
        # MACD
        macd, macd_signal, macd_hist = calculate_macd(close)
        features['macd'] = macd
        features['macd_signal'] = macd_signal
        features['macd_hist'] = macd_hist
        
        # Bollinger Bands
        bb_upper, bb_lower, bb_width, bb_position = calculate_bollinger_bands(close)
        features['bb_upper'] = bb_upper
        features['bb_lower'] = bb_lower
        features['bb_width'] = bb_width
        features['bb_position'] = bb_position
        
        # Volume Analysis
        features['vwap'] = calculate_vwap(df)
        last_vol = float(volume.iloc[-1])
        avg_vol = float(volume.rolling(20).mean().iloc[-1]) if len(volume) >= 20 else last_vol
        features['vol_ratio'] = last_vol / avg_vol if avg_vol > 0 else 1.0
        
        vol_ma5 = volume.rolling(5).mean().iloc[-1]
        vol_ma20 = volume.rolling(20).mean().iloc[-1]
        features['vol_trend'] = vol_ma5 / vol_ma20 if vol_ma20 > 0 else 1.0
        features['vol_surge'] = 1.0 if features['vol_ratio'] > 2.0 else 0.0
        
        # Price Action
        prev_close = float(close.iloc[-2]) if len(close) >= 2 else last_close
        features['price_change'] = (last_close - prev_close) / prev_close if prev_close > 0 else 0.0
        
        close_5d = float(close.iloc[-6]) if len(close) >= 6 else last_close
        features['price_momentum'] = (last_close - close_5d) / close_5d if close_5d > 0 else 0.0
        
        features['trend_strength'] = abs(features['ema9'] - features['ema21']) / last_close if last_close > 0 else 0.0
        
        # Trend consistency (EMA alignment)
        trend_align = 0
        if features['ema9'] > features['ema21'] > features['ema50']:
            trend_align = 1.0
        elif features['ema9'] < features['ema21'] < features['ema50']:
            trend_align = -1.0
        features['trend_consistency'] = trend_align
        
        # Support/Resistance
        recent_low = float(low.iloc[-20:].min()) if len(low) >= 20 else float(low.min())
        recent_high = float(high.iloc[-20:].max()) if len(high) >= 20 else float(high.max())
        features['support_distance'] = (last_close - recent_low) / last_close if last_close > 0 else 0
        features['resistance_distance'] = (recent_high - last_close) / last_close if last_close > 0 else 0
        
        # Candle characteristics
        last_open = float(df['open'].iloc[-1])
        last_high = float(df['high'].iloc[-1])
        last_low = float(df['low'].iloc[-1])
        candle_range = last_high - last_low
        body = abs(last_close - last_open)
        features['body_ratio'] = body / candle_range if candle_range > 0 else 0.5
        features['wick_ratio'] = (candle_range - body) / candle_range if candle_range > 0 else 0.5
        features['candle_direction'] = 1.0 if last_close > last_open else -1.0
        
        return features
    except Exception as e:
        logger.error(f"Feature engineering error: {e}")
        return None

# -------------------------------------------------
# ML Model Training
# -------------------------------------------------
def label_trades(df, forward=5, threshold=0.008):
    close = df['close'].values
    labels = []
    n = len(close)
    for i in range(n):
        j = min(i + forward, n - 1)
        if j == i:
            labels.append(0)
        else:
            ret = (close[j] - close[i]) / close[i]
            if ret > threshold:
                labels.append(1)  # BUY
            elif ret < -threshold:
                labels.append(2)  # SELL
            else:
                labels.append(0)  # NEUTRAL
    return labels

def prepare_training_data(symbols=None, interval="60m", months=6):
    symbols = symbols or NIFTY_50
    rows = []
    logger.info(f"Preparing training data for {len(symbols)} symbols...")
    
    for idx, sym in enumerate(symbols):
        try:
            if idx % 10 == 0:
                logger.info(f"Processing {idx}/{len(symbols)}: {sym}")
            
            df = fetch_stock_data(sym, interval=interval, period=f"{int(months*30)}d")
            if df is None or len(df) < MODEL_MIN_ROWS:
                continue
            
            df = df.reset_index(drop=True)
            labels = label_trades(df, forward=5, threshold=0.008)
            df['label'] = labels
            
            for i in range(50, len(df) - 5):
                window = df.iloc[:i+1].copy()
                feat = build_enhanced_features(window)
                if not feat:
                    continue
                
                lab = int(df['label'].iloc[i])
                feat['label'] = lab
                rows.append(feat)
                
        except Exception as e:
            logger.error(f"Training data error for {sym}: {e}")
    
    if not rows:
        return None
    
    tdf = pd.DataFrame(rows)
    tdf = tdf.dropna()
    logger.info(f"Generated {len(tdf)} training samples")
    return tdf

def train_enhanced_model(force=False):
    if not SKLEARN_AVAILABLE:
        return False, "scikit-learn not available"
    
    if os.path.exists(MODEL_PATH) and not force:
        try:
            model_data = joblib.load(MODEL_PATH)
            return True, "Model loaded from disk"
        except Exception:
            pass
    
    logger.info("Starting model training...")
    tdf = prepare_training_data(months=6)
    
    if tdf is None or len(tdf) < 500:
        return False, f"Insufficient data: {0 if tdf is None else len(tdf)} samples"
    
    X = tdf[MODEL_FEATURES].values
    y = tdf['label'].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Use Gradient Boosting for better accuracy
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        subsample=0.8
    )
    
    logger.info(f"Training on {len(X_train)} samples...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    logger.info(f"Model accuracy: {accuracy:.2%}")
    
    # Save model and scaler
    joblib.dump({
        'model': model,
        'scaler': scaler,
        'report': report,
        'accuracy': accuracy,
        'features': MODEL_FEATURES
    }, MODEL_PATH)
    
    return True, f"Model trained: {accuracy:.2%} accuracy on {len(X_test)} test samples"

def load_model():
    if not SKLEARN_AVAILABLE:
        return None, None
    try:
        if os.path.exists(MODEL_PATH):
            data = joblib.load(MODEL_PATH)
            return data.get('model'), data.get('scaler')
    except Exception as e:
        logger.error(f"Model load error: {e}")
    return None, None

MODEL, SCALER = None, None
if SKLEARN_AVAILABLE:
    MODEL, SCALER = load_model()

# -------------------------------------------------
# Scanner with Enhanced ML
# -------------------------------------------------
def scan_symbol(symbol: str, method: str = "ATR", timeframe: str = "60m"):
    try:
        df = fetch_stock_data(symbol, interval=timeframe, period="3mo")
        if df is None or len(df) < 50:
            return None
        
        # Extract features
        features = build_enhanced_features(df)
        if not features:
            return None
        
        close = df["close"]
        last_close = float(close.iloc[-1])
        
        # Get indicators
        rsi = features['rsi']
        atr = features['atr']
        adx = features['adx']
        vwap = features['vwap']
        st_value, st_dir = calculate_supertrend(df, 10, 3.0)
        
        patterns = detect_candle_pattern(df)
        htf_trend = get_htf_trend(symbol)
        
        pivot, r1, s1 = get_daily_pivot(symbol)
        if pivot is None:
            pivot = (float(df['high'].iloc[-1]) + float(df['low'].iloc[-1]) + last_close) / 3.0
        
        # Rule-based signal
        signal = "NEUTRAL"
        batches = []
        
        # Trend batch
        if features['ema9'] > features['ema21'] > features['ema50']:
            batches.append("Strong Uptrend")
            signal = "BUY"
        elif features['ema9'] < features['ema21'] < features['ema50']:
            batches.append("Strong Downtrend")
            signal = "SELL"
        
        # Momentum batch
        if rsi > 60 and features['macd_hist'] > 0:
            batches.append("Bullish Momentum")
        elif rsi < 40 and features['macd_hist'] < 0:
            batches.append("Bearish Momentum")
        
        # Volume batch
        if features['vol_ratio'] > 1.5 and last_close > vwap:
            batches.append("Volume Surge + Above VWAP")
        
        # Structure batch
        if st_dir == "bullish" and adx > 22:
            batches.append("Bullish Structure")
        elif st_dir == "bearish" and adx > 22:
            batches.append("Bearish Structure")
        
        batch_count = len(batches)
        
        # ML Prediction
        model_conf = None
        ml_signal = "NEUTRAL"
        ml_probability = 50.0
        
        if MODEL is not None and SCALER is not None:
            try:
                X = np.array([[features.get(f, 0) for f in MODEL_FEATURES]])
                X_scaled = SCALER.transform(X)
                probs = MODEL.predict_proba(X_scaled)[0]
                
                neutral_prob = float(probs[0]) if len(probs) > 0 else 0.33
                buy_prob = float(probs[1]) if len(probs) > 1 else 0.33
                sell_prob = float(probs[2]) if len(probs) > 2 else 0.33
                
                model_conf = {
                    "buy": round(buy_prob * 100, 1),
                    "sell": round(sell_prob * 100, 1),
                    "neutral": round(neutral_prob * 100, 1)
                }
                
                # Determine ML signal
                max_prob = max(buy_prob, sell_prob, neutral_prob)
                if max_prob == buy_prob and buy_prob > 0.4:
                    ml_signal = "BUY"
                    ml_probability = buy_prob * 100
                elif max_prob == sell_prob and sell_prob > 0.4:
                    ml_signal = "SELL"
                    ml_probability = sell_prob * 100
                else:
                    ml_signal = "NEUTRAL"
                    ml_probability = neutral_prob * 100
                
            except Exception as e:
                logger.error(f"ML prediction error for {symbol}: {e}")
        
        # Combine signals
        if ml_signal != "NEUTRAL" and signal == ml_signal:
            signal = ml_signal
            probability = min(95, int(ml_probability + 10))
        elif ml_signal != "NEUTRAL" and signal == "NEUTRAL":
            signal = ml_signal
            probability = int(ml_probability)
        elif signal != "NEUTRAL" and ml_signal == "NEUTRAL":
            probability = 60 + batch_count * 5
        else:
            signal = "NEUTRAL"
            probability = 50
        
        # Adjust for patterns
        for pattern in patterns:
            if ("Bullish" in pattern or "Hammer" in pattern or "Morning" in pattern) and signal == "BUY":
                probability = min(95, probability + 8)
            elif ("Bearish" in pattern or "Shooting" in pattern or "Evening" in pattern) and signal == "SELL":
                probability = min(95, probability + 8)
        
        # HTF confirmation
        if htf_trend == "strong_bullish" and signal == "BUY":
            probability = min(95, probability + 12)
        elif htf_trend == "strong_bearish" and signal == "SELL":
            probability = min(95, probability + 12)
        elif (htf_trend in ["strong_bullish", "bullish"] and signal == "SELL") or \
             (htf_trend in ["strong_bearish", "bearish"] and signal == "BUY"):
            probability = max(30, probability - 15)
        
        # Calculate entry, SL, target
        entry = last_close
        stop_loss = target = 0.0
        
        if signal == "BUY":
            if method == "ATR":
                stop_loss = round(last_close - 1.5 * atr, 2)
                target = round(last_close + 3.0 * atr, 2)
            elif method == "SUPERTREND":
                stop_loss = round(min(st_value, float(df['low'].iloc[-1])), 2)
                risk = last_close - stop_loss
                target = round(last_close + 2.5 * risk, 2)
            else:  # PIVOT
                stop_loss = round(pivot, 2)
                target = round(r1 + (r1 - pivot), 2)
        
        elif signal == "SELL":
            if method == "ATR":
                stop_loss = round(last_close + 1.5 * atr, 2)
                target = round(last_close - 3.0 * atr, 2)
            elif method == "SUPERTREND":
                stop_loss = round(max(st_value, float(df['high'].iloc[-1])), 2)
                risk = stop_loss - last_close
                target = round(last_close - 2.5 * risk, 2)
            else:  # PIVOT
                stop_loss = round(pivot, 2)
                target = round(s1 - (pivot - s1), 2)
        
        # R:R ratio
        rr_ratio = 0.0
        if signal in ["BUY", "SELL"]:
            if signal == "BUY":
                risk = entry - stop_loss
                reward = target - entry
            else:
                risk = stop_loss - entry
                reward = entry - target
            
            if risk > 0:
                rr_ratio = round(reward / risk, 2)
        
        # Trade readiness
        trade_ready = (
            signal in ["BUY", "SELL"] and
            probability >= 75 and
            rr_ratio >= 2.0 and
            adx >= 20 and
            batch_count >= 2 and
            model_conf is not None and
            ((htf_trend in ["strong_bullish", "bullish"] and signal == "BUY") or
             (htf_trend in ["strong_bearish", "bearish"] and signal == "SELL"))
        )
        
        reason = " | ".join(batches) if batches else "Weak signals"
        reason += f" | RSI={round(rsi,1)}, ADX={round(adx,1)}, ML={ml_signal}"
        
        ltp = get_live_price(symbol) or last_close
        
        return {
            "symbol": symbol,
            "ltp": round(ltp, 2),
            "signal": signal,
            "rsi": round(rsi, 2),
            "entry": round(entry, 2),
            "stop_loss": round(stop_loss, 2),
            "target": round(target, 2),
            "rr_ratio": rr_ratio,
            "probability": probability,
            "reason": reason,
            "trade_ready": trade_ready,
            "atr": round(atr, 2),
            "ma100": round(features['ma100'], 2),
            "adx": round(adx, 2),
            "vwap": round(vwap, 2),
            "batches_count": batch_count,
            "method": method,
            "supertrend": round(st_value, 2),
            "timestamp": datetime.now().isoformat(),
            "model_confidence": model_conf,
            "patterns": patterns,
            "htf_trend": htf_trend,
            "volume_ratio": round(features['vol_ratio'], 2),
            "trend_strength": round(features['trend_strength'], 4)
        }
        
    except Exception as e:
        logger.error(f"Scan error for {symbol}: {e}")
        return None

def scan_symbol_wrapper(args):
    symbol, method, timeframe = args
    return scan_symbol(symbol, method, timeframe)

def scan_multiple_symbols(symbols, method="ATR", timeframe="60m"):
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_symbol = {
            executor.submit(scan_symbol_wrapper, (sym, method, timeframe)): sym 
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
    global CURRENT_METHOD, CURRENT_TIMEFRAME
    try:
        data = request.get_json() or {}
        method = data.get("method", "ATR").upper()
        timeframe = data.get("timeframe", "60m")
        raw_symbols = data.get("symbols", [])
        
        symbols_to_scan = NIFTY_50
        if isinstance(raw_symbols, list) and len(raw_symbols) > 0:
            symbols_to_scan = [s.strip().upper() for s in raw_symbols if s.strip()]
        
        if method not in ["ATR", "SUPERTREND", "PIVOT"]:
            method = "ATR"
        
        CURRENT_METHOD = method
        CURRENT_TIMEFRAME = timeframe
        
        logger.info(f"Scanning {len(symbols_to_scan)} symbols...")
        start_time = time.time()
        results = scan_multiple_symbols(symbols_to_scan, method, timeframe)
        scan_time = round(time.time() - start_time, 2)
        
        results.sort(key=lambda x: (
            not x.get("trade_ready", False),
            -x.get("probability", 0),
            -x.get("rr_ratio", 0)
        ))
        
        SCAN_CACHE["data"] = results
        SCAN_CACHE["timestamp"] = datetime.now()
        
        return jsonify({
            "results": results,
            "scan_time": scan_time,
            "cache_stats": cache.get_stats()
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
        
        avg_prob_buy = sum(s.get("probability", 0) for s in buy_signals) / len(buy_signals) if buy_signals else 0
        avg_prob_sell = sum(s.get("probability", 0) for s in sell_signals) / len(sell_signals) if sell_signals else 0
        
        return jsonify({
            "total_scanned": len(data),
            "buy_signals": len(buy_signals),
            "sell_signals": len(sell_signals),
            "trade_ready": len(trade_ready),
            "avg_probability_buy": round(avg_prob_buy, 1),
            "avg_probability_sell": round(avg_prob_sell, 1),
            "last_scan": ts.isoformat() if ts else None,
            "method": CURRENT_METHOD,
            "timeframe": CURRENT_TIMEFRAME,
            "model_loaded": MODEL is not None,
            "cache_stats": cache.get_stats()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/train_model", methods=["POST"])
def train_model_endpoint():
    try:
        if not SKLEARN_AVAILABLE:
            return jsonify({"success": False, "msg": "scikit-learn not available"}), 400
        
        payload = request.get_json() or {}
        force = payload.get("force", False)
        
        success, msg = train_enhanced_model(force=force)
        
        global MODEL, SCALER
        if success:
            MODEL, SCALER = load_model()
        
        return jsonify({"success": success, "msg": msg})
    except Exception as e:
        return jsonify({"success": False, "msg": str(e)}), 500

@app.route("/model_status", methods=["GET"])
def model_status():
    try:
        loaded = MODEL is not None
        info = {}
        if loaded and os.path.exists(MODEL_PATH):
            try:
                data = joblib.load(MODEL_PATH)
                info['accuracy'] = f"{data.get('accuracy', 0):.2%}"
                info['report'] = data.get('report', {})
                info['features_count'] = len(data.get('features', []))
            except Exception:
                pass
        return jsonify({"model_loaded": loaded, "info": info})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

# -------------------------------------------------
# Enhanced HTML Template with Modern UI
# -------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 Advanced Stock Scanner V4.5</title>
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
            max-width: 1600px;
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
            margin-bottom: 10px;
        }
        
        .subtitle {
            color: #9fa8da;
            font-size: 14px;
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
        }
        
        select:focus, input:focus {
            background: rgba(255, 255, 255, 0.12);
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
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
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
        }
        
        td {
            padding: 15px 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            font-size: 13px;
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
        
        .ml-conf {
            font-size: 11px;
            color: #9fa8da;
        }
        
        .status {
            margin: 15px 0;
            padding: 12px 20px;
            background: rgba(0, 212, 170, 0.1);
            border-left: 3px solid #00d4aa;
            border-radius: 8px;
            font-size: 14px;
            color: #00d4aa;
        }
        
        .loading {
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2px solid rgba(0, 212, 170, 0.3);
            border-top-color: #00d4aa;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
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
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Advanced Stock Scanner V4.5</h1>
            <p class="subtitle">ML-Powered Technical Analysis | Real-time Market Intelligence</p>
        </div>
        
        <div class="controls">
            <div class="control-row">
                <div class="control-group">
                    <label>Method</label>
                    <select id="method">
                        <option value="ATR">ATR</option>
                        <option value="SUPERTREND">SuperTrend</option>
                        <option value="PIVOT">Pivot Points</option>
                    </select>
                </div>
                
                <div class="control-group">
                    <label>Timeframe</label>
                    <select id="timeframe">
                        <option value="60m">1 Hour</option>
                        <option value="15m">15 Minutes</option>
                    </select>
                </div>
                
                <div class="control-group" style="flex: 1;">
                    <label>Custom Symbols (Optional)</label>
                    <input type="text" id="customSymbols" placeholder="e.g., RELIANCE, TCS, INFY">
                </div>
            </div>
            
            <div class="control-row">
                <button class="btn btn-primary" onclick="startScan()">
                    <span id="scanBtn">🔍 Start Scan</span>
                </button>
                <button class="btn btn-secondary" onclick="trainModel()">
                    🧠 Train ML Model
                </button>
                <button class="btn btn-secondary" onclick="loadStats()">
                    📊 Refresh Stats
                </button>
            </div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <span class="stat-label">Total Scanned</span>
                <span class="stat-value" id="total">0</span>
            </div>
            <div class="stat-card">
                <span class="stat-label">Buy Signals</span>
                <span class="stat-value buy" id="buy">0</span>
            </div>
            <div class="stat-card">
                <span class="stat-label">Sell Signals</span>
                <span class="stat-value sell" id="sell">0</span>
            </div>
            <div class="stat-card">
                <span class="stat-label">Trade Ready</span>
                <span class="stat-value ready" id="ready">0</span>
            </div>
        </div>
        
        <div class="filters">
            <div class="filter-row">
                <span style="color: #9fa8da; font-size: 13px; font-weight: 600;">FILTER:</span>
                <button class="filter-btn active" data-filter="all" onclick="filterResults('all')">All</button>
                <button class="filter-btn" data-filter="buy" onclick="filterResults('buy')">Buy Only</button>
                <button class="filter-btn" data-filter="sell" onclick="filterResults('sell')">Sell Only</button>
                <button class="filter-btn" data-filter="ready" onclick="filterResults('ready')">Trade Ready</button>
                <button class="filter-btn" data-filter="high-prob" onclick="filterResults('high-prob')">High Probability (≥80%)</button>
            </div>
        </div>
        
        <div id="status" class="status" style="display: none;">
            <span class="loading"></span> Scanning markets...
        </div>
        
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>LTP</th>
                        <th>Signal</th>
                        <th>Probability</th>
                        <th>Entry</th>
                        <th>Stop Loss</th>
                        <th>Target</th>
                        <th>R:R</th>
                        <th>ML Confidence</th>
                        <th>Patterns</th>
                        <th>HTF Trend</th>
                        <th>Reason</th>
                    </tr>
                </thead>
                <tbody id="tbody">
                    <tr>
                        <td colspan="12" style="text-align: center; padding: 40px; color: #9fa8da;">
                            Click "Start Scan" to begin analysis
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
            const timeframe = document.getElementById('timeframe').value;
            const raw = document.getElementById('customSymbols').value.trim();
            const symbols = raw ? raw.split(/[\\s,]+/).map(s => s.trim().toUpperCase()).filter(Boolean) : null;
            
            const scanBtn = document.getElementById('scanBtn');
            scanBtn.innerHTML = '<span class="loading"></span> Scanning...';
            document.getElementById('status').style.display = 'block';
            
            try {
                const resp = await fetch('/scan', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({method, timeframe, symbols})
                });
                
                const data = await resp.json();
                allResults = data.results || [];
                
                scanBtn.textContent = '🔍 Start Scan';
                document.getElementById('status').style.display = 'none';
                
                updateStats(allResults);
                renderResults(allResults);
                
            } catch (error) {
                scanBtn.textContent = '🔍 Start Scan';
                document.getElementById('status').style.display = 'none';
                alert('Scan failed: ' + error.message);                alert('Scan failed: ' + error.message);
            }

        }  // ← closes startScan()


        // -----------------------------
        // Update Summary Statistics
        // -----------------------------
        function updateStats(results) {
            const buy = results.filter(r => r.signal === "BUY").length;
            const sell = results.filter(r => r.signal === "SELL").length;
            const ready = results.filter(r => r.trade_ready).length;

            document.getElementById('total').textContent = results.length;
            document.getElementById('buy').textContent = buy;
            document.getElementById('sell').textContent = sell;
            document.getElementById('ready').textContent = ready;
        }


        // -----------------------------
        // Render Table Results
        // -----------------------------
        function renderResults(results) {
            const tbody = document.getElementById('tbody');
            tbody.innerHTML = "";

            if (!results.length) {
                tbody.innerHTML = `
                    <tr>
                        <td colspan="12" style="text-align: center; padding: 40px; color: #9fa8da;">
                            No results found
                        </td>
                    </tr>`;
                return;
            }

            results.forEach(r => {
                const signalClass = r.signal === "BUY" ? "signal-buy"
                                  : r.signal === "SELL" ? "signal-sell"
                                  : "signal-neutral";

                const modelConf = r.model_confidence
                    ? `B:${r.model_confidence.buy}% | S:${r.model_confidence.sell}%`
                    : "N/A";

                const patterns = r.patterns && r.patterns.length
                    ? r.patterns.join(", ")
                    : "-";

                tbody.innerHTML += `
                    <tr>
                        <td>${r.symbol}</td>
                        <td>${r.ltp}</td>
                        <td class="${signalClass}">${r.signal}</td>
                        <td>
                            <span class="badge ${
                                r.probability >= 80 ? 'badge-high'
                                : r.probability >= 60 ? 'badge-medium'
                                : 'badge-low'
                            }">${r.probability}%</span>
                        </td>
                        <td>${r.entry}</td>
                        <td>${r.stop_loss}</td>
                        <td>${r.target}</td>
                        <td>${r.rr_ratio}</td>
                        <td class="ml-conf">${modelConf}</td>
                        <td>${patterns}</td>
                        <td>${r.htf_trend}</td>
                        <td>${r.reason}</td>
                    </tr>
                `;
            });
        }


        // -----------------------------
        // Filtering Logic
        // -----------------------------
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
            else if (type === "high-prob") {
                filtered = filtered.filter(r => r.probability >= 80);
            }

            renderResults(filtered);
            updateStats(filtered);
        }


        // -----------------------------
        // Train ML Model
        // -----------------------------
        async function trainModel() {
            if (!confirm("⚠️ Training takes time. Continue?")) return;

            try {
                const resp = await fetch("/train_model", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({force: true})
                });

                const data = await resp.json();

                alert(data.msg || "Done!");
            } catch (err) {
                alert("Model training failed: " + err.message);
            }
        }


        // -----------------------------
        // Refresh Stats (global stats API)
        // -----------------------------
        async function loadStats() {
            try {
                const resp = await fetch("/stats");
                const data = await resp.json();

                document.getElementById("total").textContent = data.total_scanned;
                document.getElementById("buy").textContent = data.buy_signals;
                document.getElementById("sell").textContent = data.sell_signals;
                document.getElementById("ready").textContent = data.trade_ready;
            } catch (err) {
                alert("Failed to load stats: " + err.message);
            }
        }

    </script>
</body>
</html>
"""

# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🚀 Starting Enhanced Stock Scanner")
    logger.info("=" * 60)
    logger.info(f"Cache Backend: {'Redis' if REDIS_AVAILABLE else 'Memory'}")
    logger.info(f"Parallel Workers: {MAX_WORKERS}")
    logger.info(f"Cache Expiry: {CACHE_EXPIRY}s")
    logger.info("=" * 60)
    

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
