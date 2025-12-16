"""
Complete Enhanced Stock Scanner v2.0
=====================================
Production-Grade Rule-Based Trading System

Key Features:
1. Non-repainting indicators (calculated on candle close only)
2. Live LTP-based Entry/SL/Target
3. Breakout Detection (Price, Volume, Pattern)
4. Multi-layer confirmation system
5. ATR-adaptive risk management
6. False signal reduction with strict filters
7. Redis-based persistent caching
8. Parallel processing with ThreadPoolExecutor
9. Rate limiting protection
10. Deterministic and debuggable

Trading Ideology:
- Indicators are calculated ONLY on candle close
- Entry, SL, Target must be based on LIVE LTP
- ATR defines volatility-adaptive risk
- No signal if momentum or volatility is weak
- No conflicting BUY/SELL logic allowed
- System must be conservative, not over-trading
- Every trade must have a logical reason string
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
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

# -------------------------------------------------
# Optional: Redis for persistent caching
# -------------------------------------------------
try:
    import redis
    REDIS_AVAILABLE = True
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()
    logger.info("Redis connected successfully")
except:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available, using in-memory cache only")

# -------------------------------------------------
# Flask setup
# -------------------------------------------------
app = Flask(__name__)
CORS(app)

# -------------------------------------------------
# Enums and Data Classes for Type Safety
# -------------------------------------------------
class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"

class BreakoutType(Enum):
    PRICE_BREAKOUT = "PRICE_BREAKOUT"
    VOLUME_BREAKOUT = "VOLUME_BREAKOUT"
    PATTERN_BREAKOUT = "PATTERN_BREAKOUT"
    CONSOLIDATION_BREAKOUT = "CONSOLIDATION_BREAKOUT"
    RANGE_BREAKOUT = "RANGE_BREAKOUT"
    MOMENTUM_BREAKOUT = "MOMENTUM_BREAKOUT"
    NO_BREAKOUT = "NO_BREAKOUT"

@dataclass
class BreakoutInfo:
    """Container for breakout detection results"""
    detected: bool = False
    breakout_type: BreakoutType = BreakoutType.NO_BREAKOUT
    direction: str = "neutral"  # bullish/bearish/neutral
    strength: float = 0.0  # 0-100
    breakout_level: float = 0.0
    description: str = ""

@dataclass
class PatternInfo:
    """Container for pattern detection results"""
    pattern_name: str = "None"
    detected: bool = False
    direction: str = "neutral"
    reliability: float = 0.0  # 0-100
    description: str = ""

@dataclass
class SignalConfirmation:
    """Container for multi-factor signal confirmation"""
    trend_aligned: bool = False
    volume_confirmed: bool = False
    momentum_confirmed: bool = False
    structure_confirmed: bool = False
    breakout_confirmed: bool = False
    pattern_confirmed: bool = False
    confirmation_count: int = 0
    confirmation_details: List[str] = field(default_factory=list)

@dataclass
class TradeSetup:
    """Container for complete trade setup with LTP-based levels"""
    symbol: str = ""
    signal: SignalType = SignalType.NEUTRAL
    ltp: float = 0.0
    entry: float = 0.0
    stop_loss: float = 0.0
    target1: float = 0.0
    target2: float = 0.0
    target3: float = 0.0
    risk_amount: float = 0.0
    reward_amount: float = 0.0
    rr_ratio: float = 0.0
    position_size_suggested: float = 0.0
    is_achievable: bool = False
    reason: str = ""

# -------------------------------------------------
# Configuration
# -------------------------------------------------
NIFTY_50 = [
    # NIFTY 50
    "RELIANCE","TCS","HDFCBANK","ICICIBANK","INFY",
    "AXISBANK","ITC","LT","SBIN","KOTAKBANK",
    "HINDUNILVR","ASIANPAINT","MARUTI","SUNPHARMA",
    "TITAN","ULTRACEMCO","WIPRO","NTPC","POWERGRID",
    "ONGC","BAJFINANCE","BAJAJFINSV","NESTLEIND",
    "TATASTEEL","JSWSTEEL","HINDALCO","ADANIENT",
    "ADANIGREEN","TECHM","GRASIM","DRREDDY","BRITANNIA",
    "EICHERMOT","HEROMOTOCO","BAJAJ-AUTO","BPCL",
    "COALINDIA","DIVISLAB","CIPLA","APOLLOHOSP",
    "TATACONSUM","HCLTECH","M&M","UPL","SBILIFE",
    "SHRIRAMFIN","INDUSTOWER","HDFCLIFE",

    # NIFTY NEXT 50 / MID LARGE
    "ADANIPORTS","AMBUJACEM","AUROPHARMA","BANDHANBNK",
    "BHEL","BIOCON","BOSCHLTD","CHOLAFIN","COLPAL",
    "CONCOR","DABUR","DLF","GODREJCP","GAIL",
    "HAVELLS","ICICIPRULI","IDFCFIRSTB","INDIGO",
    "IOC","IRCTC","JINDALSTEL","LUPIN","MARICO",
    "MCDOWELL-N","NAUKRI","NMDC","PAGEIND",
    "PETRONET","PIDILITIND","PNB","RECLTD",
    "SAIL","SIEMENS","SRF","TORNTPHARM",
    "TVSMOTOR","UBL","VEDL","ZEEL",

    # MIDCAP EXTENSION (Liquid & Tradable)
    "ASHOKLEY","ASTRAL","ATUL","BALKRISIND","BANKBARODA",
    "BHARATFORG","CANBK","CUMMINSIND","ESCORTS",
    "FEDERALBNK","GLENMARK","HAL","HINDPETRO",
    "IBULHSGFIN","ICICIGI","IGL","INDIANB",
    "IRFC","JSWENERGY","LICHSGFIN","LTIM",
    "MFSL","MPHASIS","MRF","NAM-INDIA",
    "OBEROIRLTY","OFSS","PFC","POLYCAB",
    "SBICARD","SUNTV","TATACHEM","TATAPOWER",
    "TRENT","VOLTAS","YESBANK","ZYDUSLIFE"
]

# -------------------------------------------------
# Trading Configuration - Conservative Settings
# -------------------------------------------------
TRADING_CONFIG = {
    # Minimum Requirements for Signal Generation
    "min_adx": 18,                    # Minimum ADX for trend strength
    "min_atr_percent": 0.004,         # Minimum ATR as % of price (0.4%)
    "min_volume_ratio": 1.2,          # Minimum volume vs 20-day average
    "min_rr_ratio": 1.5,              # Minimum Risk:Reward ratio
    "min_confirmation_count": 3,       # Minimum confirmations required
    "min_probability": 60,            # Minimum probability score
    
    # RSI Zones
    "rsi_buy_min": 45,                # RSI above this for buy
    "rsi_buy_max": 70,                # RSI below this for buy (avoid overbought)
    "rsi_sell_min": 30,               # RSI above this for sell (avoid oversold)
    "rsi_sell_max": 55,               # RSI below this for sell
    
    # Breakout Detection
    "breakout_volume_multiplier": 1.8, # Volume must be 1.8x average
    "consolidation_periods": 10,       # Periods for consolidation detection
    "range_breakout_buffer": 0.002,    # 0.2% buffer for range breakout
    
    # ATR Multipliers for SL/Target
    "atr_sl_multiplier": 1.5,         # SL = Entry ± 1.5 * ATR
    "atr_target1_multiplier": 2.0,    # Target1 = Entry ± 2.0 * ATR
    "atr_target2_multiplier": 3.0,    # Target2 = Entry ± 3.0 * ATR
    "atr_target3_multiplier": 4.5,    # Target3 = Entry ± 4.5 * ATR
    
    # Entry Buffer from LTP
    "entry_buffer_percent": 0.001,    # 0.1% buffer from LTP for entry
    
    # Pattern Detection Settings
    "swing_lookback": 20,             # Periods for swing detection
    "consolidation_threshold": 0.03,  # 3% range for consolidation
}

CACHE_EXPIRY = 300  # 5 minutes cache for stock data
MAX_WORKERS = 10    # Parallel processing threads
RATE_LIMIT_DELAY = 0.05  # Delay between requests to avoid throttling

SCAN_CACHE = {"data": None, "timestamp": None}
CURRENT_METHOD = "ATR"
CURRENT_TIMEFRAME = "60m"

# -------------------------------------------------
# Enhanced Cache System
# -------------------------------------------------
class SmartCache:
    """Hybrid cache using Redis (if available) + in-memory fallback"""
    
    def __init__(self):
        self.memory_cache = {}
        self.use_redis = REDIS_AVAILABLE
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        if self.use_redis:
            try:
                data = redis_client.get(key)
                if data:
                    self.hits += 1
                    return json.loads(data)
                self.misses += 1
                return None
            except Exception as e:
                logger.warning(f"Redis get failed for {key}: {e}")
        
        # Fallback to memory
        cached = self.memory_cache.get(key)
        if cached and cached['expiry'] > time.time():
            self.hits += 1
            return cached['data']
        self.misses += 1
        return None
    
    def set(self, key: str, value: Any, expiry: int = CACHE_EXPIRY) -> None:
        """Set cached value with expiry"""
        if self.use_redis:
            try:
                redis_client.setex(key, expiry, json.dumps(value))
                return
            except Exception as e:
                logger.warning(f"Redis set failed for {key}: {e}")
        
        # Fallback to memory
        self.memory_cache[key] = {
            'data': value,
            'expiry': time.time() + expiry
        }
    
    def clear_expired(self) -> None:
        """Clean up expired memory cache entries"""
        current_time = time.time()
        expired = [k for k, v in self.memory_cache.items() 
                   if v['expiry'] <= current_time]
        for key in expired:
            del self.memory_cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 2),
            "backend": "redis" if self.use_redis else "memory"
        }

cache = SmartCache()

# -------------------------------------------------
# Data Fetching Functions
# -------------------------------------------------
def get_cache_key(symbol: str, interval: str, period: str) -> str:
    """Generate unique cache key"""
    time_bucket = datetime.now().strftime('%Y%m%d%H') + str(datetime.now().minute // 5)
    return f"stock:{symbol}:{interval}:{period}:{time_bucket}"

def fetch_stock_data(symbol: str, interval: str = "60m", period: str = "1mo") -> Optional[pd.DataFrame]:
    """Fetch stock data with caching to reduce API calls"""
    cache_key = get_cache_key(symbol, interval, period)
    
    # Try cache first
    cached_data = cache.get(cache_key)
    if cached_data:
        logger.debug(f"Cache hit for {symbol}")
        return pd.DataFrame(cached_data)
    
    # Fetch from Yahoo
    time.sleep(RATE_LIMIT_DELAY)
    df = fetch_stock_data_original(symbol, interval, period)
    
    if df is not None and not df.empty:
        # Cache for future use
        cache.set(cache_key, df.to_dict('records'), CACHE_EXPIRY)
    
    return df

def fetch_stock_data_original(symbol: str, interval: str = "60m", period: str = "1mo") -> Optional[pd.DataFrame]:
    """Original fetch function - fetches CLOSED candle data only"""
    try:
        yf_symbol = symbol + ".NS"
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            logger.warning(f"[Yahoo] No data for {symbol} ({interval})")
            return None

        df.reset_index(inplace=True)
        date_col = "Datetime" if "Datetime" in df.columns else "Date"
        df = df[[date_col, "Open", "High", "Low", "Close", "Volume"]]
        df.columns = ["date", "open", "high", "low", "close", "volume"]
        
        # IMPORTANT: Only use completed candles for indicator calculation
        # Remove the last candle if market is open (it's still forming)
        if is_market_open():
            if len(df) > 1:
                df = df.iloc[:-1]  # Remove incomplete current candle
        
        return df
    except Exception as e:
        logger.error(f"[Yahoo] Error fetching {symbol}: {e}")
        return None

def is_market_open() -> bool:
    """Check if Indian market is currently open"""
    now = datetime.now()
    # IST timezone consideration (UTC+5:30)
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    # Check if it's a weekday
    if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    
    return market_open <= now <= market_close

def get_live_ltp(symbol: str) -> Optional[float]:
    """
    Get LIVE Last Traded Price for real-time entry/exit calculation.
    This is separate from historical data used for indicators.
    """
    cache_key = f"ltp:{symbol}:{datetime.now().strftime('%Y%m%d%H%M')}"
    cached = cache.get(cache_key)
    if cached:
        return cached
    
    try:
        yf_symbol = symbol + ".NS"
        ticker = yf.Ticker(yf_symbol)
        
        # Get 1-minute data for most recent price
        data = ticker.history(period="1d", interval="1m")
        if not data.empty:
            ltp = float(data["Close"].iloc[-1])
            cache.set(cache_key, ltp, 30)  # Cache for 30 seconds only
            return ltp
        
        # Fallback to daily data
        data = ticker.history(period="1d")
        if not data.empty:
            ltp = float(data["Close"].iloc[-1])
            cache.set(cache_key, ltp, 30)
            return ltp
            
    except Exception as e:
        logger.error(f"[Yahoo LTP] Error fetching {symbol}: {e}")
    return None

def get_intraday_data(symbol: str) -> Optional[pd.DataFrame]:
    """Get intraday 1-minute data for pattern detection"""
    cache_key = f"intraday:{symbol}:{datetime.now().strftime('%Y%m%d%H')}"
    cached = cache.get(cache_key)
    if cached:
        return pd.DataFrame(cached)
    
    try:
        yf_symbol = symbol + ".NS"
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period="1d", interval="1m")
        
        if df.empty:
            return None
            
        df.reset_index(inplace=True)
        date_col = "Datetime" if "Datetime" in df.columns else "Date"
        df = df[[date_col, "Open", "High", "Low", "Close", "Volume"]]
        df.columns = ["date", "open", "high", "low", "close", "volume"]
        
        cache.set(cache_key, df.to_dict('records'), 60)
        return df
        
    except Exception as e:
        logger.error(f"[Yahoo Intraday] Error fetching {symbol}: {e}")
        return None

# -------------------------------------------------
# Technical Indicator Functions (Non-Repainting)
# All calculations use CLOSED candle data only
# -------------------------------------------------

def calculate_rsi(prices: np.ndarray, period: int = 14) -> float:
    """
    Calculate RSI using Wilder's smoothing method.
    Non-repainting: Uses only closed candle data.
    """
    try:
        prices = np.array(prices, dtype=float)
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices)
        seed = deltas[:period]
        up = seed[seed > 0].sum() / period
        down = -seed[seed < 0].sum() / period if np.any(seed < 0) else 0.0001

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
    except Exception as e:
        logger.error(f"RSI error: {e}")
        return 50.0

def calculate_ema(series: pd.Series, length: int) -> float:
    """Calculate EMA - Non-repainting on closed data"""
    if len(series) < length:
        return float(series.iloc[-1])
    return float(series.ewm(span=length, adjust=False).mean().iloc[-1])

def calculate_sma(series: pd.Series, length: int) -> float:
    """Calculate SMA - Non-repainting on closed data"""
    if len(series) < length:
        return float(series.iloc[-1])
    return float(series.rolling(window=length).mean().iloc[-1])

def calculate_ema_series(series: pd.Series, length: int) -> pd.Series:
    """Calculate EMA series for the entire dataframe"""
    return series.ewm(span=length, adjust=False).mean()

def calculate_sma_series(series: pd.Series, length: int) -> pd.Series:
    """Calculate SMA series for the entire dataframe"""
    return series.rolling(window=length).mean()

def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """
    Calculate Average True Range for volatility measurement.
    Non-repainting: Uses only closed candle data.
    """
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
        
        if atr.dropna().empty:
            return 0.0
        return float(atr.iloc[-1])
    except Exception as e:
        logger.error(f"ATR error: {e}")
        return 0.0

def calculate_atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR series for the entire dataframe"""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Tuple[float, str]:
    """
    Calculate SuperTrend indicator.
    Non-repainting: Uses only closed candle data.
    Returns: (supertrend_value, direction)
    """
    try:
        if len(df) < period + 1:
            return 0.0, "neutral"
            
        atr_series = calculate_atr_series(df, period)
        
        high = df["high"]
        low = df["low"]
        close = df["close"]
        hl2 = (high + low) / 2.0

        upper_band = hl2 + multiplier * atr_series
        lower_band = hl2 - multiplier * atr_series
        
        # Initialize SuperTrend
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)
        
        supertrend.iloc[0] = upper_band.iloc[0]
        direction.iloc[0] = -1  # Start bearish
        
        for i in range(1, len(df)):
            if close.iloc[i] > upper_band.iloc[i-1]:
                direction.iloc[i] = 1
            elif close.iloc[i] < lower_band.iloc[i-1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i-1]
                
                if direction.iloc[i] == 1 and lower_band.iloc[i] < lower_band.iloc[i-1]:
                    lower_band.iloc[i] = lower_band.iloc[i-1]
                if direction.iloc[i] == -1 and upper_band.iloc[i] > upper_band.iloc[i-1]:
                    upper_band.iloc[i] = upper_band.iloc[i-1]
            
            supertrend.iloc[i] = lower_band.iloc[i] if direction.iloc[i] == 1 else upper_band.iloc[i]
        
        last_direction = "bullish" if direction.iloc[-1] == 1 else "bearish"
        return float(supertrend.iloc[-1]), last_direction
        
    except Exception as e:
        logger.error(f"SuperTrend error: {e}")
        return 0.0, "neutral"

def calculate_adx(df: pd.DataFrame, period: int = 14) -> Tuple[float, float, float]:
    """
    Calculate ADX with +DI and -DI.
    Non-repainting: Uses only closed candle data.
    Returns: (ADX, +DI, -DI)
    """
    try:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # Calculate directional movement
        high_diff = high.diff()
        low_diff = low.diff().abs() * -1
        
        plus_dm = pd.Series(np.where((high_diff > low_diff.abs()) & (high_diff > 0), high_diff, 0.0), index=df.index)
        minus_dm = pd.Series(np.where((low_diff.abs() > high_diff) & (low_diff.abs() > 0), low_diff.abs(), 0.0), index=df.index)

        # Calculate True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Smooth TR and DMs
        tr_smooth = tr.rolling(window=period).sum()
        plus_dm_smooth = plus_dm.rolling(window=period).sum()
        minus_dm_smooth = minus_dm.rolling(window=period).sum()

        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)

        # Calculate DX and ADX
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 0.0001)) * 100
        adx = dx.rolling(window=period).mean()

        if adx.dropna().empty:
            return 20.0, 0.0, 0.0
            
        return float(adx.iloc[-1]), float(plus_di.iloc[-1]), float(minus_di.iloc[-1])
    except Exception as e:
        logger.error(f"ADX error: {e}")
        return 20.0, 0.0, 0.0

def calculate_vwap(df: pd.DataFrame) -> float:
    """
    Calculate VWAP.
    Non-repainting: Uses only closed candle data.
    """
    try:
        typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
        vwap = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()
        return float(vwap.iloc[-1])
    except Exception as e:
        logger.error(f"VWAP error: {e}")
        return 0.0

def calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
    """
    Calculate MACD, Signal, and Histogram.
    Non-repainting: Uses only closed candle data.
    """
    try:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return float(macd_line.iloc[-1]), float(signal_line.iloc[-1]), float(histogram.iloc[-1])
    except Exception as e:
        logger.error(f"MACD error: {e}")
        return 0.0, 0.0, 0.0

def calculate_bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[float, float, float]:
    """
    Calculate Bollinger Bands.
    Non-repainting: Uses only closed candle data.
    Returns: (upper_band, middle_band, lower_band)
    """
    try:
        middle = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        return float(upper.iloc[-1]), float(middle.iloc[-1]), float(lower.iloc[-1])
    except Exception as e:
        logger.error(f"Bollinger Bands error: {e}")
        return 0.0, 0.0, 0.0

def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[float, float]:
    """
    Calculate Stochastic Oscillator.
    Non-repainting: Uses only closed candle data.
    Returns: (%K, %D)
    """
    try:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, 0.0001)
        d = k.rolling(window=d_period).mean()
        
        return float(k.iloc[-1]), float(d.iloc[-1])
    except Exception as e:
        logger.error(f"Stochastic error: {e}")
        return 50.0, 50.0

def get_daily_pivot(symbol: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Get classic pivot levels based on previous day's OHLC.
    Non-repainting: Uses only completed daily candle.
    Returns: (pivot, r1, s1, r2, s2)
    """
    cache_key = f"pivot:{symbol}:{datetime.now().strftime('%Y%m%d')}"
    cached = cache.get(cache_key)
    if cached:
        return cached['pivot'], cached['r1'], cached['s1'], cached['r2'], cached['s2']
    
    try:
        yf_symbol = symbol + ".NS"
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period="5d", interval="1d")
        if df.empty or len(df) < 2:
            return None, None, None, None, None
            
        df = df.reset_index()
        prev = df.iloc[-2]  # Previous completed day
        
        high = float(prev["High"])
        low = float(prev["Low"])
        close = float(prev["Close"])
        
        pivot = (high + low + close) / 3.0
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        
        result = {'pivot': pivot, 'r1': r1, 's1': s1, 'r2': r2, 's2': s2}
        cache.set(cache_key, result, 86400)  # Cache for 24h
        
        return pivot, r1, s1, r2, s2
    except Exception as e:
        logger.error(f"Daily pivot error for {symbol}: {e}")
        return None, None, None, None, None

# -------------------------------------------------
# Breakout Detection Functions
# -------------------------------------------------

def detect_volume_breakout(df: pd.DataFrame, lookback: int = 20) -> BreakoutInfo:
    """
    Detect volume breakout - unusually high volume indicating institutional interest.
    Non-repainting: Uses only closed candle data.
    """
    try:
        if len(df) < lookback + 1:
            return BreakoutInfo()
        
        volume = df["volume"]
        close = df["close"]
        
        avg_volume = volume.iloc[-lookback-1:-1].mean()
        current_volume = volume.iloc[-1]
        prev_close = close.iloc[-2]
        current_close = close.iloc[-1]
        
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Volume breakout threshold
        threshold = TRADING_CONFIG["breakout_volume_multiplier"]
        
        if volume_ratio >= threshold:
            direction = "bullish" if current_close > prev_close else "bearish"
            strength = min(100, (volume_ratio / threshold) * 50)
            
            return BreakoutInfo(
                detected=True,
                breakout_type=BreakoutType.VOLUME_BREAKOUT,
                direction=direction,
                strength=strength,
                breakout_level=current_close,
                description=f"Volume {volume_ratio:.1f}x average ({direction})"
            )
        
        return BreakoutInfo()
        
    except Exception as e:
        logger.error(f"Volume breakout detection error: {e}")
        return BreakoutInfo()

def detect_price_breakout(df: pd.DataFrame, lookback: int = 20) -> BreakoutInfo:
    """
    Detect price breakout above resistance or below support.
    Non-repainting: Uses only closed candle data.
    """
    try:
        if len(df) < lookback + 1:
            return BreakoutInfo()
        
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        # Calculate resistance and support
        resistance = high.iloc[-lookback-1:-1].max()
        support = low.iloc[-lookback-1:-1].min()
        
        current_close = close.iloc[-1]
        current_high = high.iloc[-1]
        current_low = low.iloc[-1]
        
        buffer = TRADING_CONFIG["range_breakout_buffer"] * current_close
        
        # Bullish breakout above resistance
        if current_close > resistance + buffer:
            strength = min(100, ((current_close - resistance) / resistance) * 1000)
            return BreakoutInfo(
                detected=True,
                breakout_type=BreakoutType.PRICE_BREAKOUT,
                direction="bullish",
                strength=strength,
                breakout_level=resistance,
                description=f"Price broke above resistance {resistance:.2f}"
            )
        
        # Bearish breakout below support
        if current_close < support - buffer:
            strength = min(100, ((support - current_close) / support) * 1000)
            return BreakoutInfo(
                detected=True,
                breakout_type=BreakoutType.PRICE_BREAKOUT,
                direction="bearish",
                strength=strength,
                breakout_level=support,
                description=f"Price broke below support {support:.2f}"
            )
        
        return BreakoutInfo()
        
    except Exception as e:
        logger.error(f"Price breakout detection error: {e}")
        return BreakoutInfo()

def detect_consolidation_breakout(df: pd.DataFrame, lookback: int = 10) -> BreakoutInfo:
    """
    Detect breakout from consolidation/tight range.
    Non-repainting: Uses only closed candle data.
    """
    try:
        if len(df) < lookback + 3:
            return BreakoutInfo()
        
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        # Check if recent candles were in consolidation
        consolidation_high = high.iloc[-lookback-1:-1].max()
        consolidation_low = low.iloc[-lookback-1:-1].min()
        consolidation_range = (consolidation_high - consolidation_low) / consolidation_low
        
        # Must be tight consolidation
        threshold = TRADING_CONFIG["consolidation_threshold"]
        if consolidation_range > threshold:
            return BreakoutInfo()
        
        current_close = close.iloc[-1]
        atr = calculate_atr(df, 14)
        
        # Breakout with momentum
        if current_close > consolidation_high + 0.5 * atr:
            return BreakoutInfo(
                detected=True,
                breakout_type=BreakoutType.CONSOLIDATION_BREAKOUT,
                direction="bullish",
                strength=min(100, 70 + (consolidation_range * 100)),
                breakout_level=consolidation_high,
                description=f"Consolidation breakout above {consolidation_high:.2f}"
            )
        
        if current_close < consolidation_low - 0.5 * atr:
            return BreakoutInfo(
                detected=True,
                breakout_type=BreakoutType.CONSOLIDATION_BREAKOUT,
                direction="bearish",
                strength=min(100, 70 + (consolidation_range * 100)),
                breakout_level=consolidation_low,
                description=f"Consolidation breakdown below {consolidation_low:.2f}"
            )
        
        return BreakoutInfo()
        
    except Exception as e:
        logger.error(f"Consolidation breakout detection error: {e}")
        return BreakoutInfo()

def detect_momentum_breakout(df: pd.DataFrame) -> BreakoutInfo:
    """
    Detect momentum breakout using RSI and MACD.
    Non-repainting: Uses only closed candle data.
    """
    try:
        if len(df) < 30:
            return BreakoutInfo()
        
        close = df["close"]
        
        # Calculate momentum indicators
        rsi = calculate_rsi(close.values, 14)
        macd, signal, histogram = calculate_macd(close)
        prev_histogram = float((close.ewm(span=12, adjust=False).mean() - 
                                close.ewm(span=26, adjust=False).mean()).ewm(span=9, adjust=False).mean().iloc[-2])
        
        # RSI momentum breakout
        rsi_breakout_bull = rsi > 60 and rsi < 80
        rsi_breakout_bear = rsi < 40 and rsi > 20
        
        # MACD momentum (histogram expanding)
        macd_bull = histogram > 0 and histogram > prev_histogram
        macd_bear = histogram < 0 and histogram < prev_histogram
        
        if rsi_breakout_bull and macd_bull:
            strength = min(100, rsi + abs(histogram) * 10)
            return BreakoutInfo(
                detected=True,
                breakout_type=BreakoutType.MOMENTUM_BREAKOUT,
                direction="bullish",
                strength=strength,
                breakout_level=float(close.iloc[-1]),
                description=f"Momentum surge (RSI={rsi:.1f}, MACD expanding)"
            )
        
        if rsi_breakout_bear and macd_bear:
            strength = min(100, (100 - rsi) + abs(histogram) * 10)
            return BreakoutInfo(
                detected=True,
                breakout_type=BreakoutType.MOMENTUM_BREAKOUT,
                direction="bearish",
                strength=strength,
                breakout_level=float(close.iloc[-1]),
                description=f"Momentum drop (RSI={rsi:.1f}, MACD expanding)"
            )
        
        return BreakoutInfo()
        
    except Exception as e:
        logger.error(f"Momentum breakout detection error: {e}")
        return BreakoutInfo()

def detect_range_breakout(df: pd.DataFrame, lookback: int = 20) -> BreakoutInfo:
    """
    Detect breakout from trading range.
    Non-repainting: Uses only closed candle data.
    """
    try:
        if len(df) < lookback + 5:
            return BreakoutInfo()
        
        high = df["high"]
        low = df["low"]
        close = df["close"]
        volume = df["volume"]
        
        # Find the range
        range_high = high.iloc[-lookback-1:-1].max()
        range_low = low.iloc[-lookback-1:-1].min()
        range_mid = (range_high + range_low) / 2
        
        current_close = close.iloc[-1]
        current_volume = volume.iloc[-1]
        avg_volume = volume.iloc[-lookback-1:-1].mean()
        
        # Breakout conditions
        buffer = TRADING_CONFIG["range_breakout_buffer"] * current_close
        volume_confirm = current_volume > avg_volume * 1.3
        
        # Bullish range breakout
        if current_close > range_high + buffer and volume_confirm:
            strength = min(100, 60 + (current_volume / avg_volume) * 10)
            return BreakoutInfo(
                detected=True,
                breakout_type=BreakoutType.RANGE_BREAKOUT,
                direction="bullish",
                strength=strength,
                breakout_level=range_high,
                description=f"Range breakout above {range_high:.2f} with volume"
            )
        
        # Bearish range breakdown
        if current_close < range_low - buffer and volume_confirm:
            strength = min(100, 60 + (current_volume / avg_volume) * 10)
            return BreakoutInfo(
                detected=True,
                breakout_type=BreakoutType.RANGE_BREAKOUT,
                direction="bearish",
                strength=strength,
                breakout_level=range_low,
                description=f"Range breakdown below {range_low:.2f} with volume"
            )
        
        return BreakoutInfo()
        
    except Exception as e:
        logger.error(f"Range breakout detection error: {e}")
        return BreakoutInfo()

def detect_all_breakouts(df: pd.DataFrame) -> List[BreakoutInfo]:
    """Detect all types of breakouts and return list of detected ones."""
    breakouts = []
    
    # Check each type of breakout
    volume_bo = detect_volume_breakout(df)
    if volume_bo.detected:
        breakouts.append(volume_bo)
    
    price_bo = detect_price_breakout(df)
    if price_bo.detected:
        breakouts.append(price_bo)
    
    consolidation_bo = detect_consolidation_breakout(df)
    if consolidation_bo.detected:
        breakouts.append(consolidation_bo)
    
    momentum_bo = detect_momentum_breakout(df)
    if momentum_bo.detected:
        breakouts.append(momentum_bo)
    
    range_bo = detect_range_breakout(df)
    if range_bo.detected:
        breakouts.append(range_bo)
    
    return breakouts

# -------------------------------------------------
# Pattern Detection Functions
# -------------------------------------------------

def detect_higher_high_higher_low(df: pd.DataFrame, lookback: int = 10) -> PatternInfo:
    """
    Detect Higher High and Higher Low pattern (bullish structure).
    Non-repainting: Uses only closed candle data.
    """
    try:
        if len(df) < lookback + 4:
            return PatternInfo()
        
        high = df["high"].iloc[-lookback-1:-1]
        low = df["low"].iloc[-lookback-1:-1]
        
        # Find swing points
        highs = []
        lows = []
        
        for i in range(2, len(high) - 2):
            if high.iloc[i] > high.iloc[i-1] and high.iloc[i] > high.iloc[i-2] and \
               high.iloc[i] > high.iloc[i+1] and high.iloc[i] > high.iloc[i+2]:
                highs.append(high.iloc[i])
            
            if low.iloc[i] < low.iloc[i-1] and low.iloc[i] < low.iloc[i-2] and \
               low.iloc[i] < low.iloc[i+1] and low.iloc[i] < low.iloc[i+2]:
                lows.append(low.iloc[i])
        
        # Check for HH-HL pattern
        if len(highs) >= 2 and len(lows) >= 2:
            if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
                return PatternInfo(
                    pattern_name="Higher High Higher Low",
                    detected=True,
                    direction="bullish",
                    reliability=75,
                    description="Bullish structure: HH-HL confirmed"
                )
        
        return PatternInfo()
        
    except Exception as e:
        logger.error(f"HH-HL pattern detection error: {e}")
        return PatternInfo()

def detect_lower_high_lower_low(df: pd.DataFrame, lookback: int = 10) -> PatternInfo:
    """
    Detect Lower High and Lower Low pattern (bearish structure).
    Non-repainting: Uses only closed candle data.
    """
    try:
        if len(df) < lookback + 4:
            return PatternInfo()
        
        high = df["high"].iloc[-lookback-1:-1]
        low = df["low"].iloc[-lookback-1:-1]
        
        # Find swing points
        highs = []
        lows = []
        
        for i in range(2, len(high) - 2):
            if high.iloc[i] > high.iloc[i-1] and high.iloc[i] > high.iloc[i-2] and \
               high.iloc[i] > high.iloc[i+1] and high.iloc[i] > high.iloc[i+2]:
                highs.append(high.iloc[i])
            
            if low.iloc[i] < low.iloc[i-1] and low.iloc[i] < low.iloc[i-2] and \
               low.iloc[i] < low.iloc[i+1] and low.iloc[i] < low.iloc[i+2]:
                lows.append(low.iloc[i])
        
        # Check for LH-LL pattern
        if len(highs) >= 2 and len(lows) >= 2:
            if highs[-1] < highs[-2] and lows[-1] < lows[-2]:
                return PatternInfo(
                    pattern_name="Lower High Lower Low",
                    detected=True,
                    direction="bearish",
                    reliability=75,
                    description="Bearish structure: LH-LL confirmed"
                )
        
        return PatternInfo()
        
    except Exception as e:
        logger.error(f"LH-LL pattern detection error: {e}")
        return PatternInfo()

def detect_engulfing_pattern(df: pd.DataFrame) -> PatternInfo:
    """
    Detect Bullish or Bearish Engulfing patterns.
    Non-repainting: Uses only closed candle data.
    """
    try:
        if len(df) < 3:
            return PatternInfo()
        
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        current_body = current["close"] - current["open"]
        prev_body = prev["close"] - prev["open"]
        
        # Bullish Engulfing
        if prev_body < 0 and current_body > 0:
            if current["open"] <= prev["close"] and current["close"] >= prev["open"]:
                if abs(current_body) > abs(prev_body) * 1.5:
                    return PatternInfo(
                        pattern_name="Bullish Engulfing",
                        detected=True,
                        direction="bullish",
                        reliability=70,
                        description="Strong bullish reversal pattern"
                    )
        
        # Bearish Engulfing
        if prev_body > 0 and current_body < 0:
            if current["open"] >= prev["close"] and current["close"] <= prev["open"]:
                if abs(current_body) > abs(prev_body) * 1.5:
                    return PatternInfo(
                        pattern_name="Bearish Engulfing",
                        detected=True,
                        direction="bearish",
                        reliability=70,
                        description="Strong bearish reversal pattern"
                    )
        
        return PatternInfo()
        
    except Exception as e:
        logger.error(f"Engulfing pattern detection error: {e}")
        return PatternInfo()

def detect_inside_bar_breakout(df: pd.DataFrame) -> PatternInfo:
    """
    Detect Inside Bar Breakout pattern.
    Non-repainting: Uses only closed candle data.
    """
    try:
        if len(df) < 4:
            return PatternInfo()
        
        # Check if previous candle was inside the one before
        candle_3 = df.iloc[-3]  # Mother bar
        candle_2 = df.iloc[-2]  # Inside bar
        candle_1 = df.iloc[-1]  # Breakout bar
        
        # Inside bar condition
        is_inside = (candle_2["high"] < candle_3["high"] and 
                     candle_2["low"] > candle_3["low"])
        
        if not is_inside:
            return PatternInfo()
        
        # Bullish breakout
        if candle_1["close"] > candle_3["high"]:
            return PatternInfo(
                pattern_name="Inside Bar Bullish Breakout",
                detected=True,
                direction="bullish",
                reliability=65,
                description=f"Breakout above mother bar high {candle_3['high']:.2f}"
            )
        
        # Bearish breakout
        if candle_1["close"] < candle_3["low"]:
            return PatternInfo(
                pattern_name="Inside Bar Bearish Breakout",
                detected=True,
                direction="bearish",
                reliability=65,
                description=f"Breakdown below mother bar low {candle_3['low']:.2f}"
            )
        
        return PatternInfo()
        
    except Exception as e:
        logger.error(f"Inside bar detection error: {e}")
        return PatternInfo()

def detect_all_patterns(df: pd.DataFrame) -> List[PatternInfo]:
    """Detect all patterns and return list of detected ones."""
    patterns = []
    
    hh_hl = detect_higher_high_higher_low(df)
    if hh_hl.detected:
        patterns.append(hh_hl)
    
    lh_ll = detect_lower_high_lower_low(df)
    if lh_ll.detected:
        patterns.append(lh_ll)
    
    engulfing = detect_engulfing_pattern(df)
    if engulfing.detected:
        patterns.append(engulfing)
    
    inside_bar = detect_inside_bar_breakout(df)
    if inside_bar.detected:
        patterns.append(inside_bar)
    
    return patterns

# -------------------------------------------------
# Signal Confirmation System
# -------------------------------------------------

def get_signal_confirmation(
    df: pd.DataFrame,
    ltp: float,
    rsi: float,
    adx: float,
    plus_di: float,
    minus_di: float,
    ema9: float,
    ema21: float,
    ma100: float,
    vwap: float,
    st_direction: str,
    volume_ratio: float,
    breakouts: List[BreakoutInfo],
    patterns: List[PatternInfo],
    signal_direction: str
) -> SignalConfirmation:
    """
    Multi-factor confirmation system for signal validation.
    Reduces false signals by requiring multiple confirmations.
    """
    confirmation = SignalConfirmation()
    details = []
    
    if signal_direction == "bullish":
        # Trend Alignment
        if ema9 > ema21 and ltp > ma100:
            confirmation.trend_aligned = True
            details.append("Trend aligned (EMA9>EMA21, Price>MA100)")
        
        # Volume Confirmation
        if volume_ratio >= TRADING_CONFIG["min_volume_ratio"]:
            confirmation.volume_confirmed = True
            details.append(f"Volume confirmed ({volume_ratio:.1f}x avg)")
        
        # Momentum Confirmation
        if rsi >= TRADING_CONFIG["rsi_buy_min"] and rsi <= TRADING_CONFIG["rsi_buy_max"]:
            if plus_di > minus_di:
                confirmation.momentum_confirmed = True
                details.append(f"Momentum confirmed (RSI={rsi:.1f}, +DI>{-minus_di:.1f})")
        
        # Structure Confirmation
        if ltp > vwap and st_direction == "bullish":
            confirmation.structure_confirmed = True
            details.append("Structure confirmed (Price>VWAP, SuperTrend bullish)")
        
        # Breakout Confirmation
        bullish_breakouts = [b for b in breakouts if b.direction == "bullish"]
        if bullish_breakouts:
            confirmation.breakout_confirmed = True
            breakout_types = [b.breakout_type.value for b in bullish_breakouts]
            details.append(f"Breakout confirmed ({', '.join(breakout_types)})")
        
        # Pattern Confirmation
        bullish_patterns = [p for p in patterns if p.direction == "bullish"]
        if bullish_patterns:
            confirmation.pattern_confirmed = True
            pattern_names = [p.pattern_name for p in bullish_patterns]
            details.append(f"Pattern confirmed ({', '.join(pattern_names)})")
    
    elif signal_direction == "bearish":
        # Trend Alignment
        if ema9 < ema21 and ltp < ma100:
            confirmation.trend_aligned = True
            details.append("Trend aligned (EMA9<EMA21, Price<MA100)")
        
        # Volume Confirmation
        if volume_ratio >= TRADING_CONFIG["min_volume_ratio"]:
            confirmation.volume_confirmed = True
            details.append(f"Volume confirmed ({volume_ratio:.1f}x avg)")
        
        # Momentum Confirmation
        if rsi >= TRADING_CONFIG["rsi_sell_min"] and rsi <= TRADING_CONFIG["rsi_sell_max"]:
            if minus_di > plus_di:
                confirmation.momentum_confirmed = True
                details.append(f"Momentum confirmed (RSI={rsi:.1f}, -DI>{plus_di:.1f})")
        
        # Structure Confirmation
        if ltp < vwap and st_direction == "bearish":
            confirmation.structure_confirmed = True
            details.append("Structure confirmed (Price<VWAP, SuperTrend bearish)")
        
        # Breakout Confirmation
        bearish_breakouts = [b for b in breakouts if b.direction == "bearish"]
        if bearish_breakouts:
            confirmation.breakout_confirmed = True
            breakout_types = [b.breakout_type.value for b in bearish_breakouts]
            details.append(f"Breakout confirmed ({', '.join(breakout_types)})")
        
        # Pattern Confirmation
        bearish_patterns = [p for p in patterns if p.direction == "bearish"]
        if bearish_patterns:
            confirmation.pattern_confirmed = True
            pattern_names = [p.pattern_name for p in bearish_patterns]
            details.append(f"Pattern confirmed ({', '.join(pattern_names)})")
    
    # Count confirmations
    confirmation.confirmation_count = sum([
        confirmation.trend_aligned,
        confirmation.volume_confirmed,
        confirmation.momentum_confirmed,
        confirmation.structure_confirmed,
        confirmation.breakout_confirmed,
        confirmation.pattern_confirmed
    ])
    
    confirmation.confirmation_details = details
    
    return confirmation

# -------------------------------------------------
# Trade Setup Calculator (LTP-Based)
# -------------------------------------------------

def calculate_trade_setup(
    symbol: str,
    signal: SignalType,
    ltp: float,
    atr: float,
    method: str,
    pivot: Optional[float],
    r1: Optional[float],
    s1: Optional[float],
    st_value: float,
    last_high: float,
    last_low: float
) -> TradeSetup:
    """
    Calculate Entry, SL, and Targets based on LIVE LTP.
    All levels are realistic and achievable.
    """
    setup = TradeSetup(symbol=symbol, signal=signal, ltp=ltp)
    
    if signal == SignalType.NEUTRAL:
        setup.reason = "No trade setup - neutral signal"
        return setup
    
    if atr <= 0:
        atr = ltp * 0.01  # Default to 1% if ATR invalid
    
    config = TRADING_CONFIG
    entry_buffer = ltp * config["entry_buffer_percent"]
    
    if signal == SignalType.BUY:
        if method == "ATR":
            setup.entry = round(ltp + entry_buffer, 2)
            setup.stop_loss = round(ltp - config["atr_sl_multiplier"] * atr, 2)
            setup.target1 = round(ltp + config["atr_target1_multiplier"] * atr, 2)
            setup.target2 = round(ltp + config["atr_target2_multiplier"] * atr, 2)
            setup.target3 = round(ltp + config["atr_target3_multiplier"] * atr, 2)
            
        elif method == "SUPERTREND":
            setup.entry = round(ltp + entry_buffer, 2)
            sl_candidate = min(st_value, last_low)
            setup.stop_loss = round(sl_candidate - (atr * 0.3), 2)
            risk = setup.entry - setup.stop_loss
            setup.target1 = round(setup.entry + 2.0 * risk, 2)
            setup.target2 = round(setup.entry + 3.0 * risk, 2)
            setup.target3 = round(setup.entry + 4.0 * risk, 2)
            
        elif method == "PIVOT":
            if r1 and pivot:
                setup.entry = round(max(ltp, pivot) + entry_buffer, 2)
                setup.stop_loss = round(pivot - (atr * 0.5), 2)
                setup.target1 = round(r1, 2)
                setup.target2 = round(r1 + (r1 - pivot), 2)
                setup.target3 = round(r1 + 2 * (r1 - pivot), 2)
            else:
                # Fallback to ATR
                setup.entry = round(ltp + entry_buffer, 2)
                setup.stop_loss = round(ltp - config["atr_sl_multiplier"] * atr, 2)
                setup.target1 = round(ltp + config["atr_target1_multiplier"] * atr, 2)
                setup.target2 = round(ltp + config["atr_target2_multiplier"] * atr, 2)
                setup.target3 = round(ltp + config["atr_target3_multiplier"] * atr, 2)
        else:
            # Default to ATR
            setup.entry = round(ltp + entry_buffer, 2)
            setup.stop_loss = round(ltp - config["atr_sl_multiplier"] * atr, 2)
            setup.target1 = round(ltp + config["atr_target1_multiplier"] * atr, 2)
            setup.target2 = round(ltp + config["atr_target2_multiplier"] * atr, 2)
            setup.target3 = round(ltp + config["atr_target3_multiplier"] * atr, 2)
    
    elif signal == SignalType.SELL:
        if method == "ATR":
            setup.entry = round(ltp - entry_buffer, 2)
            setup.stop_loss = round(ltp + config["atr_sl_multiplier"] * atr, 2)
            setup.target1 = round(ltp - config["atr_target1_multiplier"] * atr, 2)
            setup.target2 = round(ltp - config["atr_target2_multiplier"] * atr, 2)
            setup.target3 = round(ltp - config["atr_target3_multiplier"] * atr, 2)
            
        elif method == "SUPERTREND":
            setup.entry = round(ltp - entry_buffer, 2)
            sl_candidate = max(st_value, last_high)
            setup.stop_loss = round(sl_candidate + (atr * 0.3), 2)
            risk = setup.stop_loss - setup.entry
            setup.target1 = round(setup.entry - 2.0 * risk, 2)
            setup.target2 = round(setup.entry - 3.0 * risk, 2)
            setup.target3 = round(setup.entry - 4.0 * risk, 2)
            
        elif method == "PIVOT":
            if s1 and pivot:
                setup.entry = round(min(ltp, pivot) - entry_buffer, 2)
                setup.stop_loss = round(pivot + (atr * 0.5), 2)
                setup.target1 = round(s1, 2)
                setup.target2 = round(s1 - (pivot - s1), 2)
                setup.target3 = round(s1 - 2 * (pivot - s1), 2)
            else:
                setup.entry = round(ltp - entry_buffer, 2)
                setup.stop_loss = round(ltp + config["atr_sl_multiplier"] * atr, 2)
                setup.target1 = round(ltp - config["atr_target1_multiplier"] * atr, 2)
                setup.target2 = round(ltp - config["atr_target2_multiplier"] * atr, 2)
                setup.target3 = round(ltp - config["atr_target3_multiplier"] * atr, 2)
        else:
            setup.entry = round(ltp - entry_buffer, 2)
            setup.stop_loss = round(ltp + config["atr_sl_multiplier"] * atr, 2)
            setup.target1 = round(ltp - config["atr_target1_multiplier"] * atr, 2)
            setup.target2 = round(ltp - config["atr_target2_multiplier"] * atr, 2)
            setup.target3 = round(ltp - config["atr_target3_multiplier"] * atr, 2)
    
    # Calculate Risk/Reward
    if signal == SignalType.BUY:
        setup.risk_amount = round(setup.entry - setup.stop_loss, 2)
        setup.reward_amount = round(setup.target1 - setup.entry, 2)
    elif signal == SignalType.SELL:
        setup.risk_amount = round(setup.stop_loss - setup.entry, 2)
        setup.reward_amount = round(setup.entry - setup.target1, 2)
    
    if setup.risk_amount > 0:
        setup.rr_ratio = round(setup.reward_amount / setup.risk_amount, 2)
    
    # Check if trade is achievable
    setup.is_achievable = (
        setup.risk_amount > 0 and
        setup.rr_ratio >= config["min_rr_ratio"] and
        abs(setup.entry - ltp) / ltp < 0.005  # Entry within 0.5% of LTP
    )
    
    return setup

# -------------------------------------------------
# Main Scanning Function
# -------------------------------------------------

def scan_symbol(symbol: str, method: str = "ATR", timeframe: str = "60m") -> Optional[Dict[str, Any]]:
    """
    Enhanced scan function with:
    - Non-repainting indicators (closed candle only)
    - Live LTP-based Entry/SL/Target
    - Breakout detection
    - Pattern recognition
    - Multi-factor confirmation
    - Strict filtering for false signal reduction
    """
    try:
        # Fetch historical data (closed candles only)
        df = fetch_stock_data(symbol, interval=timeframe, period="1mo")
        if df is None or df.empty or len(df) < 50:
            return None

        # Get LIVE LTP for trade setup
        ltp = get_live_ltp(symbol)
        if ltp is None:
            ltp = float(df["close"].iloc[-1])

        # Extract series
        close_series = df["close"]
        high_series = df["high"]
        low_series = df["low"]
        vol_series = df["volume"]

        # Last completed candle values
        last_close = float(close_series.iloc[-1])
        prev_close = float(close_series.iloc[-2])
        last_high = float(high_series.iloc[-1])
        last_low = float(low_series.iloc[-1])
        prev_high = float(high_series.iloc[-2])
        prev_low = float(low_series.iloc[-2])
        last_vol = float(vol_series.iloc[-1])
        avg_vol = float(vol_series.rolling(20).mean().iloc[-1])

        # Calculate indicators (non-repainting, closed candle)
        rsi = round(calculate_rsi(close_series.values), 2)
        ema9 = calculate_ema(close_series, 9)
        ema21 = calculate_ema(close_series, 21)
        ma100 = calculate_ema(close_series, 100)
        atr = calculate_atr(df, 14)
        st_value, st_dir = calculate_supertrend(df, 10, 3.0)
        adx, plus_di, minus_di = calculate_adx(df, 14)
        adx = round(adx, 2)
        vwap = calculate_vwap(df)
        macd, macd_signal, macd_hist = calculate_macd(close_series)
        stoch_k, stoch_d = calculate_stochastic(df)
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close_series)

        # Get pivot levels
        pivot, r1, s1, r2, s2 = get_daily_pivot(symbol)
        if pivot is None:
            pivot = (last_high + last_low + last_close) / 3.0
            r1 = 2 * pivot - last_low
            s1 = 2 * pivot - last_high
            r2 = pivot + (last_high - last_low)
            s2 = pivot - (last_high - last_low)

        # Calculate additional metrics
        volume_ratio = last_vol / avg_vol if avg_vol > 0 else 1.0
        volatility_percent = (atr / ltp * 100) if ltp > 0 else 0
        price_change_percent = ((last_close - prev_close) / prev_close * 100) if prev_close > 0 else 0

        # Detect breakouts
        breakouts = detect_all_breakouts(df)
        breakout_count = len(breakouts)
        breakout_descriptions = [b.description for b in breakouts]

        # Detect patterns
        patterns = detect_all_patterns(df)
        pattern_count = len(patterns)
        pattern_descriptions = [p.description for p in patterns]

        # ==========================================
        # SIGNAL GENERATION LOGIC - 8 BATCHES
        # ==========================================
        batches = []
        buy_signals = []
        sell_signals = []

        # Batch 1: Trend BUY/SELL
        trend_buy = (ema9 > ema21) and (ltp > ma100)
        trend_sell = (ema9 < ema21) and (ltp < ma100)
        if trend_buy:
            batches.append("Trend BUY: EMA9>EMA21 & Price>MA100")
            buy_signals.append("trend")
        if trend_sell:
            batches.append("Trend SELL: EMA9<EMA21 & Price<MA100")
            sell_signals.append("trend")

        # Batch 2: Structure BUY/SELL
        structure_buy = (ltp > vwap) and (ltp > pivot)
        structure_sell = (ltp < vwap) and (ltp < pivot)
        if structure_buy:
            batches.append("Structure BUY: Price>VWAP & Price>Pivot")
            buy_signals.append("structure")
        if structure_sell:
            batches.append("Structure SELL: Price<VWAP & Price<Pivot")
            sell_signals.append("structure")

        # Batch 3: Continuation BUY/SELL (SuperTrend + ADX)
        continuation_buy = (st_dir == "bullish") and (adx > 20)
        continuation_sell = (st_dir == "bearish") and (adx > 20)
        if continuation_buy:
            batches.append(f"Continuation BUY: SuperTrend bullish & ADX={adx}>20")
            buy_signals.append("continuation")
        if continuation_sell:
            batches.append(f"Continuation SELL: SuperTrend bearish & ADX={adx}>20")
            sell_signals.append("continuation")

        # Batch 4: Volume/Momentum BUY/SELL
        volume_buy = (volume_ratio > 1.5) and (ltp > prev_high)
        volume_sell = (volume_ratio > 1.5) and (ltp < prev_low)
        if volume_buy:
            batches.append(f"Volume BUY: {volume_ratio:.1f}x volume & Price>PrevHigh")
            buy_signals.append("volume")
        if volume_sell:
            batches.append(f"Volume SELL: {volume_ratio:.1f}x volume & Price<PrevLow")
            sell_signals.append("volume")

        # Batch 5: Breakout BUY/SELL
        bullish_breakouts = [b for b in breakouts if b.direction == "bullish"]
        bearish_breakouts = [b for b in breakouts if b.direction == "bearish"]
        if bullish_breakouts:
            for bo in bullish_breakouts:
                batches.append(f"Breakout BUY: {bo.description}")
                buy_signals.append(f"breakout_{bo.breakout_type.value}")
        if bearish_breakouts:
            for bo in bearish_breakouts:
                batches.append(f"Breakout SELL: {bo.description}")
                sell_signals.append(f"breakout_{bo.breakout_type.value}")

        # Batch 6: Pattern BUY/SELL
        bullish_patterns = [p for p in patterns if p.direction == "bullish"]
        bearish_patterns = [p for p in patterns if p.direction == "bearish"]
        if bullish_patterns:
            for pat in bullish_patterns:
                batches.append(f"Pattern BUY: {pat.pattern_name}")
                buy_signals.append(f"pattern_{pat.pattern_name}")
        if bearish_patterns:
            for pat in bearish_patterns:
                batches.append(f"Pattern SELL: {pat.pattern_name}")
                sell_signals.append(f"pattern_{pat.pattern_name}")

        # Batch 7: RSI Momentum BUY/SELL
        rsi_buy = rsi >= 55 and rsi <= 75 and plus_di > minus_di
        rsi_sell = rsi <= 45 and rsi >= 25 and minus_di > plus_di
        if rsi_buy:
            batches.append(f"RSI Momentum BUY: RSI={rsi}, +DI>-DI")
            buy_signals.append("rsi_momentum")
        if rsi_sell:
            batches.append(f"RSI Momentum SELL: RSI={rsi}, -DI>+DI")
            sell_signals.append("rsi_momentum")

        # Batch 8: MACD Confirmation
        macd_buy = macd > macd_signal and macd_hist > 0
        macd_sell = macd < macd_signal and macd_hist < 0
        if macd_buy:
            batches.append(f"MACD BUY: MACD>Signal, Hist={macd_hist:.2f}")
            buy_signals.append("macd")
        if macd_sell:
            batches.append(f"MACD SELL: MACD<Signal, Hist={macd_hist:.2f}")
            sell_signals.append("macd")

        batch_count = len(batches)
        buy_batch_count = len(buy_signals)
        sell_batch_count = len(sell_signals)

        # ==========================================
        # STRICT FILTERS FOR FALSE SIGNAL REDUCTION
        # ==========================================
        config = TRADING_CONFIG
        
        # Filter 1: Minimum ADX (Trend Strength)
        if adx < config["min_adx"]:
            signal = SignalType.NEUTRAL
            reason = f"Low momentum (ADX={adx} < {config['min_adx']}). No trade."
            return build_neutral_result(symbol, ltp, rsi, adx, atr, vwap, ma100, 
                                       batch_count, method, st_value, reason, breakouts, patterns)

        # Filter 2: Minimum Volatility
        min_atr_percent = config["min_atr_percent"]
        if atr <= 0 or (atr / ltp) < min_atr_percent:
            signal = SignalType.NEUTRAL
            reason = f"Very low volatility (ATR={atr:.2f}, {(atr/ltp*100):.2f}% < {min_atr_percent*100}%). No trade."
            return build_neutral_result(symbol, ltp, rsi, adx, atr, vwap, ma100,
                                       batch_count, method, st_value, reason, breakouts, patterns)

        # Filter 3: No Signals
        if batch_count == 0:
            signal = SignalType.NEUTRAL
            reason = "No batch conditions satisfied."
            return build_neutral_result(symbol, ltp, rsi, adx, atr, vwap, ma100,
                                       batch_count, method, st_value, reason, breakouts, patterns)

        # Filter 4: Conflicting Signals (STRICT - No overlap allowed)
        if buy_signals and sell_signals:
            # Only proceed if one direction is clearly dominant
            if buy_batch_count > sell_batch_count * 2:
                # Clear buy dominance
                sell_signals = []
            elif sell_batch_count > buy_batch_count * 2:
                # Clear sell dominance
                buy_signals = []
            else:
                signal = SignalType.NEUTRAL
                reason = f"Conflicting signals: {buy_batch_count} BUY vs {sell_batch_count} SELL batches."
                return build_neutral_result(symbol, ltp, rsi, adx, atr, vwap, ma100,
                                           batch_count, method, st_value, reason, breakouts, patterns)

        # ==========================================
        # DETERMINE FINAL SIGNAL DIRECTION
        # ==========================================
        signal = SignalType.NEUTRAL
        signal_direction = "neutral"

        # RSI Zone Check
        rsi_buy_ok = config["rsi_buy_min"] <= rsi <= config["rsi_buy_max"]
        rsi_sell_ok = config["rsi_sell_min"] <= rsi <= config["rsi_sell_max"]

        if buy_signals and not sell_signals:
            if rsi_buy_ok:
                signal = SignalType.BUY
                signal_direction = "bullish"
            else:
                signal = SignalType.NEUTRAL
                reason = f"BUY batches present but RSI={rsi} not in zone ({config['rsi_buy_min']}-{config['rsi_buy_max']})"
        elif sell_signals and not buy_signals:
            if rsi_sell_ok:
                signal = SignalType.SELL
                signal_direction = "bearish"
            else:
                signal = SignalType.NEUTRAL
                reason = f"SELL batches present but RSI={rsi} not in zone ({config['rsi_sell_min']}-{config['rsi_sell_max']})"

        if signal == SignalType.NEUTRAL and not (buy_signals or sell_signals):
            reason = "Batches present but no clear direction."
            return build_neutral_result(symbol, ltp, rsi, adx, atr, vwap, ma100,
                                       batch_count, method, st_value, reason, breakouts, patterns)

        # ==========================================
        # SIGNAL CONFIRMATION
        # ==========================================
        confirmation = get_signal_confirmation(
            df=df,
            ltp=ltp,
            rsi=rsi,
            adx=adx,
            plus_di=plus_di,
            minus_di=minus_di,
            ema9=ema9,
            ema21=ema21,
            ma100=ma100,
            vwap=vwap,
            st_direction=st_dir,
            volume_ratio=volume_ratio,
            breakouts=breakouts,
            patterns=patterns,
            signal_direction=signal_direction
        )

        # Filter 5: Minimum Confirmations
        min_confirmations = config["min_confirmation_count"]
        if signal != SignalType.NEUTRAL and confirmation.confirmation_count < min_confirmations:
            reason = f"Insufficient confirmations ({confirmation.confirmation_count}/{min_confirmations}): {', '.join(confirmation.confirmation_details[:3])}"
            return build_neutral_result(symbol, ltp, rsi, adx, atr, vwap, ma100,
                                       batch_count, method, st_value, reason, breakouts, patterns)

        # ==========================================
        # CALCULATE TRADE SETUP (LTP-BASED)
        # ==========================================
        trade_setup = calculate_trade_setup(
            symbol=symbol,
            signal=signal,
            ltp=ltp,
            atr=atr,
            method=method.upper(),
            pivot=pivot,
            r1=r1,
            s1=s1,
            st_value=st_value,
            last_high=last_high,
            last_low=last_low
        )

        # Filter 6: Minimum R:R Ratio
        min_rr = config["min_rr_ratio"]
        if signal != SignalType.NEUTRAL and trade_setup.rr_ratio < min_rr:
            signal = SignalType.NEUTRAL
            reason = f"Poor Risk:Reward ({trade_setup.rr_ratio:.2f} < {min_rr})"
            return build_neutral_result(symbol, ltp, rsi, adx, atr, vwap, ma100,
                                       batch_count, method, st_value, reason, breakouts, patterns)

        # ==========================================
        # CALCULATE PROBABILITY SCORE
        # ==========================================
        probability = 50  # Base probability

        # Add for confirmations
        probability += confirmation.confirmation_count * 7

        # Add for strong breakouts
        for bo in breakouts:
            if bo.strength > 70:
                probability += 5
            elif bo.strength > 50:
                probability += 3

        # Add for patterns
        for pat in patterns:
            if pat.reliability > 70:
                probability += 5
            elif pat.reliability > 60:
                probability += 3

        # Add for trend strength
        if adx >= 30:
            probability += 5
        elif adx >= 25:
            probability += 3

        # Add for volume
        if volume_ratio >= 2.0:
            probability += 5
        elif volume_ratio >= 1.5:
            probability += 3

        # Add for R:R
        if trade_setup.rr_ratio >= 2.5:
            probability += 5
        elif trade_setup.rr_ratio >= 2.0:
            probability += 3

        probability = max(0, min(95, probability))

        # ==========================================
        # TRADE READINESS CHECK
        # ==========================================
        trade_ready = (
            signal in [SignalType.BUY, SignalType.SELL] and
            confirmation.confirmation_count >= config["min_confirmation_count"] and
            probability >= config["min_probability"] and
            trade_setup.rr_ratio >= config["min_rr_ratio"] and
            trade_setup.is_achievable and
            adx >= config["min_adx"] and
            volume_ratio >= config["min_volume_ratio"]
        )

        # ==========================================
        # BUILD REASON STRING
        # ==========================================
        batch_text = " | ".join(batches[:4]) if batches else "No batches"
        confirmation_text = ", ".join(confirmation.confirmation_details[:3])
        breakout_text = ", ".join(breakout_descriptions[:2]) if breakout_descriptions else "None"
        pattern_text = ", ".join(pattern_descriptions[:2]) if pattern_descriptions else "None"

        if signal == SignalType.BUY:
            reason_prefix = "BUY: "
        elif signal == SignalType.SELL:
            reason_prefix = "SELL: "
        else:
            reason_prefix = ""

        reason = f"{reason_prefix}{batch_text} | Confirmations: {confirmation_text}"
        if breakout_descriptions:
            reason += f" | Breakouts: {breakout_text}"
        if pattern_descriptions:
            reason += f" | Patterns: {pattern_text}"

        # ==========================================
        # BUILD FINAL RESULT
        # ==========================================
        return {
            "symbol": symbol,
            "ltp": round(float(ltp), 2),
            "signal": signal.value,
            "rsi": rsi,
            "entry": trade_setup.entry,
            "stop_loss": trade_setup.stop_loss,
            "target": trade_setup.target1,  # Primary target
            "target2": trade_setup.target2,
            "target3": trade_setup.target3,
            "rr_ratio": trade_setup.rr_ratio,
            "risk_amount": trade_setup.risk_amount,
            "reward_amount": trade_setup.reward_amount,
            "probability": int(probability),
            "reason": reason,
            "trade_ready": bool(trade_ready),
            "atr": round(float(atr), 2),
            "atr_percent": round(volatility_percent, 2),
            "ma100": round(float(ma100), 2),
            "adx": adx,
            "plus_di": round(plus_di, 2),
            "minus_di": round(minus_di, 2),
            "vwap": round(float(vwap), 2),
            "ema9": round(ema9, 2),
            "ema21": round(ema21, 2),
            "macd": round(macd, 2),
            "macd_signal": round(macd_signal, 2),
            "macd_hist": round(macd_hist, 2),
            "stoch_k": round(stoch_k, 2),
            "stoch_d": round(stoch_d, 2),
            "volume_ratio": round(volume_ratio, 2),
            "batches_count": int(batch_count),
            "buy_batch_count": buy_batch_count,
            "sell_batch_count": sell_batch_count,
            "confirmation_count": confirmation.confirmation_count,
            "confirmations": confirmation.confirmation_details,
            "breakout_count": breakout_count,
            "breakouts": breakout_descriptions,
            "pattern_count": pattern_count,
            "patterns": pattern_descriptions,
            "method": method.upper(),
            "supertrend": round(float(st_value), 2),
            "supertrend_direction": st_dir,
            "pivot": round(pivot, 2) if pivot else 0.0,
            "r1": round(r1, 2) if r1 else 0.0,
            "s1": round(s1, 2) if s1 else 0.0,
            "r2": round(r2, 2) if r2 else 0.0,
            "s2": round(s2, 2) if s2 else 0.0,
            "price_change_percent": round(price_change_percent, 2),
            "is_achievable": trade_setup.is_achievable,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"scan_symbol error for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None


def build_neutral_result(
    symbol: str,
    ltp: float,
    rsi: float,
    adx: float,
    atr: float,
    vwap: float,
    ma100: float,
    batch_count: int,
    method: str,
    st_value: float,
    reason: str,
    breakouts: List[BreakoutInfo],
    patterns: List[PatternInfo]
) -> Dict[str, Any]:
    """Build result object for neutral signals."""
    breakout_descriptions = [b.description for b in breakouts]
    pattern_descriptions = [p.description for p in patterns]
    
    return {
        "symbol": symbol,
        "ltp": round(float(ltp), 2) if ltp else 0.0,
        "signal": "NEUTRAL",
        "rsi": round(float(rsi), 2),
        "entry": 0.0,
        "stop_loss": 0.0,
        "target": 0.0,
        "target2": 0.0,
        "target3": 0.0,
        "rr_ratio": 0.0,
        "risk_amount": 0.0,
        "reward_amount": 0.0,
        "probability": 50,
        "reason": reason,
        "trade_ready": False,
        "atr": round(float(atr), 2) if atr else 0.0,
        "atr_percent": round((atr / ltp * 100), 2) if ltp > 0 else 0.0,
        "ma100": round(float(ma100), 2) if ma100 else 0.0,
        "adx": float(adx),
        "plus_di": 0.0,
        "minus_di": 0.0,
        "vwap": round(float(vwap), 2) if vwap else 0.0,
        "ema9": 0.0,
        "ema21": 0.0,
        "macd": 0.0,
        "macd_signal": 0.0,
        "macd_hist": 0.0,
        "stoch_k": 50.0,
        "stoch_d": 50.0,
        "volume_ratio": 1.0,
        "batches_count": int(batch_count),
        "buy_batch_count": 0,
        "sell_batch_count": 0,
        "confirmation_count": 0,
        "confirmations": [],
        "breakout_count": len(breakouts),
        "breakouts": breakout_descriptions,
        "pattern_count": len(patterns),
        "patterns": pattern_descriptions,
        "method": method,
        "supertrend": round(float(st_value), 2) if st_value else 0.0,
        "supertrend_direction": "neutral",
        "pivot": 0.0,
        "r1": 0.0,
        "s1": 0.0,
        "r2": 0.0,
        "s2": 0.0,
        "price_change_percent": 0.0,
        "is_achievable": False,
        "timestamp": datetime.now().isoformat(),
    }


# -------------------------------------------------
# Parallel Scanning
# -------------------------------------------------
def scan_symbol_wrapper(args: Tuple[str, str, str]) -> Optional[Dict[str, Any]]:
    """Wrapper for parallel execution"""
    symbol, method, timeframe = args
    return scan_symbol(symbol, method, timeframe)


def scan_multiple_symbols(symbols: List[str], method: str = "ATR", timeframe: str = "60m") -> List[Dict[str, Any]]:
    """Scan multiple symbols in parallel"""
    results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_symbol = {
            executor.submit(scan_symbol_wrapper, (sym, method, timeframe)): sym 
            for sym in symbols
        }
        
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
    
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
        if isinstance(raw_symbols, list) and len(raw_symbols) > 0:
            symbols_to_scan = [s.strip().upper() for s in raw_symbols if len(s.strip()) > 0]
        else:
            symbols_to_scan = NIFTY_50

        if method not in ["ATR", "SUPERTREND", "PIVOT"]:
            method = "ATR"
        if timeframe not in ["60m", "15m", "5m"]:
            timeframe = "60m"

        CURRENT_METHOD = method
        CURRENT_TIMEFRAME = timeframe

        logger.info(f"Scanning {len(symbols_to_scan)} symbols in parallel...")
        start_time = time.time()
        
        results = scan_multiple_symbols(symbols_to_scan, method, timeframe)
        
        scan_time = round(time.time() - start_time, 2)
        logger.info(f"Scan completed in {scan_time}s")

        # Sort: trade_ready first, then by probability
        results.sort(key=lambda x: (
            not x.get("trade_ready", False),
            -x.get("probability", 0),
            -x.get("confirmation_count", 0)
        ))

        SCAN_CACHE["data"] = results
        SCAN_CACHE["timestamp"] = datetime.now()

        # Statistics
        buy_signals = len([s for s in results if s.get("signal") == "BUY"])
        sell_signals = len([s for s in results if s.get("signal") == "SELL"])
        trade_ready = len([s for s in results if s.get("trade_ready")])
        breakouts_detected = len([s for s in results if s.get("breakout_count", 0) > 0])
        patterns_detected = len([s for s in results if s.get("pattern_count", 0) > 0])

        return jsonify({
            "results": results,
            "scan_time": scan_time,
            "cache_stats": cache.get_stats(),
            "summary": {
                "total_scanned": len(results),
                "buy_signals": buy_signals,
                "sell_signals": sell_signals,
                "trade_ready": trade_ready,
                "breakouts_detected": breakouts_detected,
                "patterns_detected": patterns_detected,
                "method": method,
                "timeframe": timeframe
            }
        })
    except Exception as e:
        logger.error(f"/scan error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/stats", methods=["GET"])
def stats():
    try:
        data = SCAN_CACHE.get("data") or []
        ts = SCAN_CACHE.get("timestamp")
        cache_stats = cache.get_stats()
        
        return jsonify({
            "total_scanned": len(data),
            "buy_signals": len([s for s in data if s.get("signal") == "BUY"]),
            "sell_signals": len([s for s in data if s.get("signal") == "SELL"]),
            "trade_ready": len([s for s in data if s.get("trade_ready")]),
            "breakouts_detected": len([s for s in data if s.get("breakout_count", 0) > 0]),
            "patterns_detected": len([s for s in data if s.get("pattern_count", 0) > 0]),
            "last_scan": ts.isoformat() if ts else None,
            "method": CURRENT_METHOD,
            "timeframe": CURRENT_TIMEFRAME,
            "cache_backend": "redis" if REDIS_AVAILABLE else "memory",
            "cache_hit_rate": cache_stats.get("hit_rate", 0),
            "parallel_workers": MAX_WORKERS,
            "trading_config": TRADING_CONFIG
        })
    except Exception as e:
        logger.error(f"/stats error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/config", methods=["GET", "POST"])
def config_endpoint():
    """Get or update trading configuration"""
    global TRADING_CONFIG
    
    try:
        if request.method == "GET":
            return jsonify(TRADING_CONFIG)
        
        elif request.method == "POST":
            data = request.get_json() or {}
            
            # Only update allowed parameters
            allowed_params = {
                "min_adx": (int, 10, 40),
                "min_atr_percent": (float, 0.001, 0.01),
                "min_volume_ratio": (float, 1.0, 3.0),
                "min_rr_ratio": (float, 1.2, 3.0),
                "min_confirmation_count": (int, 1, 6),
                "min_probability": (int, 40, 80),
                "rsi_buy_min": (int, 40, 60),
                "rsi_buy_max": (int, 60, 80),
                "rsi_sell_min": (int, 20, 50),
                "rsi_sell_max": (int, 40, 60),
                "breakout_volume_multiplier": (float, 1.2, 3.0),
                "consolidation_periods": (int, 5, 30),
                "range_breakout_buffer": (float, 0.001, 0.01),
                "atr_sl_multiplier": (float, 1.0, 3.0),
                "atr_target1_multiplier": (float, 1.5, 4.0),
                "atr_target2_multiplier": (float, 2.5, 6.0),
                "atr_target3_multiplier": (float, 4.0, 8.0),
                "entry_buffer_percent": (float, 0.0005, 0.005),
                "swing_lookback": (int, 10, 30),
                "consolidation_threshold": (float, 0.02, 0.1),
            }
            
            updates = {}
            for key, (type_func, min_val, max_val) in allowed_params.items():
                if key in data:
                    try:
                        value = type_func(data[key])
                        if min_val <= value <= max_val:
                            updates[key] = value
                        else:
                            return jsonify({
                                "error": f"{key} must be between {min_val} and {max_val}"
                            }), 400
                    except (ValueError, TypeError):
                        return jsonify({
                            "error": f"Invalid value for {key}"
                        }), 400
            
            # Update configuration
            TRADING_CONFIG.update(updates)
            
            # Log the update
            if updates:
                logger.info(f"Updated trading config: {list(updates.keys())}")
            
            return jsonify({
                "message": "Configuration updated successfully",
                "updates": updates,
                "current_config": TRADING_CONFIG
            })
    
    except Exception as e:
        logger.error(f"/config error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/analyze/<symbol>", methods=["GET"])
def analyze_symbol(symbol):
    """Detailed analysis of a single symbol"""
    try:
        method = request.args.get("method", "ATR").upper()
        timeframe = request.args.get("timeframe", "60m")
        
        result = scan_symbol(symbol, method, timeframe)
        
        if not result:
            return jsonify({
                "symbol": symbol,
                "error": "Unable to analyze symbol"
            }), 404
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"/analyze/{symbol} error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/clear_cache", methods=["POST"])
def clear_cache():
    """Clear all caches"""
    try:
        if REDIS_AVAILABLE:
            redis_client.flushdb()
        
        cache.memory_cache.clear()
        cache.hits = 0
        cache.misses = 0
        
        SCAN_CACHE["data"] = None
        SCAN_CACHE["timestamp"] = None
        
        logger.info("All caches cleared")
        
        return jsonify({
            "message": "Cache cleared successfully",
            "cache_stats": cache.get_stats()
        })
    except Exception as e:
        logger.error(f"/clear_cache error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    try:
        # Test data fetching
        test_symbol = "RELIANCE"
        df = fetch_stock_data(test_symbol, "60m", "1d")
        data_ok = df is not None and not df.empty
        
        # Test LTP
        ltp = get_live_ltp(test_symbol)
        ltp_ok = ltp is not None
        
        # Check Redis if available
        redis_ok = not REDIS_AVAILABLE or redis_client.ping()
        
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "data_fetching": data_ok,
                "live_ltp": ltp_ok,
                "redis": redis_ok,
                "cache": cache.get_stats()
            },
            "config": {
                "method": CURRENT_METHOD,
                "timeframe": CURRENT_TIMEFRAME,
                "parallel_workers": MAX_WORKERS,
                "rate_limit_delay": RATE_LIMIT_DELAY
            }
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500


@app.route("/symbols", methods=["GET"])
def get_symbols():
    """Get available symbols"""
    try:
        category = request.args.get("category", "all")
        
        if category == "nifty50":
            symbols = NIFTY_50[:50]
        elif category == "nifty100":
            symbols = NIFTY_50[:100]
        elif category == "midcap":
            symbols = [s for s in NIFTY_50 if s not in NIFTY_50[:100]][:50]
        else:
            symbols = NIFTY_50
        
        return jsonify({
            "symbols": symbols,
            "count": len(symbols),
            "category": category
        })
    except Exception as e:
        logger.error(f"/symbols error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    """Serve the web interface"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Enhanced Stock Scanner v2.0</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
            body { background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%); color: #fff; min-height: 100vh; }
            .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
            header { text-align: center; margin-bottom: 30px; padding-bottom: 20px; border-bottom: 2px solid #00d4ff; }
            h1 { font-size: 2.5rem; background: linear-gradient(90deg, #00d4ff, #00ff88); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 10px; }
            .subtitle { color: #a0d2ff; font-size: 1.1rem; }
            .dashboard { display: grid; grid-template-columns: 300px 1fr; gap: 20px; margin-bottom: 30px; }
            .control-panel { background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); border-radius: 15px; padding: 25px; border: 1px solid rgba(0, 212, 255, 0.2); }
            .results-panel { background: rgba(255, 255, 255, 0.03); border-radius: 15px; padding: 20px; }
            .card { background: rgba(255, 255, 255, 0.05); border-radius: 10px; padding: 20px; margin-bottom: 20px; border: 1px solid rgba(255, 255, 255, 0.1); }
            .card-title { font-size: 1.2rem; color: #00d4ff; margin-bottom: 15px; display: flex; align-items: center; gap: 10px; }
            .card-title i { font-size: 1.4rem; }
            .form-group { margin-bottom: 20px; }
            label { display: block; margin-bottom: 8px; color: #a0d2ff; font-weight: 500; }
            select, input, button { width: 100%; padding: 12px 15px; border-radius: 8px; border: 1px solid #2c5364; background: rgba(255, 255, 255, 0.08); color: white; font-size: 1rem; }
            select:focus, input:focus { outline: none; border-color: #00d4ff; box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.3); }
            button { background: linear-gradient(90deg, #00d4ff, #00a8ff); border: none; cursor: pointer; font-weight: bold; transition: all 0.3s; }
            button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0, 212, 255, 0.4); }
            button:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
            .btn-secondary { background: linear-gradient(90deg, #ff6b6b, #ff8e53); }
            .btn-group { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
            .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 25px; }
            .stat-card { background: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 10px; text-align: center; border: 1px solid rgba(0, 212, 255, 0.2); }
            .stat-value { font-size: 2rem; font-weight: bold; color: #00ff88; margin: 5px 0; }
            .stat-label { color: #a0d2ff; font-size: 0.9rem; }
            .badge { display: inline-block; padding: 4px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: bold; margin-right: 5px; }
            .badge-buy { background: rgba(0, 255, 136, 0.2); color: #00ff88; border: 1px solid #00ff88; }
            .badge-sell { background: rgba(255, 107, 107, 0.2); color: #ff6b6b; border: 1px solid #ff6b6b; }
            .badge-neutral { background: rgba(160, 210, 255, 0.2); color: #a0d2ff; border: 1px solid #a0d2ff; }
            .badge-ready { background: rgba(0, 212, 255, 0.2); color: #00d4ff; border: 1px solid #00d4ff; }
            .badge-breakout { background: rgba(255, 193, 7, 0.2); color: #ffc107; border: 1px solid #ffc107; }
            .table-container { overflow-x: auto; }
            table { width: 100%; border-collapse: collapse; }
            th { background: rgba(0, 212, 255, 0.1); color: #00d4ff; padding: 15px; text-align: left; font-weight: 600; }
            td { padding: 15px; border-bottom: 1px solid rgba(255, 255, 255, 0.05); }
            tr:hover { background: rgba(0, 212, 255, 0.05); }
            .signal-buy { color: #00ff88; font-weight: bold; }
            .signal-sell { color: #ff6b6b; font-weight: bold; }
            .signal-neutral { color: #a0d2ff; }
            .progress-bar { height: 10px; background: rgba(255, 255, 255, 0.1); border-radius: 5px; overflow: hidden; margin-top: 5px; }
            .progress-fill { height: 100%; border-radius: 5px; }
            .progress-high { background: linear-gradient(90deg, #00ff88, #00d4ff); }
            .progress-medium { background: linear-gradient(90deg, #ffc107, #ff8e53); }
            .progress-low { background: linear-gradient(90deg, #ff6b6b, #ff4757); }
            .chip { display: inline-block; background: rgba(0, 212, 255, 0.1); padding: 4px 10px; border-radius: 15px; font-size: 0.8rem; margin: 2px; }
            .loading { display: none; text-align: center; padding: 40px; }
            .spinner { border: 4px solid rgba(255, 255, 255, 0.1); border-radius: 50%; border-top: 4px solid #00d4ff; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto 15px; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
            .message { padding: 15px; border-radius: 8px; margin-bottom: 20px; display: none; }
            .message.success { background: rgba(0, 255, 136, 0.1); border: 1px solid #00ff88; color: #00ff88; }
            .message.error { background: rgba(255, 107, 107, 0.1); border: 1px solid #ff6b6b; color: #ff6b6b; }
            .footer { text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid rgba(255, 255, 255, 0.1); color: #a0d2ff; font-size: 0.9rem; }
            .tab-buttons { display: flex; gap: 10px; margin-bottom: 20px; }
            .tab-button { padding: 10px 20px; background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px; cursor: pointer; transition: all 0.3s; }
            .tab-button.active { background: rgba(0, 212, 255, 0.2); border-color: #00d4ff; color: #00d4ff; }
            .tab-content { display: none; }
            .tab-content.active { display: block; }
            @media (max-width: 1024px) { .dashboard { grid-template-columns: 1fr; } }
            @media (max-width: 768px) { .stats-grid { grid-template-columns: 1fr; } }
        </style>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    </head>
    <body>
        <div class="container">
            <header>
                <h1><i class="fas fa-chart-line"></i> Enhanced Stock Scanner v2.0</h1>
                <p class="subtitle">Production-Grade Rule-Based Trading System | Non-Repainting | Live LTP-Based | Multi-Confirmation</p>
            </header>

            <div class="dashboard">
                <!-- Control Panel -->
                <div class="control-panel">
                    <div class="card">
                        <div class="card-title"><i class="fas fa-cogs"></i> Scanner Controls</div>
                        <div class="form-group">
                            <label><i class="fas fa-list"></i> Symbols Category</label>
                            <select id="symbolCategory">
                                <option value="all">All Symbols (150+)</option>
                                <option value="nifty50">NIFTY 50</option>
                                <option value="nifty100">NIFTY 100</option>
                                <option value="midcap">Midcap (50)</option>
                                <option value="custom">Custom Symbols</option>
                            </select>
                        </div>
                        <div class="form-group" id="customSymbolsGroup" style="display: none;">
                            <label><i class="fas fa-pen"></i> Custom Symbols (comma separated)</label>
                            <input type="text" id="customSymbols" placeholder="RELIANCE, TCS, HDFCBANK">
                        </div>
                        <div class="form-group">
                            <label><i class="fas fa-magic"></i> Trading Method</label>
                            <select id="method">
                                <option value="ATR">ATR Adaptive (Recommended)</option>
                                <option value="SUPERTREND">SuperTrend + EMA</option>
                                <option value="PIVOT">Pivot + VWAP</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label><i class="fas fa-clock"></i> Timeframe</label>
                            <select id="timeframe">
                                <option value="60m">60 Minutes (Swing)</option>
                                <option value="15m">15 Minutes (Intraday)</option>
                                <option value="5m">5 Minutes (Scalping)</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <button id="scanBtn" onclick="startScan()">
                                <i class="fas fa-search"></i> START SCAN
                            </button>
                        </div>
                        <div class="form-group btn-group">
                            <button class="btn-secondary" onclick="clearCache()">
                                <i class="fas fa-broom"></i> Clear Cache
                            </button>
                            <button class="btn-secondary" onclick="checkHealth()">
                                <i class="fas fa-heartbeat"></i> Health Check
                            </button>
                        </div>
                    </div>

                    <div class="card">
                        <div class="card-title"><i class="fas fa-chart-bar"></i> Live Stats</div>
                        <div id="liveStats">
                            <p style="color: #a0d2ff; text-align: center;">Scan to see statistics...</p>
                        </div>
                    </div>

                    <div class="card">
                        <div class="card-title"><i class="fas fa-info-circle"></i> System Info</div>
                        <div style="font-size: 0.9rem; color: #a0d2ff;">
                            <p><i class="fas fa-bolt"></i> <strong>Non-Repainting</strong>: Indicators use closed candles only</p>
                            <p><i class="fas fa-sync"></i> <strong>Live LTP</strong>: Entry/SL/Target based on real-time price</p>
                            <p><i class="fas fa-filter"></i> <strong>8-Batch Logic</strong>: Multi-factor signal generation</p>
                            <p><i class="fas fa-check-double"></i> <strong>6 Confirmations</strong>: Strict validation system</p>
                            <p><i class="fas fa-database"></i> <strong>Smart Cache</strong>: Redis + Memory hybrid</p>
                            <p><i class="fas fa-microchip"></i> <strong>Parallel Processing</strong>: 10 threads for speed</p>
                        </div>
                    </div>
                </div>

                <!-- Results Panel -->
                <div class="results-panel">
                    <div class="tab-buttons">
                        <div class="tab-button active" onclick="showTab('results')">Scan Results</div>
                        <div class="tab-button" onclick="showTab('analysis')">Detailed Analysis</div>
                        <div class="tab-button" onclick="showTab('config')">Configuration</div>
                    </div>

                    <div id="message" class="message"></div>

                    <div class="loading" id="loading">
                        <div class="spinner"></div>
                        <p>Scanning symbols in parallel...</p>
                        <p id="scanProgress">Initializing...</p>
                    </div>

                    <!-- Results Tab -->
                    <div class="tab-content active" id="resultsTab">
                        <div id="statsGrid" class="stats-grid"></div>
                        
                        <div class="card">
                            <div class="card-title"><i class="fas fa-table"></i> Scan Results</div>
                            <div class="table-container">
                                <table id="resultsTable">
                                    <thead>
                                        <tr>
                                            <th>Symbol</th>
                                            <th>Signal</th>
                                            <th>LTP</th>
                                            <th>Entry/SL/Target</th>
                                            <th>Prob.</th>
                                            <th>RR Ratio</th>
                                            <th>ATR%</th>
                                            <th>Details</th>
                                        </tr>
                                    </thead>
                                    <tbody id="resultsBody">
                                        <!-- Results will be inserted here -->
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>

                    <!-- Analysis Tab -->
                    <div class="tab-content" id="analysisTab">
                        <div class="card">
                            <div class="card-title"><i class="fas fa-chart-area"></i> Symbol Analysis</div>
                            <div class="form-group">
                                <input type="text" id="analyzeSymbol" placeholder="Enter symbol (e.g., RELIANCE)" style="margin-bottom: 15px;">
                                <button onclick="analyzeSymbol()"><i class="fas fa-search"></i> Analyze Symbol</button>
                            </div>
                            <div id="analysisResult"></div>
                        </div>
                    </div>

                    <!-- Configuration Tab -->
                    <div class="tab-content" id="configTab">
                        <div class="card">
                            <div class="card-title"><i class="fas fa-sliders-h"></i> Trading Configuration</div>
                            <div id="configForm">
                                <!-- Config will be loaded here -->
                            </div>
                            <button onclick="saveConfig()"><i class="fas fa-save"></i> Save Configuration</button>
                        </div>
                    </div>
                </div>
            </div>

            <div class="footer">
                <p><i class="fas fa-code"></i> Enhanced Stock Scanner v2.0 | Production-Grade Trading System</p>
                <p>All indicators are non-repainting (calculated on candle close only). Entry/SL/Target based on Live LTP.</p>
                <p>Risk Management: ATR-adaptive with minimum 1.5:1 Risk:Reward ratio required.</p>
            </div>
        </div>

        <script>
            let currentTab = 'results';
            
            function showTab(tabName) {
                document.querySelectorAll('.tab-button').forEach(btn => {
                    btn.classList.remove('active');
                });
                document.querySelectorAll('.tab-content').forEach(content => {
                    content.classList.remove('active');
                });
                
                event.target.classList.add('active');
                document.getElementById(tabName + 'Tab').classList.add('active');
                currentTab = tabName;
            }
            
            document.getElementById('symbolCategory').addEventListener('change', function() {
                document.getElementById('customSymbolsGroup').style.display = 
                    this.value === 'custom' ? 'block' : 'none';
            });
            
            function showMessage(text, type) {
                const msg = document.getElementById('message');
                msg.textContent = text;
                msg.className = 'message ' + type;
                msg.style.display = 'block';
                setTimeout(() => msg.style.display = 'none', 5000);
            }
            
            async function startScan() {
                const btn = document.getElementById('scanBtn');
                btn.disabled = true;
                btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> SCANNING...';
                
                const loading = document.getElementById('loading');
                loading.style.display = 'block';
                
                const category = document.getElementById('symbolCategory').value;
                const method = document.getElementById('method').value;
                const timeframe = document.getElementById('timeframe').value;
                let symbols = [];
                
                if (category === 'custom') {
                    const customSymbols = document.getElementById('customSymbols').value;
                    symbols = customSymbols.split(',').map(s => s.trim().toUpperCase()).filter(s => s.length > 0);
                } else {
                    // Fetch symbols from API
                    const response = await fetch(`/symbols?category=${category}`);
                    const data = await response.json();
                    symbols = data.symbols || [];
                }
                
                if (symbols.length === 0) {
                    showMessage('No symbols selected for scanning', 'error');
                    btn.disabled = false;
                    btn.innerHTML = '<i class="fas fa-search"></i> START SCAN';
                    loading.style.display = 'none';
                    return;
                }
                
                document.getElementById('scanProgress').textContent = `Scanning ${symbols.length} symbols...`;
                
                try {
                    const response = await fetch('/scan', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            symbols: symbols,
                            method: method,
                            timeframe: timeframe
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (data.error) {
                        showMessage('Error: ' + data.error, 'error');
                    } else {
                        displayResults(data.results);
                        displayStats(data.summary);
                        showMessage(`Scan completed in ${data.scan_time}s. Found ${data.summary.trade_ready} trade-ready signals.`, 'success');
                    }
                } catch (error) {
                    showMessage('Network error: ' + error.message, 'error');
                } finally {
                    btn.disabled = false;
                    btn.innerHTML = '<i class="fas fa-search"></i> START SCAN';
                    loading.style.display = 'none';
                }
            }
            
            function displayStats(summary) {
                const statsGrid = document.getElementById('statsGrid');
                statsGrid.innerHTML = `
                    <div class="stat-card">
                        <div class="stat-label">Total Scanned</div>
                        <div class="stat-value">${summary.total_scanned}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">BUY Signals</div>
                        <div class="stat-value" style="color: #00ff88;">${summary.buy_signals}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">SELL Signals</div>
                        <div class="stat-value" style="color: #ff6b6b;">${summary.sell_signals}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Trade Ready</div>
                        <div class="stat-value" style="color: #00d4ff;">${summary.trade_ready}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Breakouts</div>
                        <div class="stat-value" style="color: #ffc107;">${summary.breakouts_detected}</div>
                    </div>
                `;
                
                document.getElementById('liveStats').innerHTML = `
                    <div style="font-size: 0.9rem;">
                        <p><i class="fas fa-chart-line"></i> <strong>Method</strong>: ${summary.method}</p>
                        <p><i class="fas fa-clock"></i> <strong>Timeframe</strong>: ${summary.timeframe}</p>
                        <p><i class="fas fa-bolt"></i> <strong>Trade Ready</strong>: ${summary.trade_ready} signals</p>
                        <p><i class="fas fa-fire"></i> <strong>Breakouts</strong>: ${summary.breakouts_detected} detected</p>
                    </div>
                `;
            }
            
            function displayResults(results) {
                const tbody = document.getElementById('resultsBody');
                tbody.innerHTML = '';
                
                results.forEach(result => {
                    const signalClass = result.signal === 'BUY' ? 'signal-buy' : 
                                      result.signal === 'SELL' ? 'signal-sell' : 'signal-neutral';
                    
                    const badges = [];
                    if (result.trade_ready) badges.push('<span class="badge badge-ready">Trade Ready</span>');
                    if (result.breakout_count > 0) badges.push(`<span class="badge badge-breakout">${result.breakout_count} Breakouts</span>`);
                    
                    const progressClass = result.probability >= 70 ? 'progress-high' :
                                         result.probability >= 60 ? 'progress-medium' : 'progress-low';
                    
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td><strong>${result.symbol}</strong></td>
                        <td><span class="${signalClass}">${result.signal}</span> ${badges.join(' ')}</td>
                        <td><strong>₹${result.ltp}</strong></td>
                        <td>
                            <div><small>Entry: ₹${result.entry || '—'}</small></div>
                            <div><small>SL: ₹${result.stop_loss || '—'}</small></div>
                            <div><small>T1: ₹${result.target || '—'}</small></div>
                        </td>
                        <td>
                            <div>${result.probability}%</div>
                            <div class="progress-bar">
                                <div class="progress-fill ${progressClass}" style="width: ${result.probability}%"></div>
                            </div>
                        </td>
                        <td><strong>${result.rr_ratio}:1</strong></td>
                        <td>${result.atr_percent}%</td>
                        <td>
                            <div style="font-size: 0.85rem; max-width: 300px; color: #a0d2ff;">
                                ${result.reason.substring(0, 80)}${result.reason.length > 80 ? '...' : ''}
                            </div>
                            <div style="margin-top: 5px;">
                                ${result.confirmations.slice(0, 2).map(c => `<span class="chip">${c}</span>`).join('')}
                            </div>
                        </td>
                    `;
                    tbody.appendChild(row);
                });
            }
            
            async function analyzeSymbol() {
                const symbol = document.getElementById('analyzeSymbol').value.trim().toUpperCase();
                if (!symbol) {
                    showMessage('Please enter a symbol', 'error');
                    return;
                }
                
                const method = document.getElementById('method').value;
                const timeframe = document.getElementById('timeframe').value;
                
                try {
                    const response = await fetch(`/analyze/${symbol}?method=${method}&timeframe=${timeframe}`);
                    const data = await response.json();
                    
                    if (data.error) {
                        showMessage('Error: ' + data.error, 'error');
                        return;
                    }
                    
                    const analysisDiv = document.getElementById('analysisResult');
                    analysisDiv.innerHTML = `
                        <div style="background: rgba(255, 255, 255, 0.03); border-radius: 10px; padding: 20px;">
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px;">
                                <div class="stat-card">
                                    <div class="stat-label">Signal</div>
                                    <div class="stat-value" style="color: ${data.signal === 'BUY' ? '#00ff88' : data.signal === 'SELL' ? '#ff6b6b' : '#a0d2ff'};">${data.signal}</div>
                                </div>
                                <div class="stat-card">
                                    <div class="stat-label">LTP</div>
                                    <div class="stat-value">₹${data.ltp}</div>
                                </div>
                                <div class="stat-card">
                                    <div class="stat-label">Probability</div>
                                    <div class="stat-value">${data.probability}%</div>
                                </div>
                                <div class="stat-card">
                                    <div class="stat-label">R:R Ratio</div>
                                    <div class="stat-value">${data.rr_ratio}:1</div>
                                </div>
                            </div>
                            
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                                <div>
                                    <h3 style="color: #00d4ff; margin-bottom: 10px;">Trade Levels</h3>
                                    <div style="background: rgba(0, 212, 255, 0.05); padding: 15px; border-radius: 8px;">
                                        <p><strong>Entry:</strong> ₹${data.entry || '—'}</p>
                                        <p><strong>Stop Loss:</strong> ₹${data.stop_loss || '—'}</p>
                                        <p><strong>Target 1:</strong> ₹${data.target || '—'}</p>
                                        <p><strong>Target 2:</strong> ₹${data.target2 || '—'}</p>
                                        <p><strong>Target 3:</strong> ₹${data.target3 || '—'}</p>
                                        <p><strong>Risk Amount:</strong> ₹${data.risk_amount || '—'}</p>
                                        <p><strong>Reward Amount:</strong> ₹${data.reward_amount || '—'}</p>
                                    </div>
                                </div>
                                
                                <div>
                                    <h3 style="color: #00d4ff; margin-bottom: 10px;">Indicators</h3>
                                    <div style="background: rgba(0, 212, 255, 0.05); padding: 15px; border-radius: 8px;">
                                        <p><strong>RSI:</strong> ${data.rsi}</p>
                                        <p><strong>ADX:</strong> ${data.adx} (+DI: ${data.plus_di}, -DI: ${data.minus_di})</p>
                                        <p><strong>ATR:</strong> ₹${data.atr} (${data.atr_percent}%)</p>
                                        <p><strong>Volume Ratio:</strong> ${data.volume_ratio}x</p>
                                        <p><strong>VWAP:</strong> ₹${data.vwap}</p>
                                        <p><strong>EMA9/21:</strong> ₹${data.ema9} / ₹${data.ema21}</p>
                                        <p><strong>MA100:</strong> ₹${data.ma100}</p>
                                    </div>
                                </div>
                            </div>
                            
                            <div style="margin-top: 20px;">
                                <h3 style="color: #00d4ff; margin-bottom: 10px;">Signal Details</h3>
                                <div style="background: rgba(0, 212, 255, 0.05); padding: 15px; border-radius: 8px;">
                                    <p><strong>Reason:</strong> ${data.reason || '—'}</p>
                                    <p><strong>Batches Count:</strong> ${data.batches_count} (BUY: ${data.buy_batch_count}, SELL: ${data.sell_batch_count})</p>
                                    <p><strong>Confirmations:</strong> ${data.confirmation_count}/6</p>
                                    <p><strong>Breakouts:</strong> ${data.breakout_count}</p>
                                    <p><strong>Patterns:</strong> ${data.pattern_count}</p>
                                    <p><strong>Trade Ready:</strong> ${data.trade_ready ? 'Yes' : 'No'}</p>
                                </div>
                            </div>
                        </div>
                    `;
                    
                    showTab('analysis');
                } catch (error) {
                    showMessage('Error analyzing symbol: ' + error.message, 'error');
                }
            }
            
            async function clearCache() {
                try {
                    const response = await fetch('/clear_cache', { method: 'POST' });
                    const data = await response.json();
                    showMessage(data.message, 'success');
                } catch (error) {
                    showMessage('Error clearing cache: ' + error.message, 'error');
                }
            }
            
            async function checkHealth() {
                try {
                    const response = await fetch('/health');
                    const data = await response.json();
                    
                    if (data.status === 'healthy') {
                        showMessage('System is healthy! All components operational.', 'success');
                    } else {
                        showMessage('System issues detected: ' + (data.error || 'Unknown'), 'error');
                    }
                } catch (error) {
                    showMessage('Health check failed: ' + error.message, 'error');
                }
            }
            
            async function loadConfig() {
                try {
                    const response = await fetch('/config');
                    const config = await response.json();
                    
                    let configHtml = '';
                    for (const [key, value] of Object.entries(config)) {
                        const type = typeof value === 'number' ? 'number' : 'text';
                        configHtml += `
                            <div class="form-group">
                                <label>${key.replace(/_/g, ' ').toUpperCase()}</label>
                                <input type="${type}" id="config_${key}" value="${value}" step="${type === 'number' ? '0.01' : '1'}">
                            </div>
                        `;
                    }
                    
                    document.getElementById('configForm').innerHTML = configHtml;
                } catch (error) {
                    showMessage('Error loading config: ' + error.message, 'error');
                }
            }
            
            async function saveConfig() {
                try {
                    const config = {};
                    document.querySelectorAll('#configForm input').forEach(input => {
                        const key = input.id.replace('config_', '');
                        const value = input.type === 'number' ? parseFloat(input.value) : input.value;
                        config[key] = value;
                    });
                    
                    const response = await fetch('/config', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(config)
                    });
                    
                    const data = await response.json();
                    if (data.error) {
                        showMessage('Error: ' + data.error, 'error');
                    } else {
                        showMessage('Configuration saved successfully!', 'success');
                        loadConfig();
                    }
                } catch (error) {
                    showMessage('Error saving config: ' + error.message, 'error');
                }
            }
            
            // Load config when config tab is opened
            document.querySelector('.tab-button[onclick*="config"]').addEventListener('click', loadConfig);
            
            // Initial page load - fetch latest scan if available
            window.addEventListener('load', async () => {
                try {
                    const response = await fetch('/stats');
                    const stats = await response.json();
                    
                    if (stats.last_scan) {
                        document.getElementById('liveStats').innerHTML = `
                            <div style="font-size: 0.9rem;">
                                <p><i class="fas fa-history"></i> <strong>Last Scan</strong>: ${new Date(stats.last_scan).toLocaleString()}</p>
                                <p><i class="fas fa-chart-line"></i> <strong>Method</strong>: ${stats.method}</p>
                                <p><i class="fas fa-clock"></i> <strong>Timeframe</strong>: ${stats.timeframe}</p>
                                <p><i class="fas fa-bolt"></i> <strong>Trade Ready</strong>: ${stats.trade_ready}</p>
                                <p><i class="fas fa-database"></i> <strong>Cache Hit Rate</strong>: ${stats.cache_hit_rate}%</p>
                            </div>
                        `;
                    }
                } catch (error) {
                    console.log('Could not load initial stats');
                }
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template)


# -------------------------------------------------
# Main Entry Point
# -------------------------------------------------
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Enhanced Stock Scanner v2.0 - Production System")
    logger.info("=" * 60)
    logger.info(f"Trading Method: {CURRENT_METHOD}")
    logger.info(f"Timeframe: {CURRENT_TIMEFRAME}")
    logger.info(f"Cache Backend: {'Redis' if REDIS_AVAILABLE else 'Memory'}")
    logger.info(f"Parallel Workers: {MAX_WORKERS}")
    logger.info(f"Symbols Loaded: {len(NIFTY_50)}")
    logger.info(f"Rate Limit Delay: {RATE_LIMIT_DELAY}s")
    logger.info("=" * 60)
    
    # Run the Flask app
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
