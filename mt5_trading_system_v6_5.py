import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
import logging
import sys
import json

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('trading_system.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import DateEntry

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.dates as mdates

import threading
import copy
import time
import json
import os
import itertools


# =============================================================================
# CONFIG
# =============================================================================

class Config:
    """Global konfigürasyon - SCALPING & DAY TRADING VERSION"""

    MODE_PRESETS = {
        'SCALPING': {
            'FORWARD_PERIODS': 10,
            'PROFIT_THRESHOLD': 0.002,
            'MIN_CONFIDENCE': 0.65,
            'STOP_LOSS_PCT': 0.003,
            'TAKE_PROFIT_PCT': 0.006,
            'POSITION_RISK_PCT': 0.005,
            'MAX_TRADES_PER_DAY': 15,
            'MIN_BARS_BETWEEN_TRADES': 3
        },
        'DAY_TRADING': {
            'FORWARD_PERIODS': 20,
            'PROFIT_THRESHOLD': 0.004,
            'MIN_CONFIDENCE': 0.60,
            'STOP_LOSS_PCT': 0.005,
            'TAKE_PROFIT_PCT': 0.012,
            'POSITION_RISK_PCT': 0.008,
            'MAX_TRADES_PER_DAY': 8,
            'MIN_BARS_BETWEEN_TRADES': 5
        },
        'SWING': {
            'FORWARD_PERIODS': 40,
            'PROFIT_THRESHOLD': 0.008,
            'MIN_CONFIDENCE': 0.55,
            'STOP_LOSS_PCT': 0.012,
            'TAKE_PROFIT_PCT': 0.030,
            'POSITION_RISK_PCT': 0.015,
            'MAX_TRADES_PER_DAY': 3,
            'MIN_BARS_BETWEEN_TRADES': 20
        }
    }

    def __init__(self, mode="DAY_TRADING"):
        self.TRADING_MODE = mode
        self.USE_ADAPTIVE_STOPS = True
        self.MAX_CAPITAL_USE = 0.95
        self.PAPER_TRADING = False
        self.USE_ADAPTIVE_THRESHOLD = True
        self.ADAPTIVE_THRESHOLD_ATR_MULT = 1.0
        self.LABEL_MODE = 'PERCENTILE'
        self.LABEL_PERCENTILE = 25
        self.SWING_LOOKBACK = 5
        self.SWING_MIN_ATR = 1.0
        self.SWING_LABEL_WINDOW = 3
        self.USE_FEATURE_SELECTION = True
        self.MAX_FEATURES = 20
        self.MODEL_PRESET = 'CONSERVATIVE'
        self.MODEL_PRESETS = {
            'AGGRESSIVE': {'n_estimators': 200, 'max_depth': 8,
                             'min_samples_split': 20, 'min_samples_leaf': 10},
            'BALANCED':   {'n_estimators': 150, 'max_depth': 5,
                             'min_samples_split': 40, 'min_samples_leaf': 20},
            'CONSERVATIVE':{'n_estimators': 100, 'max_depth': 3,
                             'min_samples_split': 80, 'min_samples_leaf': 40},
        }
        self.SLIPPAGE_ATR_RATIO = 0.1
        self.SIGNAL_QUALITY_FILTER = False
        self.SQ_MIN_CONFIDENCE = 0.72
        self.SQ_MIN_SCORE = 4
        self.SQ_REQUIRE_H4_H1_ALIGN = True
        self.SQ_SESSION_POS_STRICT = True
        self.SQ_BUY_SESSION_MIN = 0.60
        self.SQ_SELL_SESSION_MAX = 0.40
        self.SQ_COOLDOWN_BARS = 60
        self.SQ_REQUIRE_EMA_TREND = True
        self.SQ_MIN_ATR_CHANGE = 0.8
        self.SQ_MAX_ATR_CHANGE = 2.0
        self.WALK_FORWARD_WINDOW = 60
        self.WALK_FORWARD_STEP = 20
        self.LOT_MODE = 'RISK_PCT'
        self.FIXED_LOT = 0.01
        self.MAX_LOT = 1.0
        self.LIVE_RISK_PCT = 0.01
        self.ATR_SL_MULTIPLIER = 1.5
        self.POSITION_MANAGEMENT = True
        self.TRAILING_STOP = True
        self.TRAILING_ATR_MULT = 2.0
        self.TRAILING_ACTIVATE_RR = 1.0
        self.BREAKEVEN_ENABLED = True
        self.BREAKEVEN_RR = 0.5
        self.BREAKEVEN_OFFSET_PIPS = 2
        self.PARTIAL_CLOSE = True
        self.PARTIAL_CLOSE_PCT = 50
        self.PARTIAL_CLOSE_RR = 1.5
        self.TIME_STOP_BARS = 0
        self.REVERSE_SIGNAL_CLOSE = True
        # ── 3-Tier Architecture ──
        self.USE_H4_DIRECTION_FILTER = True   # H4 rule-based direction gate
        self.H4_MIN_SCORE = 4                 # Min H4 criteria score (max 5)
        self.USE_M15_ENTRY_CONFIRM = True      # M15 pullback+candle confirm
        self.VOLATILITY_REGIMES = {
            'LOW':    {'sl_multiplier': 1.0, 'tp_multiplier': 2.0, 'threshold': 0.5},
            'MEDIUM': {'sl_multiplier': 1.5, 'tp_multiplier': 3.0, 'threshold': 1.5},
            'HIGH':   {'sl_multiplier': 2.5, 'tp_multiplier': 5.0, 'threshold': float('inf')}
        }
        self.SL_ATR_TIMEFRAME = 'HTF'
        self.MIN_SL_PCT = 0.005
        self.MIN_TP_RR = 2.0
        self.TRADING_SESSIONS = {
            'TOKYO':              {'start': 0,  'end': 9},
            'LONDON':             {'start': 7,  'end': 16},
            'NEW_YORK':           {'start': 12, 'end': 21},
            'OVERLAP_LONDON_NY':  {'start': 12, 'end': 16}
        }
        self.USE_SESSION_FILTER = False
        self.ALLOWED_SESSIONS = ['LONDON', 'NEW_YORK', 'OVERLAP_LONDON_NY']
        self.SYMBOL = "EURUSD"
        self.set_mode(mode)

    def set_mode(self, mode):
        if mode not in self.MODE_PRESETS:
            mode = "DAY_TRADING"
        self.TRADING_MODE = mode
        preset = self.MODE_PRESETS[mode]
        for key, value in preset.items():
            setattr(self, key, value)


# =============================================================================
# TIMEFRAME UTILITY
# =============================================================================

TIMEFRAME_PRIORITY = ['M1', 'M5', 'M15', 'H1', 'H4', 'D1']


def get_base_timeframe(data_dict: Dict[str, pd.DataFrame]) -> Optional[str]:
    for tf in TIMEFRAME_PRIORITY:
        if tf in data_dict:
            return tf
    return None


def timeframe_to_minutes(tf: str) -> int:
    mapping = {'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
               'H1': 60, 'H4': 240, 'D1': 1440, 'W1': 10080}
    return mapping.get(tf, 5)


# =============================================================================
# MT5 DATA FETCHER
# =============================================================================

class MT5DataFetcher:
    def __init__(self):
        self.connected = False
        self.terminal_path = r"C:\Program Files\MetaTrader 5\terminal64.exe"

    def connect(self):
        if not mt5.initialize(self.terminal_path):
            return False, f"MT5 başlatılamadı! Path: {self.terminal_path}"
        self.connected = True
        account_info = mt5.account_info()
        if account_info:
            return True, f"Bağlantı başarılı! Hesap: {account_info.login}"
        return True, "Bağlantı başarılı!"

    def disconnect(self):
        mt5.shutdown()
        self.connected = False

    def get_available_symbols(self):
        if not self.connected:
            return []
        symbols = mt5.symbols_get()
        forex_pairs, crypto, metals, indices, others = [], [], [], [], []
        major_forex = ['EUR', 'GBP', 'USD', 'JPY', 'AUD', 'NZD', 'CAD', 'CHF']
        metals_list = ['GOLD', 'SILVER', 'XAU', 'XAG', 'XAUUSD', 'XAGUSD']
        crypto_list = ['BTC', 'ETH', 'LTC', 'XRP', 'DOGE']
        for s in symbols:
            if not s.visible:
                continue
            name = s.name.upper()
            if any(metal in name for metal in metals_list):
                metals.append(s.name)
            elif any(c in name for c in crypto_list):
                crypto.append(s.name)
            elif any(pair in name for pair in major_forex) and len(name) <= 10:
                forex_pairs.append(s.name)
            elif any(idx in name for idx in ['US30', 'US500', 'NAS100', 'GER30', 'UK100']):
                indices.append(s.name)
            else:
                others.append(s.name)
        return {
            'forex': sorted(forex_pairs), 'crypto': sorted(crypto),
            'metals': sorted(metals), 'indices': sorted(indices),
            'others': sorted(others),
            'all': sorted([s.name for s in symbols if s.visible])
        }

    def try_symbol_formats(self, base_symbol: str) -> str:
        if not self.connected:
            return base_symbol
        formats = [base_symbol, base_symbol + 'm', base_symbol + '.pro',
                   base_symbol + '.', base_symbol + 'i', base_symbol.lower()]
        all_symbols = [s.name for s in mt5.symbols_get()]
        for fmt in formats:
            if fmt in all_symbols:
                return fmt
        return base_symbol

    def fetch_data(self, symbol: str, timeframe: int, start_date: datetime,
                   end_date: datetime) -> pd.DataFrame:
        if not self.connected:
            raise Exception("MT5 bağlı değil!")
        rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
        if rates is None or len(rates) == 0:
            raise Exception(f"Veri alınamadı! Sembol: {symbol}")
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        return df[['open', 'high', 'low', 'close', 'volume']]

    def fetch_all_timeframes(self, symbol: str, start_date: datetime,
                             end_date: datetime, log_callback=None,
                             mode="DAY_TRADING",
                             progress_callback=None) -> Dict[str, pd.DataFrame]:
        def log(msg):
            if log_callback:
                log_callback(msg)
            else:
                print(msg)

        actual_symbol = self.try_symbol_formats(symbol)

        if mode == "SCALPING":
            timeframes = {'H1': mt5.TIMEFRAME_H1, 'M15': mt5.TIMEFRAME_M15,
                          'M5': mt5.TIMEFRAME_M5, 'M1': mt5.TIMEFRAME_M1}
        elif mode == "DAY_TRADING":
            timeframes = {'H4': mt5.TIMEFRAME_H4, 'H1': mt5.TIMEFRAME_H1,
                          'M15': mt5.TIMEFRAME_M15, 'M5': mt5.TIMEFRAME_M5}
        else:
            timeframes = {'D1': mt5.TIMEFRAME_D1, 'H4': mt5.TIMEFRAME_H4,
                          'H1': mt5.TIMEFRAME_H1, 'M15': mt5.TIMEFRAME_M15}

        data = {}
        total_tf = len(timeframes)
        for tf_idx, (name, tf) in enumerate(timeframes.items(), 1):
            try:
                if progress_callback:
                    progress_callback(name, tf_idx, total_tf)
                log(f"   {name} verisi çekiliyor...")
                df = self.fetch_data(actual_symbol, tf, start_date, end_date)
                if len(df) == 0:
                    log(f"   ⚠️  {name}: Veri yok")
                    continue
                data[name] = df
                log(f"   ✅ {name}: {len(df)} bar")
            except Exception as e:
                log(f"   ❌ {name}: {str(e)}")

        if len(data) == 0:
            raise Exception("Hiç veri alınamadı!")
        return data


# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

class TechnicalIndicators:
    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if len(df) < 50:
            raise ValueError(f"Yetersiz veri! {len(df)} bar var")
        try:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            with np.errstate(divide='ignore', invalid='ignore'):
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))

            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']

            df['bb_middle'] = df['close'].rolling(20).mean()
            std = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + (std * 2)
            df['bb_lower'] = df['bb_middle'] - (std * 2)
            bb_range = (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)
            df['bb_position'] = (df['close'] - df['bb_lower']) / bb_range

            for period in [9, 21, 50]:
                df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = true_range.rolling(14).mean()
            df['atr_pct'] = df['atr'] / df['close'] * 100

            low_14 = df['low'].rolling(14).min()
            high_14 = df['high'].rolling(14).max()
            stoch_range = (high_14 - low_14).replace(0, np.nan)
            df['stoch_k'] = 100 * (df['close'] - low_14) / stoch_range
            df['stoch_d'] = df['stoch_k'].rolling(3).mean()

            typical_price = (df['high'] + df['low'] + df['close']) / 3
            tp_sma = typical_price.rolling(20).mean()
            tp_mad = typical_price.rolling(20).apply(
                lambda x: np.abs(x - x.mean()).mean(), raw=True).replace(0, np.nan)
            df['cci'] = (typical_price - tp_sma) / (0.015 * tp_mad)

            high_14_wr = df['high'].rolling(14).max()
            low_14_wr = df['low'].rolling(14).min()
            wr_range = (high_14_wr - low_14_wr).replace(0, np.nan)
            df['williams_r'] = -100 * (high_14_wr - df['close']) / wr_range

            tp_mfi = (df['high'] + df['low'] + df['close']) / 3
            has_volume = df['volume'].sum() > 0 and (df['volume'] > 0).mean() > 0.5
            if has_volume:
                raw_money_flow = tp_mfi * df['volume']
                tp_diff = tp_mfi.diff()
                pos_flow = raw_money_flow.where(tp_diff > 0, 0).rolling(14).sum()
                neg_flow = raw_money_flow.where(tp_diff < 0, 0).rolling(14).sum().replace(0, np.nan)
                df['mfi'] = 100 - (100 / (1 + pos_flow / neg_flow))
            else:
                df['mfi'] = 50.0

            df['momentum'] = df['close'] - df['close'].shift(10)
            df['roc'] = df['close'].pct_change(10) * 100

            if has_volume:
                df['volume_sma'] = df['volume'].rolling(20).mean()
                vol_sma = df['volume_sma'].replace(0, np.nan)
                df['volume_ratio'] = (df['volume'] / vol_sma).fillna(1.0).clip(0.01, 10.0)
                df['volume_trend'] = (df['volume'].rolling(5).mean() / vol_sma).fillna(1.0).clip(0.01, 10.0)
                vol_std = df['volume'].rolling(20).std().replace(0, np.nan)
                df['volume_spike'] = ((df['volume'] - df['volume_sma']) / vol_std).fillna(0).clip(-3, 5)
            else:
                df['volume_sma'] = 1.0
                df['volume_ratio'] = 1.0
                df['volume_trend'] = 1.0
                df['volume_spike'] = 0.0

            df['price_ema9_dist'] = (df['close'] - df['ema_9']) / df['close'] * 100
            df['price_ema21_dist'] = (df['close'] - df['ema_21']) / df['close'] * 100
            df['rsi_deviation'] = df['rsi'] - 50
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100
            df['ema9_slope'] = df['ema_9'].pct_change(3) * 100
            df['ema21_slope'] = df['ema_21'].pct_change(5) * 100
            atr_safe = df['atr'].replace(0, np.nan)
            df['ema_ribbon'] = (df['ema_9'] - df['ema_21']) / atr_safe

            session_high = df['high'].rolling(12).max()
            session_low = df['low'].rolling(12).min()
            session_range = (session_high - session_low).replace(0, np.nan)
            df['session_position'] = (df['close'] - session_low) / session_range

            if has_volume:
                tp_vwap = (df['high'] + df['low'] + df['close']) / 3
                cum_tp_vol = (tp_vwap * df['volume']).rolling(20).sum()
                cum_vol = df['volume'].rolling(20).sum().replace(0, np.nan)
                df['vwap'] = cum_tp_vol / cum_vol
                vwap_safe = df['vwap'].replace(0, np.nan)
                df['vwap_dist'] = (df['close'] - df['vwap']) / vwap_safe * 100
                df['vwap_above'] = (df['close'] > df['vwap']).astype(float)
            else:
                df['vwap'] = df['close']
                df['vwap_dist'] = 0.0
                df['vwap_above'] = 0.5

            df['momentum_short'] = df['close'].pct_change(5) * 100
            df['momentum_long'] = df['close'].pct_change(20) * 100
            df['momentum_divergence'] = df['momentum_short'] - df['momentum_long']

            lookback_div = 14
            price_min_lb = df['close'].rolling(lookback_div).min()
            price_max_lb = df['close'].rolling(lookback_div).max()
            rsi_min_lb = df['rsi'].rolling(lookback_div).min()
            rsi_max_lb = df['rsi'].rolling(lookback_div).max()
            price_lower = df['close'] <= price_min_lb.shift(1) * 1.001
            rsi_higher = df['rsi'] > rsi_min_lb.shift(1)
            df['rsi_bull_div'] = (price_lower & rsi_higher).astype(float)
            price_higher = df['close'] >= price_max_lb.shift(1) * 0.999
            rsi_lower = df['rsi'] < rsi_max_lb.shift(1)
            df['rsi_bear_div'] = (price_higher & rsi_lower).astype(float)
            macd_min_lb = df['macd_hist'].rolling(lookback_div).min()
            macd_max_lb = df['macd_hist'].rolling(lookback_div).max()
            df['macd_bull_div'] = (price_lower & (df['macd_hist'] > macd_min_lb.shift(1))).astype(float)
            df['macd_bear_div'] = (price_higher & (df['macd_hist'] < macd_max_lb.shift(1))).astype(float)
            df['divergence_score'] = (df['rsi_bull_div'] + df['macd_bull_div']
                                       - df['rsi_bear_div'] - df['macd_bear_div'])

            body = df['close'] - df['open']
            body_abs = body.abs()
            candle_range = (df['high'] - df['low']).replace(0, np.nan)
            upper_wick = df['high'] - df[['close', 'open']].max(axis=1)
            lower_wick = df[['close', 'open']].min(axis=1) - df['low']
            df['candle_hammer'] = ((lower_wick >= 2 * body_abs) &
                                    (upper_wick <= body_abs * 0.5) &
                                    (body_abs > candle_range * 0.1)).astype(float)
            df['candle_shooting_star'] = ((upper_wick >= 2 * body_abs) &
                                           (lower_wick <= body_abs * 0.5) &
                                           (body_abs > candle_range * 0.1)).astype(float)
            df['candle_doji'] = (body_abs <= candle_range * 0.1).astype(float)
            prev_body = body.shift(1)
            df['candle_bull_engulf'] = ((prev_body < 0) & (body > 0) &
                                         (df['open'] <= df['close'].shift(1)) &
                                         (df['close'] >= df['open'].shift(1))).astype(float)
            df['candle_bear_engulf'] = ((prev_body > 0) & (body < 0) &
                                         (df['open'] >= df['close'].shift(1)) &
                                         (df['close'] <= df['open'].shift(1))).astype(float)
            df['candle_marubozu'] = (body_abs >= candle_range * 0.8).astype(float)
            df['candle_score'] = (df['candle_hammer'] + df['candle_bull_engulf']
                                   - df['candle_shooting_star'] - df['candle_bear_engulf'])
            df['body_ratio'] = body_abs / candle_range
            df['upper_shadow'] = upper_wick / candle_range
            df['lower_shadow'] = lower_wick / candle_range

            rsi_min = df['rsi'].rolling(14).min()
            rsi_max = df['rsi'].rolling(14).max()
            rsi_range = (rsi_max - rsi_min).replace(0, np.nan)
            df['stoch_rsi'] = (df['rsi'] - rsi_min) / rsi_range * 100

            if hasattr(df.index, 'hour'):
                hour = df.index.hour
                df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
                df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

            atr_sma = df['atr'].rolling(20).mean()
            df['atr_change'] = df['atr'] / atr_sma.replace(0, np.nan)
            df['range_atr_ratio'] = (df['high'] - df['low']) / atr_safe

            direction = np.sign(df['close'] - df['open'])
            streak = direction.copy()
            for i in range(1, len(streak)):
                if direction.iloc[i] == direction.iloc[i-1] and direction.iloc[i] != 0:
                    streak.iloc[i] = streak.iloc[i-1] + direction.iloc[i]
                else:
                    streak.iloc[i] = direction.iloc[i]
            df['consecutive_bars'] = streak
            df['exhaustion'] = df['consecutive_bars'].clip(-8, 8) / 8.0
            df['wick_rejection'] = lower_wick - upper_wick
            df['bb_extreme'] = np.where(df['close'] <= df['bb_lower'], -1.0,
                                np.where(df['close'] >= df['bb_upper'], 1.0, 0.0))
            df['bb_return'] = np.where(
                (df['bb_extreme'].shift(1) == -1.0) & (df['close'] > df['bb_lower']), 1.0,
                np.where((df['bb_extreme'].shift(1) == 1.0) & (df['close'] < df['bb_upper']), -1.0, 0.0))

            if has_volume:
                vol_z = ((df['volume'] - df['volume_sma']) /
                         df['volume'].rolling(20).std().replace(0, np.nan)).fillna(0)
                df['volume_climax'] = np.where(
                    (vol_z > 2.0) & (df['bb_position'] < 0.2), 1.0,
                    np.where((vol_z > 2.0) & (df['bb_position'] > 0.8), -1.0, 0.0))
            else:
                df['volume_climax'] = 0.0

            delta_s = df['close'].diff()
            gain_s = delta_s.where(delta_s > 0, 0).rolling(7).mean()
            loss_s = -delta_s.where(delta_s < 0, 0).rolling(7).mean()
            with np.errstate(divide='ignore', invalid='ignore'):
                rsi_7 = 100 - (100 / (1 + gain_s / loss_s))
            df['rsi_divergence'] = rsi_7 - df['rsi']

            recent_low = df['low'].rolling(20).min()
            recent_high = df['high'].rolling(20).max()
            recent_range = (recent_high - recent_low).replace(0, np.nan)
            df['dist_from_low'] = (df['close'] - recent_low) / recent_range
            df['dist_from_high'] = (recent_high - df['close']) / recent_range

            mom_5 = df['close'].pct_change(5)
            mom_1 = df['close'].pct_change(1)
            df['momentum_reversal'] = np.where(
                (mom_5 < 0) & (mom_1 > 0), 1.0,
                np.where((mom_5 > 0) & (mom_1 < 0), -1.0, 0.0))

            df['ema_cross'] = (df['ema_9'] > df['ema_21']).astype(int)
            df = df.replace([np.inf, -np.inf], np.nan)
            return df
        except Exception as e:
            raise ValueError(f"Teknik indikatör hatası: {str(e)}")


# =============================================================================
# H4 DIRECTION FILTER  (Tier-1: rule-based direction gate)
# =============================================================================

class H4DirectionFilter:
    """
    H4 rule-based direction filter — 5 criteria.

    Tier-1 in the 3-tier architecture:
      H4 Direction → ML Entry Timing (M15) → M15 Candle Confirmation

    Returns BUY (1) or SELL (-1) only when ≥ H4_MIN_SCORE criteria agree.
    A NEUTRAL (0) result blocks any trade for that bar.
    """

    CRITERIA = [
        "EMA9 > EMA21 (trend)",
        "RSI vs 50 (momentum)",
        "MACD Histogram sign",
        "BB Position vs 0.5",
        "Price vs EMA50",
    ]

    def get_direction(self, h4_df: pd.DataFrame,
                      min_score: int = 4) -> Tuple[int, int, str]:
        """
        Evaluate 5 rule-based criteria on H4 data.

        Returns
        -------
        direction : int   1=BUY, -1=SELL, 0=NEUTRAL
        score     : int   number of criteria favouring winning direction
        reason    : str   human-readable breakdown
        """
        if h4_df is None or len(h4_df) < 52:
            return 0, 0, "H4 yetersiz veri"

        try:
            df = TechnicalIndicators.calculate_all(h4_df)
        except Exception as e:
            return 0, 0, f"H4 hesaplama hatası: {e}"

        last = df.iloc[-1]
        buy_pts, sell_pts = 0, 0
        tags = []

        # ── Kriter 1: EMA Trend ──
        if last['ema_9'] > last['ema_21']:
            buy_pts += 1;  tags.append("EMA↑")
        else:
            sell_pts += 1; tags.append("EMA↓")

        # ── Kriter 2: RSI ──
        rsi = last['rsi'] if not np.isnan(last['rsi']) else 50
        if rsi > 50:
            buy_pts += 1;  tags.append(f"RSI↑{rsi:.0f}")
        else:
            sell_pts += 1; tags.append(f"RSI↓{rsi:.0f}")

        # ── Kriter 3: MACD Histogram ──
        macd_h = last['macd_hist'] if not np.isnan(last['macd_hist']) else 0
        if macd_h > 0:
            buy_pts += 1;  tags.append("MACD↑")
        else:
            sell_pts += 1; tags.append("MACD↓")

        # ── Kriter 4: Bollinger Band Position ──
        bb_pos = last['bb_position'] if not np.isnan(last['bb_position']) else 0.5
        if bb_pos > 0.5:
            buy_pts += 1;  tags.append(f"BB↑{bb_pos:.2f}")
        else:
            sell_pts += 1; tags.append(f"BB↓{bb_pos:.2f}")

        # ── Kriter 5: Price vs EMA50 ──
        if last['close'] > last['ema_50']:
            buy_pts += 1;  tags.append("P>E50")
        else:
            sell_pts += 1; tags.append("P<E50")

        tag_str = " | ".join(tags)

        if buy_pts >= min_score:
            return 1, buy_pts, f"BUY {buy_pts}/5: {tag_str}"
        elif sell_pts >= min_score:
            return -1, sell_pts, f"SELL {sell_pts}/5: {tag_str}"
        else:
            return 0, max(buy_pts, sell_pts), f"NEUTRAL {buy_pts}v{sell_pts}: {tag_str}"


# =============================================================================
# M15 ENTRY CONFIRMER  (Tier-3: pullback + candle timing)
# =============================================================================

class M15EntryConfirmer:
    """
    M15 entry confirmation — Tier-3 in the 3-tier architecture.

    Checks for a pullback towards EMA9 AND a confirming candle pattern.
    Avoids entering at extreme extension from moving average.
    """

    def check_entry(self, m15_df: pd.DataFrame,
                    direction: int) -> Tuple[bool, str]:
        """
        Parameters
        ----------
        m15_df    : pd.DataFrame  M15 OHLCV data (already has indicators)
        direction : int           1=BUY, -1=SELL

        Returns
        -------
        confirmed : bool
        reason    : str
        """
        if m15_df is None or len(m15_df) < 30:
            return True, "M15 veri yok — atlandı"

        try:
            if 'ema_9' not in m15_df.columns:
                df = TechnicalIndicators.calculate_all(m15_df)
            else:
                df = m15_df
        except Exception:
            return True, "M15 hesaplama hatası — atlandı"

        last = df.iloc[-1]
        close = last['close']
        ema9  = last['ema_9']
        rsi   = last['rsi'] if not np.isnan(last['rsi']) else 50

        if direction == 1:  # ── BUY ──
            # Pullback: fiyat EMA9'a ≤ %0.2 uzaklıkta veya altında
            pullback = close <= ema9 * 1.002
            # Candle onayı: hammer veya bullish engulfing
            candle_ok = (last.get('candle_hammer', 0) == 1 or
                         last.get('candle_bull_engulf', 0) == 1)
            # RSI destekliyor: aşırı satımda değil, momentum dönüyor
            rsi_ok = 35 <= rsi <= 68
            # Wick rejection bullish
            wick_ok = last.get('wick_rejection', 0) > 0

            confirmed = pullback and (candle_ok or rsi_ok or wick_ok)
            details = (f"pullback={pullback}({close:.5f}≤{ema9*1.002:.5f}) "
                       f"candle={candle_ok} rsi={rsi:.0f} wick={wick_ok:.1f}")
            reason = f"M15 BUY: {'✅' if confirmed else '❌'} {details}"

        else:  # ── SELL ──
            pullback = close >= ema9 * 0.998
            candle_ok = (last.get('candle_shooting_star', 0) == 1 or
                         last.get('candle_bear_engulf', 0) == 1)
            rsi_ok = 32 <= rsi <= 65
            wick_ok = last.get('wick_rejection', 0) < 0

            confirmed = pullback and (candle_ok or rsi_ok or wick_ok)
            details = (f"pullback={pullback}({close:.5f}≥{ema9*0.998:.5f}) "
                       f"candle={candle_ok} rsi={rsi:.0f} wick={wick_ok:.1f}")
            reason = f"M15 SELL: {'✅' if confirmed else '❌'} {details}"

        return confirmed, reason


# =============================================================================
# ML MODEL
# =============================================================================

class MLModel:
    """Machine Learning Model - Tier-2: entry timing on M15 features"""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        preset = self.config.MODEL_PRESETS.get(
            self.config.MODEL_PRESET, self.config.MODEL_PRESETS['BALANCED'])
        self.model = RandomForestClassifier(
            n_estimators=preset['n_estimators'],
            max_depth=preset['max_depth'],
            min_samples_split=preset['min_samples_split'],
            min_samples_leaf=preset['min_samples_leaf'],
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.trained = False
        self.feature_names = []
        self.selected_features = []
        self.rfe_selector = None
        self.model_preset_used = self.config.MODEL_PRESET
        self.last_features = None

    def prepare_features(self, data_dict: Dict[str, pd.DataFrame],
                         log_callback=None) -> pd.DataFrame:
        def log(msg):
            if log_callback:
                log_callback(msg)

        base_tf = get_base_timeframe(data_dict)
        if not base_tf:
            raise ValueError("Hiçbir timeframe bulunamadı!")

        base_df = data_dict[base_tf].copy()
        base_index = base_df.index
        all_features = []

        feature_cols = [
            'rsi', 'macd_hist', 'bb_position', 'atr_pct',
            'stoch_k', 'stoch_d', 'cci', 'williams_r', 'mfi',
            'volume_ratio', 'volume_trend', 'volume_spike',
            'body_ratio', 'upper_shadow', 'lower_shadow',
            'candle_hammer', 'candle_shooting_star', 'candle_doji',
            'candle_bull_engulf', 'candle_bear_engulf', 'candle_marubozu',
            'candle_score',
            'rsi_bull_div', 'rsi_bear_div', 'macd_bull_div', 'macd_bear_div',
            'divergence_score',
            'price_ema9_dist', 'price_ema21_dist', 'roc',
            'rsi_deviation', 'bb_width', 'ema9_slope', 'ema21_slope',
            'ema_ribbon', 'session_position', 'momentum_short',
            'momentum_long', 'momentum_divergence', 'stoch_rsi',
            'atr_change', 'range_atr_ratio',
            'vwap_dist', 'vwap_above',
            'exhaustion', 'wick_rejection',
            'bb_extreme', 'bb_return', 'volume_climax',
            'rsi_divergence', 'dist_from_low', 'dist_from_high',
            'momentum_reversal', 'ema_cross',
            'hour_sin', 'hour_cos',
        ]

        for tf_name, df in data_dict.items():
            df = TechnicalIndicators.calculate_all(df)
            has_vol = df['volume'].sum() > 0 and (df['volume'] > 0).mean() > 0.5
            if not has_vol:
                log(f"   ⚠️  {tf_name}: volume yok → volume_ratio=1.0")
            existing_cols = [c for c in feature_cols if c in df.columns]
            features = df[existing_cols].copy()
            if tf_name != base_tf:
                features = features.reindex(base_index, method='ffill')
            features.columns = [f'{tf_name}_{col}' for col in features.columns]
            all_features.append(features)

        combined = pd.concat(all_features, axis=1)
        combined = combined.reindex(sorted(combined.columns), axis=1)
        combined = combined.ffill()
        if combined.isna().sum().sum() > 0:
            combined = combined.dropna()
        self.feature_names = list(combined.columns)
        return combined

    def select_features(self, X: pd.DataFrame, y: pd.Series, log_callback=None):
        def log(msg):
            if log_callback: log_callback(msg)
            else: logger.info(msg)

        if not self.config.USE_FEATURE_SELECTION:
            self.selected_features = list(X.columns)
            return X

        n_features = min(self.config.MAX_FEATURES, X.shape[1])
        log(f"\n🔍 Feature Selection: {X.shape[1]}→{n_features}...")

        self.rfe_selector = RFE(
            estimator=RandomForestClassifier(n_estimators=50, max_depth=5,
                                             random_state=42, n_jobs=-1),
            n_features_to_select=n_features, step=0.2)
        X_scaled = self.scaler.fit_transform(X)
        self.rfe_selector.fit(X_scaled, y)
        self.selected_features = X.columns[self.rfe_selector.support_].tolist()
        importances = self.rfe_selector.estimator_.feature_importances_
        indices = np.argsort(importances)[::-1]
        log(f"✅ {len(self.selected_features)} feature seçildi:")
        for i, idx in enumerate(indices[:10]):
            fname = self.selected_features[idx] if idx < len(self.selected_features) else "N/A"
            log(f"   {i+1}. {fname}: {importances[idx]:.4f}")
        return X[self.selected_features]

    def create_labels(self, df: pd.DataFrame, log_callback=None) -> np.ndarray:
        def log(msg):
            if log_callback: log_callback(msg)
            print(msg)

        future_returns = df['close'].shift(-self.config.FORWARD_PERIODS) / df['close'] - 1
        mode = self.config.LABEL_MODE
        log(f"\n   🏷️  LABEL MODE: {mode}")

        if mode == 'SWING':
            lb = self.config.SWING_LOOKBACK
            min_atr = self.config.SWING_MIN_ATR
            label_win = self.config.SWING_LABEL_WINDOW
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            atr_vals = df['atr'].values if 'atr' in df.columns else np.full(len(df), np.nan)
            n = len(df)
            swing_labels = np.zeros(n, dtype=int)
            swing_lows, swing_highs = [], []
            for i in range(lb, n - lb):
                is_sl = all(lows[i-j] > lows[i] and lows[i+j] > lows[i] for j in range(1, lb+1))
                if is_sl:
                    pre_high = highs[max(0, i-lb*3):i].max()
                    atr_i = atr_vals[i] if not np.isnan(atr_vals[i]) else 1.0
                    if (pre_high - lows[i]) >= atr_i * min_atr:
                        swing_lows.append(i)
                is_sh = all(highs[i-j] < highs[i] and highs[i+j] < highs[i] for j in range(1, lb+1))
                if is_sh:
                    pre_low = lows[max(0, i-lb*3):i].min()
                    atr_i = atr_vals[i] if not np.isnan(atr_vals[i]) else 1.0
                    if (highs[i] - pre_low) >= atr_i * min_atr:
                        swing_highs.append(i)
            for idx in swing_lows:
                swing_labels[max(0, idx-label_win):idx+1] = 1
            for idx in swing_highs:
                swing_labels[max(0, idx-label_win):idx+1] = -1
            buy_mask = swing_labels == 1
            sell_mask = swing_labels == -1
            n_buy = buy_mask.sum(); n_sell = sell_mask.sum(); n_hold = n - n_buy - n_sell
            log(f"   Swing dip: {len(swing_lows)}, tepe: {len(swing_highs)}")
            atr_med = (df['atr'] / df['close']).median() if 'atr' in df.columns else 0.005
            self._effective_threshold_buy = atr_med * min_atr
            self._effective_threshold_sell = atr_med * min_atr
            labels = swing_labels

        elif mode == 'PERCENTILE':
            valid_returns = future_returns.dropna()
            pct = self.config.LABEL_PERCENTILE
            upper_thresh = np.percentile(valid_returns, 100 - pct)
            lower_thresh = np.percentile(valid_returns, pct)
            log(f"   P{100-pct}: {upper_thresh*100:.4f}% | P{pct}: {lower_thresh*100:.4f}%")
            buy_mask = future_returns > upper_thresh
            sell_mask = future_returns < lower_thresh
            n_total = len(future_returns.dropna())
            n_buy = buy_mask.sum(); n_sell = sell_mask.sum(); n_hold = n_total - n_buy - n_sell
            self._effective_threshold_buy = upper_thresh
            self._effective_threshold_sell = abs(lower_thresh)
            labels = np.where(buy_mask, 1, np.where(sell_mask, -1, 0))

        elif mode == 'ATR' and 'atr' in df.columns:
            atr_pct = df['atr'] / df['close']
            effective_thresh = max((atr_pct * self.config.ADAPTIVE_THRESHOLD_ATR_MULT).median(), 0.0005)
            log(f"   ATR threshold: {effective_thresh*100:.4f}%")
            buy_mask = future_returns > effective_thresh
            sell_mask = future_returns < -effective_thresh
            n_total = len(future_returns.dropna())
            n_buy = buy_mask.sum(); n_sell = sell_mask.sum(); n_hold = n_total - n_buy - n_sell
            self._effective_threshold_buy = effective_thresh
            self._effective_threshold_sell = effective_thresh
            labels = np.where(buy_mask, 1, np.where(sell_mask, -1, 0))

        else:
            threshold_buy = self.config.PROFIT_THRESHOLD
            threshold_sell = -self.config.PROFIT_THRESHOLD
            log(f"   Fixed: {self.config.PROFIT_THRESHOLD*100:.3f}%")
            buy_mask = future_returns > threshold_buy
            sell_mask = future_returns < threshold_sell
            n_total = len(future_returns.dropna())
            n_buy = buy_mask.sum(); n_sell = sell_mask.sum(); n_hold = n_total - n_buy - n_sell
            self._effective_threshold_buy = threshold_buy
            self._effective_threshold_sell = abs(threshold_sell)
            labels = np.where(buy_mask, 1, np.where(sell_mask, -1, 0))

        log(f"   BUY: {n_buy} | SELL: {n_sell} | HOLD: {n_hold}")
        return labels

    def train(self, data_dict: Dict[str, pd.DataFrame],
              log_callback=None, progress_callback=None) -> Dict:
        def log(msg):
            if log_callback: log_callback(msg)
        def progress(val, detail=""):
            if progress_callback: progress_callback(val, detail)

        progress(15, "Feature'lar hesaplanıyor...")
        features = self.prepare_features(data_dict, log_callback=log_callback)

        progress(30, "Label'lar oluşturuluyor...")
        base_tf = get_base_timeframe(data_dict)
        df_base = TechnicalIndicators.calculate_all(data_dict[base_tf].copy())
        df_base = df_base.reindex(features.index)
        labels = pd.Series(self.create_labels(df_base, log_callback=log_callback),
                           index=df_base.index).reindex(features.index)

        valid_mask = ~features.isna().any(axis=1) & ~labels.isna()
        features = features[valid_mask]
        labels = labels[valid_mask]
        trade_mask = labels != 0
        features = features[trade_mask]
        labels = labels[trade_mask]

        if len(features) < 100:
            raise ValueError(f"Yeterli veri yok! {len(features)} sample")

        progress(45, "Train/Test bölünüyor...")
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, shuffle=False)

        progress(48, "Feature selection...")
        X_train = self.select_features(X_train, y_train, log_callback=log_callback)
        X_test = X_test[self.selected_features]

        progress(55, "Ölçekleniyor...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled  = self.scaler.transform(X_test)

        preset = self.config.MODEL_PRESETS.get(self.config.MODEL_PRESET, {})
        log(f"\n   🧠 MODEL: {self.config.MODEL_PRESET} "
            f"depth={preset.get('max_depth')} leaf={preset.get('min_samples_leaf')}")

        progress(58, "Cross-validation...")
        from sklearn.model_selection import TimeSeriesSplit, cross_val_score
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train,
                                     cv=tscv, scoring='accuracy', n_jobs=-1)
        cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
        log(f"   CV: {cv_mean:.4f} ± {cv_std:.4f}")

        progress(70, "RandomForest eğitiliyor...")
        self.model.fit(X_train_scaled, y_train)
        self.trained = True

        progress(85, "Metrikler...")
        train_score = self.model.score(X_train_scaled, y_train)
        test_score  = self.model.score(X_test_scaled, y_test)
        score_gap   = train_score - test_score
        log(f"   Train: {train_score:.4f} | Test: {test_score:.4f} | Gap: {score_gap:.4f}")
        if score_gap > 0.10:
            log("   ❌ OVERFIT!")
        elif score_gap > 0.05:
            log("   ⚠️  Hafif overfit")
        else:
            log("   ✅ İyi genelleştirme")

        progress(98, "Tamamlandı!")
        return {
            'train_accuracy': train_score, 'test_accuracy': test_score,
            'score_gap': score_gap, 'cv_mean': cv_mean, 'cv_std': cv_std,
            'total_samples': len(labels),
            'buy_signals': int((labels == 1).sum()),
            'sell_signals': int((labels == -1).sum()),
            'selected_features': len(self.selected_features),
            'model_preset': self.config.MODEL_PRESET
        }

    def predict(self, data_dict: Dict[str, pd.DataFrame]) -> Tuple[int, float]:
        if not self.trained:
            raise Exception("Model eğitilmemiş!")
        features = self.prepare_features(data_dict)
        if self.selected_features:
            features = features[self.selected_features]
        self.last_features = features.iloc[-1]
        latest_scaled = self.scaler.transform(features.iloc[-1:].values)
        signal = self.model.predict(latest_scaled)[0]
        probs  = self.model.predict_proba(latest_scaled)[0]
        return int(signal), float(max(probs))

    def predict_batch(self, data_dict: Dict[str, pd.DataFrame],
                      progress_callback=None) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        if not self.trained:
            raise Exception("Model eğitilmemiş!")
        if progress_callback:
            progress_callback(12, "Feature'lar hesaplanıyor (batch)...")
        features = self.prepare_features(data_dict)
        if self.selected_features:
            features = features[self.selected_features]
        if progress_callback:
            progress_callback(25, f"Tahminler ({len(features):,} bar)...")
        features_scaled = self.scaler.transform(features.values)
        signals = self.model.predict(features_scaled)
        probas  = self.model.predict_proba(features_scaled)
        if progress_callback:
            progress_callback(35, "Batch tamamlandı!")
        return features, signals.astype(int), probas.max(axis=1)

    def save_model(self, filepath: str, symbol: str = "", log_callback=None):
        from joblib import dump as joblib_dump
        def log(msg):
            if log_callback: log_callback(msg)
        if not self.trained:
            raise Exception("Model eğitilmemiş!")
        save_data = {
            'version': '6.5',
            'symbol': symbol,
            'saved_at': datetime.now().isoformat(),
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'selected_features': self.selected_features,
            'model_preset_used': self.model_preset_used,
            'config_snapshot': {
                'TRADING_MODE': self.config.TRADING_MODE,
                'FORWARD_PERIODS': self.config.FORWARD_PERIODS,
                'PROFIT_THRESHOLD': self.config.PROFIT_THRESHOLD,
                'LABEL_MODE': self.config.LABEL_MODE,
                'LABEL_PERCENTILE': self.config.LABEL_PERCENTILE,
                'MIN_CONFIDENCE': self.config.MIN_CONFIDENCE,
                'USE_FEATURE_SELECTION': self.config.USE_FEATURE_SELECTION,
                'MAX_FEATURES': self.config.MAX_FEATURES,
                'MODEL_PRESET': self.config.MODEL_PRESET,
                'USE_ADAPTIVE_STOPS': self.config.USE_ADAPTIVE_STOPS,
                'SL_ATR_TIMEFRAME': self.config.SL_ATR_TIMEFRAME,
                'SWING_LOOKBACK': self.config.SWING_LOOKBACK,
                'SWING_MIN_ATR': self.config.SWING_MIN_ATR,
                'SWING_LABEL_WINDOW': self.config.SWING_LABEL_WINDOW,
            }
        }
        joblib_dump(save_data, filepath, compress=3)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        log(f"✅ Model kaydedildi: {filepath} ({size_mb:.1f} MB)")

    @classmethod
    def load_model(cls, filepath: str, config: 'Config' = None,
                   log_callback=None) -> 'MLModel':
        from joblib import load as joblib_load
        def log(msg):
            if log_callback: log_callback(msg)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model bulunamadı: {filepath}")
        try:
            data = joblib_load(filepath)
        except Exception:
            import pickle
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        if config is None:
            config = Config(data['config_snapshot'].get('TRADING_MODE', 'DAY_TRADING'))
        for key, val in data['config_snapshot'].items():
            if hasattr(config, key):
                setattr(config, key, val)
        instance = cls(config)
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.feature_names = data['feature_names']
        instance.selected_features = data['selected_features']
        instance.model_preset_used = data.get('model_preset_used', 'UNKNOWN')
        instance.trained = True
        log(f"✅ Model yüklendi: {filepath} | {data.get('symbol')} | {data.get('saved_at','?')[:10]}")
        return instance


# =============================================================================
# RISK MANAGER
# =============================================================================

class RiskManager:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.current_open_trades = 0
        self.atr_window = []
        self.atr_lookback = 50
        self.daily_trades = 0
        self.last_trade_bar = -999
        self.last_trade_time = None
        self.current_day = None
        self.daily_realized_pnl = 0
        self.daily_floating_pnl = 0

    def can_trade(self, current_time, current_bar_index) -> Tuple[bool, str]:
        if self.current_open_trades >= 1:
            return False, "Maksimum açık pozisyon"
        day = current_time.date() if hasattr(current_time, 'date') else current_time
        if self.current_day != day:
            self.current_day = day
            self.daily_trades = 0
            self.daily_realized_pnl = 0
            self.daily_floating_pnl = 0
        if self.daily_trades >= self.config.MAX_TRADES_PER_DAY:
            return False, f"Günlük limit ({self.config.MAX_TRADES_PER_DAY})"
        bars_since = current_bar_index - self.last_trade_bar
        if bars_since < self.config.MIN_BARS_BETWEEN_TRADES:
            return False, f"Çok yakın ({bars_since} bar)"
        return True, "OK"

    def can_trade_live(self, current_time, mt5_open_count: int,
                       floating_pnl: float = 0) -> Tuple[bool, str]:
        self.current_open_trades = mt5_open_count
        if mt5_open_count >= 1:
            return False, "Açık pozisyon var"
        day = current_time.date() if hasattr(current_time, 'date') else current_time
        if self.current_day != day:
            self.current_day = day
            self.daily_trades = 0
            self.daily_realized_pnl = 0
        if self.daily_trades >= self.config.MAX_TRADES_PER_DAY:
            return False, f"Günlük limit ({self.config.MAX_TRADES_PER_DAY})"
        if hasattr(self, 'last_trade_time') and self.last_trade_time:
            elapsed = (current_time - self.last_trade_time).total_seconds()
            cooldown = self.config.MIN_BARS_BETWEEN_TRADES * 60
            if elapsed < cooldown:
                return False, f"Cooldown: {cooldown-elapsed:.0f}s"
        total_pnl = self.daily_realized_pnl + floating_pnl
        max_daily_loss = -self.config.MAX_CAPITAL_USE * 0.05
        if total_pnl < max_daily_loss:
            return False, f"Günlük kayıp limiti (realized:{self.daily_realized_pnl:.2f} float:{floating_pnl:.2f})"
        return True, "OK"

    def update_trade_live(self, opened: bool = False, closed: bool = False, pnl: float = 0):
        if opened:
            self.daily_trades += 1
            self.last_trade_time = datetime.now()
        elif closed:
            self.daily_realized_pnl += pnl
            self.current_open_trades = max(0, self.current_open_trades - 1)

    def update_trade(self, opened: bool = False, closed: bool = False,
                     bar_index: int = 0, pnl: float = 0):
        if opened:
            self.current_open_trades += 1
            self.daily_trades += 1
            self.last_trade_bar = bar_index
        elif closed:
            self.daily_realized_pnl += pnl
            self.current_open_trades = max(0, self.current_open_trades - 1)

    def detect_volatility_regime(self, current_atr: float) -> str:
        self.atr_window.append(current_atr)
        if len(self.atr_window) > self.atr_lookback:
            self.atr_window.pop(0)
        if len(self.atr_window) < 20:
            return 'MEDIUM'
        avg_atr = np.mean(self.atr_window)
        if avg_atr == 0:
            return 'MEDIUM'
        atr_ratio = current_atr / avg_atr
        if atr_ratio < self.config.VOLATILITY_REGIMES['LOW']['threshold']:
            return 'LOW'
        elif atr_ratio < self.config.VOLATILITY_REGIMES['MEDIUM']['threshold']:
            return 'MEDIUM'
        else:
            return 'HIGH'

    def calculate_adaptive_stops(self, current_price: float, signal: int,
                                 atr: float, volatility_regime: str) -> Tuple[float, float]:
        if not self.config.USE_ADAPTIVE_STOPS or atr == 0:
            if signal == 1:
                return (current_price * (1 - self.config.STOP_LOSS_PCT),
                        current_price * (1 + self.config.TAKE_PROFIT_PCT))
            else:
                return (current_price * (1 + self.config.STOP_LOSS_PCT),
                        current_price * (1 - self.config.TAKE_PROFIT_PCT))

        regime_cfg = self.config.VOLATILITY_REGIMES.get(
            volatility_regime, self.config.VOLATILITY_REGIMES['MEDIUM'])
        sl_dist = atr * regime_cfg['sl_multiplier']
        tp_dist = atr * regime_cfg['tp_multiplier']
        min_sl = current_price * self.config.MIN_SL_PCT
        if sl_dist < min_sl:
            sl_dist = min_sl
            tp_dist = sl_dist * self.config.MIN_TP_RR
        if signal == 1:
            return current_price - sl_dist, current_price + tp_dist
        else:
            return current_price + sl_dist, current_price - tp_dist

    def check_margin(self, symbol: str, volume: float, price: float,
                     order_type: int) -> Tuple[bool, float]:
        account_info = mt5.account_info()
        if not account_info:
            return False, 0
        margin_required = mt5.order_calc_margin(order_type, symbol, volume, price)
        if margin_required is None:
            return False, 0
        required_with_buffer = margin_required * 1.1
        if required_with_buffer > account_info.margin_free:
            max_lot = (account_info.margin_free / 1.1) / (margin_required / volume)
            volume = max(0.01, np.floor(max_lot * 100) / 100)
            return (False, volume) if volume >= 0.01 else (False, 0)
        return True, volume


# =============================================================================
# CONFIG MANAGER
# =============================================================================

class ConfigManager:
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = config_dir
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
        self.configs = {}
        self.master_file = os.path.join(config_dir, "master_config.json")
        self.load_master_config()

    def save_config(self, symbol: str, config_dict: Dict, performance_metrics: Dict = None):
        config_data = {
            'symbol': symbol,
            'saved_at': datetime.now().isoformat(),
            'parameters': config_dict,
            'performance': performance_metrics or {},
            'version': '6.5'
        }
        symbol_file = os.path.join(self.config_dir, f"{symbol}_config.json")
        with open(symbol_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        self.configs[symbol] = config_data
        self.update_master_config(symbol, config_data)

    def load_config(self, symbol: str) -> Optional[Dict]:
        if symbol in self.configs:
            return self.configs[symbol]
        symbol_file = os.path.join(self.config_dir, f"{symbol}_config.json")
        if not os.path.exists(symbol_file):
            return None
        with open(symbol_file, 'r') as f:
            config_data = json.load(f)
        self.configs[symbol] = config_data
        return config_data

    def update_master_config(self, symbol: str, config_data: Dict):
        master_config = {}
        if os.path.exists(self.master_file):
            with open(self.master_file, 'r') as f:
                master_config = json.load(f)
        master_config[symbol] = {
            'last_updated': config_data['saved_at'],
            'file': f"{symbol}_config.json",
            'performance': config_data.get('performance', {})
        }
        with open(self.master_file, 'w') as f:
            json.dump(master_config, f, indent=2)

    def load_master_config(self):
        if not os.path.exists(self.master_file):
            self.master_config = {}
            return
        with open(self.master_file, 'r') as f:
            self.master_config = json.load(f)

    def list_configs(self) -> List[str]:
        return [f.replace('_config.json', '')
                for f in os.listdir(self.config_dir)
                if f.endswith('_config.json') and f != 'master_config.json']


# =============================================================================
# PARAMETER OPTIMIZER
# =============================================================================

class ParameterOptimizer:
    def __init__(self, model: MLModel, data_dict: Dict[str, pd.DataFrame], symbol: str):
        self.model = model
        self.data_dict = data_dict
        self.symbol = symbol

    def quick_optimize(self, log_callback=None, progress_callback=None) -> Dict:
        def log(msg):
            if log_callback: log_callback(msg)
            else: print(msg)
        def progress(val, detail=""):
            if progress_callback: progress_callback(val, detail)

        base_tf = get_base_timeframe(self.data_dict)
        df = self.data_dict[base_tf]
        volatility = df['close'].pct_change().std() * 100
        log(f"\n{'='*60}\nOPTİMİZASYON: {self.symbol} | vol={volatility:.3f}%\n{'='*60}")

        if base_tf in ['M1', 'M5']:
            param_grid = {'PROFIT_THRESHOLD': [0.001, 0.002, 0.003],
                          'FORWARD_PERIODS': [5, 10, 15], 'MIN_CONFIDENCE': [0.65, 0.70]}
        elif base_tf == 'M15':
            param_grid = {'PROFIT_THRESHOLD': [0.002, 0.004, 0.005],
                          'FORWARD_PERIODS': [15, 20, 25], 'MIN_CONFIDENCE': [0.60, 0.65]}
        else:
            if volatility < 0.8:
                param_grid = {'PROFIT_THRESHOLD': [0.003, 0.004, 0.005],
                              'FORWARD_PERIODS': [25, 30, 35], 'MIN_CONFIDENCE': [0.60, 0.65]}
            elif volatility < 1.5:
                param_grid = {'PROFIT_THRESHOLD': [0.005, 0.006, 0.007],
                              'FORWARD_PERIODS': [30, 35, 40], 'MIN_CONFIDENCE': [0.60, 0.65]}
            else:
                param_grid = {'PROFIT_THRESHOLD': [0.006, 0.008, 0.010],
                              'FORWARD_PERIODS': [30, 40, 50], 'MIN_CONFIDENCE': [0.60, 0.65, 0.70]}

        keys = list(param_grid.keys())
        combinations = list(itertools.product(*param_grid.values()))
        log(f"Kombinasyon: {len(combinations)}")

        best_score = float('-inf')
        best_params = None
        best_metrics = None

        for i, combo in enumerate(combinations, 1):
            pct = 5 + (i / len(combinations)) * 90
            progress(pct, f"{i}/{len(combinations)} kombinasyon")
            test_config = Config()
            params_dict = dict(zip(keys, combo))
            for key, value in params_dict.items():
                setattr(test_config, key, value)
            log(f"[{i}/{len(combinations)}] {params_dict}")
            try:
                temp_model = MLModel(test_config)
                metrics = temp_model.train(self.data_dict)
                backtester = Backtester(10000, test_config)
                bt = backtester.run(self.data_dict, temp_model)
                if 'error' in bt:
                    continue
                score = (bt.get('profit_factor', 0) *
                         bt.get('win_rate', 0) / 100 *
                         max(0, 1 - metrics['score_gap']) *
                         min(1.0, bt.get('total_trades', 0) / 20))
                log(f"   PF={bt.get('profit_factor',0):.2f} score={score:.4f}")
                if score > best_score:
                    best_score = score
                    best_params = params_dict
                    best_metrics = bt
                    log("   ⭐ YENİ EN İYİ!")
            except Exception as e:
                log(f"   ❌ {e}")

        progress(98, "✅ Tamamlandı!")
        return {'best_params': best_params, 'best_score': best_score, 'best_metrics': best_metrics}


# =============================================================================
# SIGNAL QUALITY FILTER
# =============================================================================

class SignalQualityFilter:
    def __init__(self, config: Config):
        self.config = config
        self.last_trade_bar_idx = -9999

    def calculate_quality_score(self, features_row: pd.Series,
                                 signal: int, confidence: float,
                                 bar_idx: int = 0,
                                 log_callback=None) -> Tuple[bool, int, str]:
        if not self.config.SIGNAL_QUALITY_FILTER:
            return True, 6, "Filter devre dışı"
        if confidence < self.config.SQ_MIN_CONFIDENCE:
            return False, 0, f"Confidence düşük: {confidence:.2f}"
        bars_since = bar_idx - self.last_trade_bar_idx
        if bars_since < self.config.SQ_COOLDOWN_BARS:
            return False, 0, f"Cooldown: {bars_since}/{self.config.SQ_COOLDOWN_BARS}"

        def get_feat(name, default=np.nan):
            return features_row.get(name, default) if name in features_row.index else default

        h4_session_pos = get_feat('H4_session_position', 0.5)
        h4_ema9_dist   = get_feat('H4_price_ema9_dist', 0.0)
        h4_ema_ribbon  = get_feat('H4_ema_ribbon', 0.0)
        h4_bb_pos      = get_feat('H4_bb_position', 0.5)
        h4_stoch_rsi   = get_feat('H4_stoch_rsi', 50.0)
        h4_atr_change  = get_feat('H4_atr_change', 1.0)
        h1_ema_ribbon  = get_feat('H1_ema_ribbon', 0.0)
        h1_session_pos = get_feat('H1_session_position', 0.5)

        if not np.isnan(h4_atr_change):
            if h4_atr_change < self.config.SQ_MIN_ATR_CHANGE:
                return False, 0, f"Vol düşük: ATR={h4_atr_change:.2f}"
            if h4_atr_change > self.config.SQ_MAX_ATR_CHANGE:
                return False, 0, f"Vol yüksek: ATR={h4_atr_change:.2f}"

        score = 0
        reasons = []
        if signal == 1:
            if (h4_session_pos >= self.config.SQ_BUY_SESSION_MIN
                    if self.config.SQ_SESSION_POS_STRICT else h4_session_pos >= 0.50):
                score += 1
            else:
                reasons.append(f"SP={h4_session_pos:.2f}")
            if h4_ema9_dist > 0: score += 1
            else: reasons.append(f"EMA9={h4_ema9_dist:.2f}")
            if h4_ema_ribbon > 0: score += 1
            else: reasons.append(f"Ribbon={h4_ema_ribbon:.2f}")
            if h4_bb_pos >= 0.50: score += 1
            else: reasons.append(f"BB={h4_bb_pos:.2f}")
            if 40 <= h4_stoch_rsi <= 85: score += 1
            else: reasons.append(f"StRSI={h4_stoch_rsi:.0f}")
            h1_ok = h1_ema_ribbon > 0 or h1_session_pos > 0.50
            if h1_ok: score += 1
            else: reasons.append("H1↓")
            if self.config.SQ_REQUIRE_H4_H1_ALIGN and not h1_ok:
                return False, score, f"H4+H1 uyumsuz BUY: {', '.join(reasons)}"
            if self.config.SQ_REQUIRE_EMA_TREND and h4_ema_ribbon <= 0:
                return False, score, f"EMA trend yanlış BUY"
        else:
            if (h4_session_pos <= self.config.SQ_SELL_SESSION_MAX
                    if self.config.SQ_SESSION_POS_STRICT else h4_session_pos <= 0.50):
                score += 1
            else:
                reasons.append(f"SP={h4_session_pos:.2f}")
            if h4_ema9_dist < 0: score += 1
            else: reasons.append(f"EMA9={h4_ema9_dist:.2f}")
            if h4_ema_ribbon < 0: score += 1
            else: reasons.append(f"Ribbon={h4_ema_ribbon:.2f}")
            if h4_bb_pos <= 0.50: score += 1
            else: reasons.append(f"BB={h4_bb_pos:.2f}")
            if 15 <= h4_stoch_rsi <= 60: score += 1
            else: reasons.append(f"StRSI={h4_stoch_rsi:.0f}")
            h1_ok = h1_ema_ribbon < 0 or h1_session_pos < 0.50
            if h1_ok: score += 1
            else: reasons.append("H1↑")
            if self.config.SQ_REQUIRE_H4_H1_ALIGN and not h1_ok:
                return False, score, f"H4+H1 uyumsuz SELL: {', '.join(reasons)}"
            if self.config.SQ_REQUIRE_EMA_TREND and h4_ema_ribbon >= 0:
                return False, score, f"EMA trend yanlış SELL"

        if score < self.config.SQ_MIN_SCORE:
            return False, score, f"Skor {score}/6 < {self.config.SQ_MIN_SCORE}"
        return True, score, f"✅ {score}/6"

    def on_trade_opened(self, bar_idx: int):
        self.last_trade_bar_idx = bar_idx

    def reset(self):
        self.last_trade_bar_idx = -9999


# =============================================================================
# BACKTESTER
# =============================================================================

class Backtester:
    def __init__(self, initial_capital: float = 10000, config: Config = None):
        self.config = config or Config()
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = []
        self.trades = []
        self.equity_curve = []
        self.signal_quality_filter = SignalQualityFilter(self.config)
        self.risk_manager = RiskManager(self.config)
        if self.config.TRADING_MODE == "SCALPING":
            self.spread_pct = 0.0002; self.commission_pct = 0.00005
        elif self.config.TRADING_MODE == "DAY_TRADING":
            self.spread_pct = 0.0001; self.commission_pct = 0.00003
        else:
            self.spread_pct = 0.0001; self.commission_pct = 0.0

    def calculate_position_size(self, current_price: float, atr: float) -> float:
        risk_amount = self.capital * self.config.POSITION_RISK_PCT
        if atr > 0 and self.config.USE_ADAPTIVE_STOPS:
            regime = self.risk_manager.detect_volatility_regime(atr)
            sl_mult = self.config.VOLATILITY_REGIMES.get(
                regime, self.config.VOLATILITY_REGIMES['MEDIUM'])['sl_multiplier']
            sl_distance = atr * sl_mult
            position_value = (risk_amount / (sl_distance / current_price)
                              if sl_distance > 0 else risk_amount / self.config.STOP_LOSS_PCT)
        else:
            position_value = risk_amount / self.config.STOP_LOSS_PCT
        return min(position_value, self.capital * self.config.MAX_CAPITAL_USE)

    def apply_slippage(self, price: float, atr: float, direction: int) -> float:
        return price + atr * self.config.SLIPPAGE_ATR_RATIO * direction

    def run(self, data_dict: Dict[str, pd.DataFrame],
            model: MLModel, progress_callback=None) -> Dict:
        def progress(val, detail=""):
            if progress_callback: progress_callback(val, detail)

        base_tf = get_base_timeframe(data_dict)
        if not base_tf:
            raise Exception("Hiçbir timeframe bulunamadı!")

        progress(5, "İndikatörler hesaplanıyor...")
        df_base = TechnicalIndicators.calculate_all(data_dict[base_tf].copy())

        progress(8, "Batch tahminler...")
        features_df, all_signals, all_confidences = model.predict_batch(
            data_dict, progress_callback=progress_callback)

        common_index = df_base.index.intersection(features_df.index)
        signal_series = pd.Series(0, index=df_base.index, dtype=int)
        confidence_series = pd.Series(0.0, index=df_base.index)
        signal_series.loc[common_index] = pd.Series(
            all_signals, index=features_df.index).reindex(common_index, fill_value=0)
        confidence_series.loc[common_index] = pd.Series(
            all_confidences, index=features_df.index).reindex(common_index, fill_value=0.0)

        split_idx = int(len(df_base) * 0.8)
        test_data = df_base.iloc[split_idx:]

        # HTF ATR for SL/TP
        htf_atr_series = None
        if self.config.SL_ATR_TIMEFRAME == 'HTF':
            for tf in ['D1', 'H4', 'H1', 'M15', 'M5', 'M1']:
                if tf in data_dict and tf != base_tf:
                    df_htf = TechnicalIndicators.calculate_all(data_dict[tf].copy())
                    htf_atr_series = df_htf['atr'].reindex(df_base.index, method='ffill')
                    progress(38, f"HTF ATR: {tf}")
                    break

        self.equity_curve = []
        total_bars = len(test_data) - self.config.FORWARD_PERIODS - 1
        last_pct = 0
        progress(40, f"Backtest: {total_bars:,} bar...")

        for i in range(total_bars):
            pct = 40 + (i / total_bars) * 55
            if pct - last_pct >= 2:
                progress(pct, f"Backtest: {i:,}/{total_bars:,} ({len(self.trades)} trade)")
                last_pct = pct

            current_bar = test_data.iloc[i]
            current_price = current_bar['close']
            current_time = test_data.index[i]
            atr = current_bar['atr']

            sl_atr = atr
            if htf_atr_series is not None and current_time in htf_atr_series.index:
                htf_val = htf_atr_series.loc[current_time]
                if not np.isnan(htf_val) and htf_val > 0:
                    sl_atr = htf_val

            signal = int(signal_series.loc[current_time])
            confidence = float(confidence_series.loc[current_time])

            try:
                if (signal != 0 and confidence >= self.config.MIN_CONFIDENCE
                        and len(self.positions) == 0):
                    sq_passed = True
                    if self.config.SIGNAL_QUALITY_FILTER:
                        if current_time in features_df.index:
                            sq_passed, _, _ = self.signal_quality_filter.calculate_quality_score(
                                features_df.loc[current_time], signal, confidence, bar_idx=i)
                        else:
                            sq_passed = False

                    if sq_passed:
                        can_trade, _ = self.risk_manager.can_trade(current_time, i)
                        if can_trade:
                            pos_val = self.calculate_position_size(current_price, sl_atr)
                            vol_regime = self.risk_manager.detect_volatility_regime(sl_atr)
                            sl, tp = self.risk_manager.calculate_adaptive_stops(
                                current_price, signal, sl_atr, vol_regime)
                            entry_price = self.apply_slippage(current_price, atr, signal)
                            if signal == 1:
                                sl_dist = (entry_price - sl) / entry_price * 100
                                tp_dist = (tp - entry_price) / entry_price * 100
                            else:
                                sl_dist = (sl - entry_price) / entry_price * 100
                                tp_dist = (entry_price - tp) / entry_price * 100
                            pos = {
                                'entry_time': current_time, 'entry_price': entry_price,
                                'position_value': pos_val, 'signal': signal,
                                'stop_loss': sl, 'take_profit': tp,
                                'confidence': confidence, 'volatility_regime': vol_regime,
                                'atr': atr, 'sl_distance_pct': sl_dist, 'tp_distance_pct': tp_dist,
                                'indicators': {k: current_bar.get(k, np.nan) for k in [
                                    'rsi','macd','macd_signal','macd_hist','bb_upper','bb_middle',
                                    'bb_lower','bb_position','bb_width','ema_9','ema_21','ema_50',
                                    'ema_ribbon','stoch_k','stoch_d','stoch_rsi','momentum',
                                    'momentum_divergence','roc','atr_pct','atr_change',
                                    'session_position','volume_ratio']}
                            }
                            self.positions.append(pos)
                            self.risk_manager.update_trade(opened=True, bar_index=i)
                            if self.config.SIGNAL_QUALITY_FILTER:
                                self.signal_quality_filter.on_trade_opened(i)

                unrealized_pnl = 0
                for pos in self.positions[:]:
                    exit_price = self.apply_slippage(current_price, atr, -pos['signal'])
                    if pos['signal'] == 1:
                        unrealized_pnl += (exit_price - pos['entry_price']) / pos['entry_price'] * pos['position_value']
                        if current_price <= pos['stop_loss']:
                            self._close_position(pos, exit_price, current_time, 'SL')
                        elif current_price >= pos['take_profit']:
                            self._close_position(pos, exit_price, current_time, 'TP')
                    else:
                        unrealized_pnl += (pos['entry_price'] - exit_price) / pos['entry_price'] * pos['position_value']
                        if current_price >= pos['stop_loss']:
                            self._close_position(pos, exit_price, current_time, 'SL')
                        elif current_price <= pos['take_profit']:
                            self._close_position(pos, exit_price, current_time, 'TP')

                self.equity_curve.append({
                    'time': current_time, 'equity': self.capital + unrealized_pnl,
                    'realized_equity': self.capital, 'unrealized_pnl': unrealized_pnl,
                    'price': current_price
                })
            except Exception:
                continue

        if self.positions:
            final_bar = test_data.iloc[-1]
            for pos in self.positions[:]:
                self._close_position(pos, final_bar['close'], test_data.index[-1], 'END')

        if len(self.trades) == 0:
            return {'error': 'Hiç trade açılmadı!'}

        progress(96, "Metrikler...")
        return self._calculate_metrics()

    def _close_position(self, pos, exit_price, exit_time, reason):
        cost = pos['position_value'] * (self.spread_pct * 2 + self.commission_pct * 2)
        if pos['signal'] == 1:
            pct_chg = (exit_price - pos['entry_price']) / pos['entry_price']
        else:
            pct_chg = (pos['entry_price'] - exit_price) / pos['entry_price']
        net_pnl = pos['position_value'] * pct_chg - cost
        self.capital += net_pnl
        self.trades.append({**pos, 'exit_time': exit_time, 'exit_price': exit_price,
                             'pnl': net_pnl, 'gross_pnl': pos['position_value'] * pct_chg,
                             'costs': cost, 'return_pct': pct_chg * 100, 'reason': reason})
        self.positions.remove(pos)
        self.risk_manager.update_trade(closed=True)

    def _calculate_metrics(self) -> Dict:
        if not self.trades:
            return {'error': 'Hiç trade yapılmadı'}
        df_trades = pd.DataFrame(self.trades)
        df_equity = pd.DataFrame(self.equity_curve)
        winning = df_trades[df_trades['pnl'] > 0]
        losing  = df_trades[df_trades['pnl'] < 0]
        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100
        returns = df_equity['realized_equity'].pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() != 0 else 0
        rolling_max = df_equity['equity'].expanding().max()
        max_dd = ((df_equity['equity'] - rolling_max) / rolling_max * 100).min()

        daily_max_dd = 0.0
        if 'time' in df_equity.columns and len(df_equity) > 0:
            df_eq = df_equity.copy()
            df_eq['date'] = pd.to_datetime(df_eq['time']).dt.date
            dds = []
            for _, eqs in df_eq.groupby('date')['equity']:
                if len(eqs) < 2: continue
                pk = eqs.expanding().max()
                dds.append(((eqs - pk) / pk * 100).min())
            if dds: daily_max_dd = min(dds)

        gross_profit = winning['pnl'].sum() if len(winning) > 0 else 0
        gross_loss   = abs(losing['pnl'].sum()) if len(losing) > 0 else 0
        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_return': total_return,
            'total_trades': len(df_trades),
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': len(winning) / len(df_trades) * 100,
            'avg_win': winning['pnl'].mean() if len(winning) > 0 else 0,
            'avg_loss': losing['pnl'].mean() if len(losing) > 0 else 0,
            'profit_factor': gross_profit / gross_loss if gross_loss > 0 else 0,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'daily_max_dd': daily_max_dd,
            'equity_curve': df_equity,
            'trades': df_trades
        }


# =============================================================================
# WALK-FORWARD ANALYZER
# =============================================================================

class WalkForwardAnalyzer:
    def __init__(self, config: Config, initial_capital: float = 10000):
        self.config = config
        self.initial_capital = initial_capital

    def run(self, data_dict: Dict[str, pd.DataFrame],
            log_callback=None, progress_callback=None) -> Dict:
        def log(msg):
            if log_callback: log_callback(msg)
            else: logger.info(msg)
        def progress(val, detail=""):
            if progress_callback: progress_callback(val, detail)

        base_tf = get_base_timeframe(data_dict)
        if not base_tf:
            raise Exception("Hiçbir timeframe bulunamadı!")

        df_base = data_dict[base_tf]
        total_bars = len(df_base)
        bars_per_day = int(1440 / timeframe_to_minutes(base_tf))
        train_bars = self.config.WALK_FORWARD_WINDOW * bars_per_day
        step_bars  = self.config.WALK_FORWARD_STEP  * bars_per_day

        min_train = max(500, train_bars)
        if total_bars < min_train + step_bars:
            raise ValueError(f"Yetersiz veri! {total_bars} bar")

        windows = []
        start = 0
        while start + train_bars + step_bars <= total_bars:
            windows.append({'train_start': start, 'train_end': start + train_bars,
                             'test_start': start + train_bars,
                             'test_end': min(start + train_bars + step_bars, total_bars)})
            start += step_bars

        if not windows:
            raise ValueError("Walk-forward penceresi oluşturulamadı!")

        log(f"\n{'='*60}\n📊 WALK-FORWARD | {len(windows)} pencere\n{'='*60}")

        all_trades, all_equity, window_metrics = [], [], []
        capital = self.initial_capital

        for w_idx, window in enumerate(windows):
            pct = 5 + (w_idx / len(windows)) * 90
            progress(pct, f"Pencere {w_idx+1}/{len(windows)}")
            log(f"\n--- Pencere {w_idx+1}/{len(windows)} ---")

            train_data, test_data_dict = {}, {}
            for tf_name, df in data_dict.items():
                base_start = df_base.index[window['train_start']]
                base_train_end = df_base.index[window['train_end'] - 1]
                base_test_end = df_base.index[min(window['test_end']-1, len(df_base)-1)]
                train_data[tf_name] = df[(df.index >= base_start) & (df.index <= base_train_end)].copy()
                test_data_dict[tf_name] = df[df.index <= base_test_end].copy()

            try:
                window_model = MLModel(self.config)
                metrics = window_model.train(train_data, log_callback=log)
                gap = metrics['score_gap']
                log(f"   Train: {metrics['train_accuracy']:.3f} | Test: {metrics['test_accuracy']:.3f} | Gap: {gap:.3f}")
                if gap > 0.20:
                    log(f"   ⚠️  Yüksek overfitting!")
            except Exception as e:
                log(f"   ❌ Eğitim: {e}"); continue

            try:
                window_bt = Backtester(capital, self.config)
                bt_result = window_bt.run(test_data_dict, window_model)
                if 'error' not in bt_result:
                    n_trades = len(bt_result['trades'])
                    win_rate = bt_result['win_rate']
                    pf = bt_result['profit_factor']
                    ret = bt_result['total_return']
                    capital = bt_result['final_capital']
                    log(f"   📈 {n_trades} trade WR={win_rate:.1f}% PF={pf:.2f} R={ret:.2f}%")
                    all_trades.append(bt_result['trades'])
                    if 'equity_curve' in bt_result:
                        all_equity.append(bt_result['equity_curve'])
                    window_metrics.append({
                        'window': w_idx+1, 'train_acc': metrics['train_accuracy'],
                        'test_acc': metrics['test_accuracy'], 'gap': gap,
                        'n_trades': n_trades, 'win_rate': win_rate,
                        'profit_factor': pf, 'return_pct': ret, 'capital': capital
                    })
                else:
                    log(f"   ⚠️  {bt_result['error']}")
            except Exception as e:
                log(f"   ❌ Backtest: {e}"); continue

        progress(96, "Sonuçlar...")
        if not window_metrics:
            return {'error': 'Hiçbir pencere başarılı olmadı!'}

        df_windows = pd.DataFrame(window_metrics)
        combined_trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
        combined_equity = pd.concat(all_equity, ignore_index=True) if all_equity else pd.DataFrame()
        total_return = (capital - self.initial_capital) / self.initial_capital * 100

        log(f"\n{'='*60}\n📊 WALK-FORWARD SONUÇLARI\n{'='*60}")
        log(f"   Karlı pencereler: {(df_windows['return_pct'] > 0).sum()}/{len(window_metrics)}")
        log(f"   Toplam trade: {df_windows['n_trades'].sum()}")
        log(f"   Ort WR: {df_windows['win_rate'].mean():.1f}%")
        log(f"   Ort PF: {df_windows['profit_factor'].mean():.2f}")
        log(f"   Getiri: {total_return:.2f}% | Final: ${capital:,.0f}")

        progress(100, "✅ Walk-Forward tamamlandı!")
        return {
            'initial_capital': self.initial_capital, 'final_capital': capital,
            'total_return': total_return, 'total_trades': int(df_windows['n_trades'].sum()),
            'avg_win_rate': df_windows['win_rate'].mean(),
            'avg_profit_factor': df_windows['profit_factor'].mean(),
            'avg_gap': df_windows['gap'].mean(),
            'n_windows': len(windows), 'successful_windows': len(window_metrics),
            'profitable_windows': int((df_windows['return_pct'] > 0).sum()),
            'window_metrics': df_windows, 'trades': combined_trades, 'equity_curve': combined_equity
        }


# =============================================================================
# POSITION MANAGER
# =============================================================================

class PositionManager:
    def __init__(self, config: Config, log_callback=None):
        self.config = config
        self._log_callback = log_callback
        self._position_states = {}

    def log(self, msg):
        if self._log_callback: self._log_callback(msg)

    def register_position(self, ticket, entry_price, sl, tp, order_type, volume, atr):
        risk_distance = abs(entry_price - sl)
        self._position_states[ticket] = {
            'entry_price': entry_price, 'original_sl': sl, 'original_tp': tp,
            'current_sl': sl, 'order_type': order_type, 'volume': volume,
            'original_volume': volume, 'atr': atr, 'risk_distance': risk_distance,
            'breakeven_done': False, 'partial_done': False, 'trailing_active': False,
            'bars_since_open': 0, 'best_price': entry_price
        }

    def unregister_position(self, ticket):
        if ticket in self._position_states:
            del self._position_states[ticket]

    def manage_positions(self, open_positions, current_price, current_signal=0,
                         modify_sl_callback=None, close_position_callback=None,
                         partial_close_callback=None):
        if not self.config.POSITION_MANAGEMENT:
            return
        for pos in open_positions:
            if isinstance(pos, dict):
                ticket = pos['ticket']; pos_type = 'BUY' if pos.get('type') == 0 else 'SELL'
            else:
                ticket = pos.ticket; pos_type = 'BUY' if pos.type == 0 else 'SELL'
            if ticket not in self._position_states:
                continue
            state = self._position_states[ticket]
            state['bars_since_open'] += 1
            entry = state['entry_price']; risk_dist = state['risk_distance']
            if risk_dist <= 0: continue
            if pos_type == 'BUY':
                rr = (current_price - entry) / risk_dist
                if current_price > state['best_price']: state['best_price'] = current_price
            else:
                rr = (entry - current_price) / risk_dist
                if current_price < state['best_price']: state['best_price'] = current_price

            if self.config.REVERSE_SIGNAL_CLOSE and current_signal != 0:
                is_rev = (pos_type == 'BUY' and current_signal == -1) or (pos_type == 'SELL' and current_signal == 1)
                if is_rev and rr > -0.5:
                    self.log(f"   🔄 TERS SİNYAL: #{ticket} R:R={rr:.2f}")
                    if close_position_callback: close_position_callback(pos)
                    self.unregister_position(ticket); continue

            if self.config.TIME_STOP_BARS > 0 and state['bars_since_open'] >= self.config.TIME_STOP_BARS:
                if rr < 0.3:
                    self.log(f"   ⏰ TIME STOP: #{ticket}")
                    if close_position_callback: close_position_callback(pos)
                    self.unregister_position(ticket); continue

            if self.config.BREAKEVEN_ENABLED and not state['breakeven_done'] and rr >= self.config.BREAKEVEN_RR:
                pip_val = 0.0001
                offset = self.config.BREAKEVEN_OFFSET_PIPS * pip_val
                if pos_type == 'BUY':
                    new_sl = entry + offset
                    if new_sl > state['current_sl']:
                        if modify_sl_callback and modify_sl_callback(ticket, new_sl):
                            state['current_sl'] = new_sl; state['breakeven_done'] = True
                            self.log(f"   🔒 BE: #{ticket} SL→{new_sl:.5f}")
                else:
                    new_sl = entry - offset
                    if new_sl < state['current_sl'] or state['current_sl'] == 0:
                        if modify_sl_callback and modify_sl_callback(ticket, new_sl):
                            state['current_sl'] = new_sl; state['breakeven_done'] = True
                            self.log(f"   🔒 BE: #{ticket} SL→{new_sl:.5f}")

            if self.config.PARTIAL_CLOSE and not state['partial_done'] and rr >= self.config.PARTIAL_CLOSE_RR:
                close_vol = round(state['original_volume'] * self.config.PARTIAL_CLOSE_PCT / 100, 2)
                if close_vol >= 0.01:
                    self.log(f"   ✂️  PARTIAL: #{ticket} {close_vol}lot R:R={rr:.2f}")
                    if partial_close_callback and partial_close_callback(pos, close_vol):
                        state['partial_done'] = True
                        state['volume'] = round(state['volume'] - close_vol, 2)

            if self.config.TRAILING_STOP and rr >= self.config.TRAILING_ACTIVATE_RR:
                state['trailing_active'] = True
                trail_dist = state['atr'] * self.config.TRAILING_ATR_MULT
                if pos_type == 'BUY':
                    new_tsl = state['best_price'] - trail_dist
                    if new_tsl > state['current_sl']:
                        if modify_sl_callback and modify_sl_callback(ticket, new_tsl):
                            state['current_sl'] = new_tsl
                            self.log(f"   📈 TRAIL: #{ticket} SL→{new_tsl:.5f}")
                else:
                    new_tsl = state['best_price'] + trail_dist
                    if new_tsl < state['current_sl'] or state['current_sl'] == 0:
                        if modify_sl_callback and modify_sl_callback(ticket, new_tsl):
                            state['current_sl'] = new_tsl
                            self.log(f"   📉 TRAIL: #{ticket} SL→{new_tsl:.5f}")

    def get_position_info(self, ticket):
        return self._position_states.get(ticket)


# =============================================================================
# LIVE TRADER  — 3-Tier Architecture
#   Tier-1: H4DirectionFilter (rule-based BUY/SELL gate)
#   Tier-2: MLModel (M15-dominant features, entry timing)
#   Tier-3: M15EntryConfirmer (pullback + candle)
# =============================================================================

class LiveTrader:
    def __init__(self, config: Config, model: MLModel, capital: float = 10000):
        self.config = config
        self.model = model
        self.initial_capital = capital
        self.risk_manager = RiskManager(config)
        self.position_manager = PositionManager(config)
        self.signal_quality_filter = SignalQualityFilter(config)
        self.h4_direction_filter = H4DirectionFilter()
        self.m15_entry_confirmer = M15EntryConfirmer()

        self.is_running = False
        self.is_paused = False
        self.auto_trade = False
        self.positions = []
        self.trade_history = []
        self._tracked_tickets = {}
        self._tracked_volumes = {}
        self.last_bar_times = {}
        self.new_bar_detected = False
        self.max_daily_loss = capital * 0.05
        self.max_position_size = capital * 0.95
        self.daily_pnl = 0
        self.current_day = None
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0
        self.on_signal_callback = None
        self.on_trade_callback = None
        self.on_error_callback = None
        self.paper_positions = []
        self.paper_equity = capital
        self._lock = threading.Lock()
        self._loop_count = 0
        self._last_heartbeat = datetime.now()

    def start(self, symbol: str, timeframes: List[str], auto_trade: bool = False):
        self.symbol = symbol
        self.timeframes = timeframes
        self.auto_trade = auto_trade
        self.is_running = True
        self.is_paused = False
        self._loop_count = 0
        self._last_heartbeat = datetime.now()
        self.last_bar_times = {tf: None for tf in timeframes}
        self.position_manager._log_callback = self.log

        mode_str = "OTOMATİK" if auto_trade else "MANUEL"
        if self.config.PAPER_TRADING: mode_str += " (PAPER)"

        self.log(f"\n{'='*60}")
        self.log(f"🔴 CANLI TRADİNG — {symbol} [{mode_str}]")
        self.log(f"{'='*60}")
        self.log(f"Mode: {self.config.TRADING_MODE} | TF: {timeframes}")
        self.log(f"3-Tier: H4={'ON' if self.config.USE_H4_DIRECTION_FILTER else 'OFF'} "
                 f"ML=ON M15={'ON' if self.config.USE_M15_ENTRY_CONFIRM else 'OFF'}")
        self.log(f"SL/TP ATR: {'HTF' if self.config.SL_ATR_TIMEFRAME == 'HTF' else 'Base'}")
        self.log(f"{'='*60}")

    def stop(self):
        self.is_running = False
        self.log("⏹️  Durduruldu")

    def pause(self):
        self.is_paused = True
        self.log("⏸️  Duraklatıldı")

    def resume(self):
        self.is_paused = False
        self.log("▶️  Devam ediyor")

    def check_daily_reset(self):
        today = datetime.now().date()
        if self.current_day != today:
            with self._lock:
                self.current_day = today
                self.daily_pnl = 0
                self.risk_manager.daily_trades = 0
            self.log(f"📅 Yeni gün: {today}")

    def check_risk_limits(self, floating_pnl: float = 0) -> Tuple[bool, str]:
        if self.daily_pnl + floating_pnl < -self.max_daily_loss:
            return False, f"Günlük kayıp limiti!"
        account_info = mt5.account_info()
        if account_info and account_info.equity < self.initial_capital * 0.5:
            return False, "Hesap %50 kayıp!"
        return True, "OK"

    def check_new_bar(self, data_dict: Dict[str, pd.DataFrame]) -> bool:
        new_bar = False
        for tf_name, df in data_dict.items():
            if len(df) == 0: continue
            curr = df.index[-1]
            if self.last_bar_times.get(tf_name) is None:
                self.last_bar_times[tf_name] = curr
            elif curr > self.last_bar_times[tf_name]:
                self.last_bar_times[tf_name] = curr
                new_bar = True
        return new_bar

    def fetch_current_data(self, symbol: str, timeframes: List[str]) -> Dict:
        tf_map = {'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5,
                  'M15': mt5.TIMEFRAME_M15, 'H1': mt5.TIMEFRAME_H1,
                  'H4': mt5.TIMEFRAME_H4, 'D1': mt5.TIMEFRAME_D1}
        data_dict = {}
        for tf_name in timeframes:
            tf = tf_map.get(tf_name)
            if not tf: continue
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, 500)
            if rates is None or len(rates) == 0: continue
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            df.rename(columns={'tick_volume': 'volume'}, inplace=True)
            data_dict[tf_name] = df[['open', 'high', 'low', 'close', 'volume']]
        return data_dict

    def get_current_price(self, symbol: str) -> Optional[float]:
        tick = mt5.symbol_info_tick(symbol)
        return (tick.bid + tick.ask) / 2 if tick else None

    def trading_loop_iteration(self):
        """
        3-Tier signal pipeline:
          Tier-1  H4DirectionFilter   → direction (BUY/SELL/NEUTRAL)
          Tier-2  MLModel             → entry signal + confidence
          Tier-3  M15EntryConfirmer   → pullback + candle confirmation
        """
        if not self.is_running or self.is_paused:
            return

        self.check_daily_reset()
        self._loop_count += 1

        if self._loop_count % 60 == 0:
            elapsed = (datetime.now() - self._last_heartbeat).total_seconds()
            self.log(f"💓 #{self._loop_count} | {self.risk_manager.daily_trades} trade today | {elapsed:.0f}s")
            self._last_heartbeat = datetime.now()

        try:
            # ── 1. Fetch data ──
            data_dict = self.fetch_current_data(self.symbol, self.timeframes)
            if not data_dict:
                return

            # ── 2. Bar sync (anti-repainting) ──
            if not self.check_new_bar(data_dict):
                return

            # ══════════════════════════════════════════════════════════
            # TIER-1 — H4 DIRECTION FILTER
            # ══════════════════════════════════════════════════════════
            h4_direction = 0   # default: use ML signal as-is
            h4_reason = "H4 filter OFF"

            if self.config.USE_H4_DIRECTION_FILTER and 'H4' in data_dict:
                h4_direction, h4_score, h4_reason = self.h4_direction_filter.get_direction(
                    data_dict['H4'], min_score=self.config.H4_MIN_SCORE)

                if h4_direction == 0:
                    if self._loop_count % 12 == 0:
                        self.log(f"⛔ H4 NEUTRAL: {h4_reason}")
                    return   # No trade when H4 is neutral
            elif self.config.USE_H4_DIRECTION_FILTER:
                if self._loop_count % 20 == 0:
                    self.log("⚠️  H4 data missing — direction filter skipped")

            # ══════════════════════════════════════════════════════════
            # TIER-2 — ML MODEL (entry timing)
            # ══════════════════════════════════════════════════════════
            signal, confidence = self.model.predict(data_dict)
            signal_type = "BUY" if signal == 1 else "SELL"

            # Align ML signal with H4 direction
            if self.config.USE_H4_DIRECTION_FILTER and h4_direction != 0:
                if signal != h4_direction:
                    if self._loop_count % 8 == 0:
                        self.log(f"⛔ ML{signal_type} ≠ H4{'BUY' if h4_direction==1 else 'SELL'} — skipped")
                    return

            if confidence < self.config.MIN_CONFIDENCE:
                if self._loop_count % 12 == 0:
                    self.log(f"📉 {signal_type} conf={confidence*100:.1f}% < {self.config.MIN_CONFIDENCE*100:.0f}%")
                return

            # ── Position / risk checks ──
            positions = self.get_open_positions(self.symbol)
            mt5_open_count = len(positions)
            floating_pnl = (sum(p.profit for p in positions)
                            if not self.config.PAPER_TRADING and positions else 0)
            self._detect_closed_positions(positions)

            if self.config.POSITION_MANAGEMENT and mt5_open_count > 0:
                mgmt_price = self.get_current_price(self.symbol)
                if mgmt_price:
                    self.position_manager.manage_positions(
                        open_positions=positions, current_price=mgmt_price,
                        current_signal=signal,
                        modify_sl_callback=self.modify_sl,
                        close_position_callback=self.close_position,
                        partial_close_callback=self.partial_close)

            ok_risk, reason_risk = self.check_risk_limits(floating_pnl)
            if not ok_risk:
                self.log(f"🚨 RISK: {reason_risk}"); self.stop(); return

            if mt5_open_count > 0:
                if self._loop_count % 12 == 0:
                    self.log(f"📊 {signal_type} {confidence*100:.1f}% — açık pozisyon, bekleniyor")
                return

            if self.config.SIGNAL_QUALITY_FILTER:
                feat_row = getattr(self.model, 'last_features', None)
                if feat_row is not None:
                    sq_pass, sq_score, sq_reason = self.signal_quality_filter.calculate_quality_score(
                        feat_row, signal, confidence, bar_idx=self._loop_count)
                    if not sq_pass:
                        if self._loop_count % 6 == 0:
                            self.log(f"🔍 {signal_type} — quality: {sq_reason}")
                        return

            can_trade, reason = self.risk_manager.can_trade_live(
                datetime.now(), mt5_open_count, floating_pnl)
            if not can_trade:
                self.log(f"⚠️  {signal_type} blocked: {reason}"); return

            # ── ATR & SL/TP ──
            current_price = self.get_current_price(self.symbol)
            if not current_price: return

            base_tf = self.timeframes[0]
            df_base = TechnicalIndicators.calculate_all(data_dict[base_tf].copy())
            atr = df_base['atr'].iloc[-1]

            sl_atr = atr
            if self.config.SL_ATR_TIMEFRAME == 'HTF':
                for tf in ['D1', 'H4', 'H1', 'M15', 'M5', 'M1']:
                    if tf in data_dict and tf != base_tf:
                        df_htf = TechnicalIndicators.calculate_all(data_dict[tf].copy())
                        htf_val = df_htf['atr'].iloc[-1]
                        if not np.isnan(htf_val) and htf_val > 0:
                            sl_atr = htf_val
                        break

            # ══════════════════════════════════════════════════════════
            # TIER-3 — M15 ENTRY CONFIRMATION (pullback + candle)
            # ══════════════════════════════════════════════════════════
            if self.config.USE_M15_ENTRY_CONFIRM:
                m15_df = data_dict.get('M15')
                if m15_df is not None and len(m15_df) >= 30:
                    try:
                        m15_with_ind = TechnicalIndicators.calculate_all(m15_df.copy())
                    except Exception:
                        m15_with_ind = None

                    if m15_with_ind is not None:
                        confirmed, m15_reason = self.m15_entry_confirmer.check_entry(
                            m15_with_ind, signal)
                        if not confirmed:
                            if self._loop_count % 4 == 0:
                                self.log(f"⏳ {signal_type} — {m15_reason}")
                            return
                    # If M15 data unavailable, proceed anyway

            vol_regime = self.risk_manager.detect_volatility_regime(sl_atr)
            sl, tp = self.risk_manager.calculate_adaptive_stops(
                current_price, signal, sl_atr, vol_regime)

            # ── Signal log ──
            sl_pct = abs(current_price - sl) / current_price * 100
            tp_pct = abs(tp - current_price) / current_price * 100
            self.log(f"\n{'='*55}")
            self.log(f"📊 SINYAL | {signal_type} | conf={confidence*100:.1f}%")
            self.log(f"   Tier-1 H4: {h4_reason}")
            self.log(f"   Tier-2 ML: {signal_type} {confidence*100:.1f}%")
            if self.config.USE_M15_ENTRY_CONFIRM:
                self.log(f"   Tier-3 M15: ✅ confirmed")
            self.log(f"   Price={current_price:.5f} SL={sl_pct:.2f}% TP={tp_pct:.2f}% R:R=1:{tp_pct/sl_pct:.1f}" if sl_pct > 0 else "")
            self.log(f"   Vol:{vol_regime} ATR:{atr:.5f} Float:${floating_pnl:.2f}")
            self.log(f"{'='*55}")

            if self.on_signal_callback:
                self.on_signal_callback({
                    'type': signal_type, 'symbol': self.symbol,
                    'confidence': confidence, 'price': current_price,
                    'sl': sl, 'tp': tp, 'regime': vol_regime, 'atr': atr,
                    'floating_pnl': floating_pnl, 'h4_reason': h4_reason
                })

            if self.auto_trade:
                volume = self.calculate_position_size(
                    self.symbol, atr=sl_atr, current_price=current_price)
                if volume <= 0:
                    self.log("❌ Hesaplanan lot 0!"); return

                success = self.send_order(
                    self.symbol, signal_type, volume, current_price, sl, tp,
                    f"3T-ML conf:{confidence:.2f}")

                if success:
                    with self._lock:
                        self.total_trades += 1
                        new_pos = self.get_open_positions(self.symbol)
                        self._tracked_tickets = {
                            (p.ticket if not self.config.PAPER_TRADING else p['ticket']): True
                            for p in new_pos} if new_pos else {}
                    if self.config.POSITION_MANAGEMENT and new_pos:
                        last_pos = new_pos[-1]
                        t_ticket = (last_pos.ticket if not self.config.PAPER_TRADING
                                    else last_pos['ticket'])
                        self.position_manager.register_position(
                            ticket=t_ticket, entry_price=current_price,
                            sl=sl, tp=tp, order_type=signal_type,
                            volume=volume, atr=atr)
                    self.risk_manager.update_trade_live(opened=True)
                    if self.config.SIGNAL_QUALITY_FILTER:
                        self.signal_quality_filter.on_trade_opened(self._loop_count)
                    if self.on_trade_callback:
                        self.on_trade_callback({'type': signal_type, 'volume': volume,
                                                'price': current_price, 'sl': sl, 'tp': tp})
            else:
                self.log("✋ MANUEL ONAY BEKLENİYOR...")

        except Exception as e:
            import traceback
            self.log(f"❌ Loop hatası: {e}\n{traceback.format_exc()[-300:]}")
            if self.on_error_callback: self.on_error_callback(str(e))

    # ── Order execution ──

    def calculate_position_size(self, symbol, atr=0, current_price=0):
        account_info = mt5.account_info()
        symbol_info  = mt5.symbol_info(symbol)
        if not account_info or not symbol_info:
            return 0.01
        balance = account_info.balance
        vol_min = symbol_info.volume_min; vol_max = symbol_info.volume_max
        vol_step = symbol_info.volume_step
        digits = symbol_info.digits
        pip_size = symbol_info.point * (10 if digits in [3, 5] else 1)
        tick_value = symbol_info.trade_tick_value
        tick_size  = symbol_info.trade_tick_size
        pip_value_per_lot = ((pip_size / tick_size) * tick_value if tick_size > 0
                             else symbol_info.trade_contract_size * pip_size)

        mode = self.config.LOT_MODE
        if mode == 'FIXED':
            lot = self.config.FIXED_LOT
        elif mode in ('RISK_PCT', 'ATR_BASED'):
            risk_amount = balance * self.config.LIVE_RISK_PCT
            if atr > 0 and current_price > 0:
                sl_pips = (atr * self.config.ATR_SL_MULTIPLIER) / pip_size if pip_size > 0 else 50
            else:
                sl_pips = (current_price * self.config.STOP_LOSS_PCT) / pip_size if current_price > 0 else 50
            lot = (risk_amount / (sl_pips * pip_value_per_lot)
                   if sl_pips > 0 and pip_value_per_lot > 0
                   else risk_amount / (symbol_info.trade_contract_size * 0.01))
        else:
            lot = vol_min

        max_lot = min(self.config.MAX_LOT, vol_max)
        lot = max(vol_min, min(lot, max_lot))
        if vol_step > 0:
            lot = round(round(lot / vol_step) * vol_step, 8)
        lot = max(vol_min, lot)
        self.log(f"   📐 lot={lot} (mode={mode} risk={self.config.LIVE_RISK_PCT*100:.1f}%)")
        return lot

    def send_order(self, symbol, order_type, volume, price, sl, tp, comment=""):
        if self.config.PAPER_TRADING:
            return self._send_paper_order(symbol, order_type, volume, price, sl, tp, comment)
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None: return False
        if not symbol_info.visible: mt5.symbol_select(symbol, True)
        order_type_mt5 = mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL
        tick = mt5.symbol_info_tick(symbol)
        if not tick: return False
        price = tick.ask if order_type == "BUY" else tick.bid
        digits = symbol_info.digits
        sl = round(sl, digits); tp = round(tp, digits); price = round(price, digits)
        fm = symbol_info.filling_mode
        filling_types = []
        if fm & 1: filling_types.append(mt5.ORDER_FILLING_FOK)
        if fm & 2: filling_types.append(mt5.ORDER_FILLING_IOC)
        filling_types.append(mt5.ORDER_FILLING_RETURN)
        for filling in filling_types:
            request = {"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol,
                       "volume": volume, "type": order_type_mt5, "price": price,
                       "sl": sl, "tp": tp, "deviation": 20, "magic": 234000,
                       "comment": comment, "type_time": mt5.ORDER_TIME_GTC,
                       "type_filling": filling}
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                self.log(f"✅ Order OK: #{result.order} {order_type} {volume}@{result.price}")
                return True
        self.log(f"❌ Order failed")
        return False

    def _send_paper_order(self, symbol, order_type, volume, price, sl, tp, comment):
        ticket = len(self.paper_positions) + 100000
        self.paper_positions.append({
            'ticket': ticket, 'symbol': symbol,
            'type': mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL,
            'volume': volume, 'price_open': price, 'sl': sl, 'tp': tp,
            'time': datetime.now(), 'comment': comment, 'magic': 234000})
        self._tracked_tickets[ticket] = True
        self.log(f"📄 PAPER {order_type} {volume}@{price} SL={sl} TP={tp}")
        return True

    def close_position(self, position):
        if self.config.PAPER_TRADING:
            return self._close_paper_position(position)
        symbol = position.symbol; ticket = position.ticket
        pos_type = position.type; volume = position.volume
        order_type = mt5.ORDER_TYPE_SELL if pos_type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(symbol)
        if not tick: return False
        price = tick.bid if pos_type == mt5.ORDER_TYPE_BUY else tick.ask
        si = mt5.symbol_info(symbol)
        fm = si.filling_mode if si else 0
        filling_types = []
        if fm & 1: filling_types.append(mt5.ORDER_FILLING_FOK)
        if fm & 2: filling_types.append(mt5.ORDER_FILLING_IOC)
        filling_types.append(mt5.ORDER_FILLING_RETURN)
        for filling in filling_types:
            r = mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol,
                                "volume": volume, "type": order_type, "position": ticket,
                                "price": price, "deviation": 20, "magic": 234000,
                                "comment": "Close", "type_time": mt5.ORDER_TIME_GTC,
                                "type_filling": filling})
            if r and r.retcode == mt5.TRADE_RETCODE_DONE:
                self.log(f"✅ Closed: #{ticket}"); return True
        return False

    def _close_paper_position(self, position):
        ticket = position.get('ticket') if isinstance(position, dict) else position
        self.paper_positions = [p for p in self.paper_positions if p['ticket'] != ticket]
        if ticket in self._tracked_tickets: del self._tracked_tickets[ticket]
        return True

    def modify_sl(self, ticket, new_sl):
        if self.config.PAPER_TRADING:
            for p in self.paper_positions:
                if p['ticket'] == ticket: p['sl'] = new_sl; return True
            return False
        try:
            pos = mt5.positions_get(ticket=ticket)
            if not pos: return False
            si = mt5.symbol_info(pos[0].symbol)
            new_sl = round(new_sl, si.digits if si else 5)
            r = mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "position": ticket,
                                "symbol": pos[0].symbol, "sl": new_sl, "tp": pos[0].tp, "magic": 234000})
            return bool(r and r.retcode == mt5.TRADE_RETCODE_DONE)
        except Exception: return False

    def partial_close(self, position, close_volume):
        if self.config.PAPER_TRADING:
            if isinstance(position, dict):
                position['volume'] = round(position['volume'] - close_volume, 2)
                if position['volume'] <= 0:
                    self.paper_positions = [p for p in self.paper_positions
                                            if p['ticket'] != position['ticket']]
                return True
            return False
        try:
            symbol = position.symbol; ticket = position.ticket
            tick = mt5.symbol_info_tick(symbol)
            if not tick: return False
            order_type = (mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY
                          else mt5.ORDER_TYPE_BUY)
            price = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask
            si = mt5.symbol_info(symbol)
            fm = si.filling_mode if si else 0
            filling_types = []
            if fm & 1: filling_types.append(mt5.ORDER_FILLING_FOK)
            if fm & 2: filling_types.append(mt5.ORDER_FILLING_IOC)
            filling_types.append(mt5.ORDER_FILLING_RETURN)
            for filling in filling_types:
                r = mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol,
                                    "volume": close_volume, "type": order_type, "position": ticket,
                                    "price": price, "deviation": 20, "magic": 234000,
                                    "comment": "Partial", "type_time": mt5.ORDER_TIME_GTC,
                                    "type_filling": filling})
                if r and r.retcode == mt5.TRADE_RETCODE_DONE: return True
            return False
        except Exception: return False

    def get_open_positions(self, symbol=None):
        if self.config.PAPER_TRADING:
            return self.paper_positions
        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        if not positions: return []
        return [p for p in positions if p.magic == 234000]

    def _detect_closed_positions(self, current_positions):
        if self.config.PAPER_TRADING:
            current_tickets = {p['ticket'] for p in current_positions} if current_positions else set()
        else:
            current_tickets = {p.ticket for p in current_positions} if current_positions else set()
        if not hasattr(self, '_tracked_tickets'): self._tracked_tickets = {}; return
        for ticket in set(self._tracked_tickets.keys()) - current_tickets:
            self.log(f"📌 Kapandı: #{ticket}")
            pnl = 0
            if not self.config.PAPER_TRADING:
                try:
                    from_dt = datetime.now() - timedelta(days=1)
                    deals = mt5.history_deals_get(from_dt, datetime.now() + timedelta(hours=1))
                    if deals:
                        d = [d for d in deals if d.position_id == ticket]
                        if len(d) >= 2:
                            pnl = d[-1].profit + d[-1].commission + d[-1].swap
                except Exception: pass
            with self._lock:
                self.daily_pnl += pnl; self.total_pnl += pnl
                if pnl > 0: self.winning_trades += 1
            self.risk_manager.update_trade_live(closed=True)
        if self.config.PAPER_TRADING:
            self._tracked_tickets = {p['ticket']: True for p in current_positions} if current_positions else {}
        else:
            self._tracked_tickets = {p.ticket: True for p in current_positions} if current_positions else {}

    def log(self, message):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")


# =============================================================================
# TRADING GUI  v6.5 — Compact Layout
# =============================================================================

class TradingGUI:
    def __init__(self, symbol=None, model_path=None, autostart=False):
        self.window = tk.Tk()
        self.window.configure(bg='#1e1e1e')
        self.config = Config(mode="DAY_TRADING")
        self.mt5_fetcher = MT5DataFetcher()
        self.model = MLModel(self.config)
        self.backtester = None
        self.config_manager = ConfigManager()
        self._data_lock = threading.Lock()
        self.data_dict = None
        self.backtest_results = None
        self.live_trader = None
        self.live_thread = None
        self.multi_traders = {}
        self.multi_threads = {}
        self.pending_signal = None
        self._init_symbol = symbol
        self._init_model_path = model_path
        self._init_autostart = autostart

        title = "MT5 Trading System v6.5"
        if symbol: title += f" — {symbol}"
        self.window.title(title)
        self.window.geometry("1600x850")   # ← compact (was 1800x900)

        self.create_widgets()
        if symbol: self.combo_symbol.set(symbol)
        if model_path: self.window.after(500, self._auto_load_model)

    # ─────────────────────────────────────────────────────────────────────────
    # WIDGET CREATION
    # ─────────────────────────────────────────────────────────────────────────

    def create_widgets(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TLabel', background='#1e1e1e', foreground='white', font=('Arial', 9))
        style.configure('TButton', font=('Arial', 8, 'bold'))
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'))
        style.configure('green.Horizontal.TProgressbar',
                        troughcolor='#1e1e1e', background='#4CAF50',
                        lightcolor='#66BB6A', darkcolor='#388E3C',
                        bordercolor='#2d2d2d', thickness=16)

        header = ttk.Label(self.window,
                           text="🚀 MT5 Trading System v6.5 | H4→ML→M15 (3-Tier)",
                           style='Header.TLabel')
        header.pack(pady=5)

        main_paned = tk.PanedWindow(self.window, orient=tk.HORIZONTAL,
                                     bg='#1e1e1e', sashwidth=4, sashrelief=tk.RAISED)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=3)

        # ── Left panel (scrollable, compact) ──
        left_container = tk.Frame(main_paned, bg='#2d2d2d', relief=tk.RAISED, bd=2)
        canvas = tk.Canvas(left_container, bg='#2d2d2d', highlightthickness=0, width=290)
        scrollbar = tk.Scrollbar(left_container, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas, bg='#2d2d2d')
        self.scrollable_frame.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind_all("<MouseWheel>",
            lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.create_control_panel(self.scrollable_frame)

        middle_panel = tk.Frame(main_paned, bg='#2d2d2d', relief=tk.RAISED, bd=2)
        self.create_results_panel(middle_panel)

        right_panel = tk.Frame(main_paned, bg='#2d2d2d', relief=tk.RAISED, bd=2)
        self.create_log_panel(right_panel)

        main_paned.add(left_container, minsize=280, width=300)  # ← compact
        main_paned.add(middle_panel,   minsize=500, width=700)
        main_paned.add(right_panel,    minsize=260, width=340)

    def create_control_panel(self, parent):
        pad = dict(padx=5, pady=2)

        # ── MT5 Connection ──
        conn_frame = tk.LabelFrame(parent, text="MT5 Bağlantı", bg='#2d2d2d', fg='white',
                                   font=('Arial', 9, 'bold'))
        conn_frame.pack(fill=tk.X, **pad)
        self.btn_connect = tk.Button(conn_frame, text="🔌 MT5'e Bağlan", command=self.connect_mt5,
                  bg='#4CAF50', fg='white', font=('Arial', 9, 'bold'))
        self.btn_connect.pack(pady=3, padx=5, fill=tk.X)
        self.lbl_connection = ttk.Label(conn_frame, text="Bağlı değil", foreground='red')
        self.lbl_connection.pack(pady=2)

        # ── Trading Mode ──
        mode_frame = tk.LabelFrame(parent, text="🎯 Mode", bg='#2d2d2d', fg='white',
                                   font=('Arial', 9, 'bold'))
        mode_frame.pack(fill=tk.X, **pad)
        self.trading_mode = tk.StringVar(value="DAY_TRADING")
        mode_row = tk.Frame(mode_frame, bg='#2d2d2d'); mode_row.pack(fill=tk.X, padx=5)
        for text, value in [('⚡ Scalp', 'SCALPING'), ('📊 Day', 'DAY_TRADING'), ('📈 Swing', 'SWING')]:
            tk.Radiobutton(mode_row, text=text, variable=self.trading_mode, value=value,
                           bg='#2d2d2d', fg='white', font=('Arial', 8),
                           command=self.change_trading_mode).pack(side=tk.LEFT, padx=3, pady=1)
        self.lbl_mode_desc = tk.Label(mode_frame, text="", bg='#2d2d2d', fg='#FFD700',
                                      font=('Arial', 7), justify='left')
        self.lbl_mode_desc.pack(padx=5, pady=2)
        self.update_mode_description()

        # ── Parameters ──
        pf = tk.LabelFrame(parent, text="⚙️ Parametreler", bg='#2d2d2d', fg='white',
                           font=('Arial', 9, 'bold'))
        pf.pack(fill=tk.X, **pad)

        def row(parent, label, row_n):
            ttk.Label(parent, text=label, font=('Arial', 8)).grid(
                row=row_n, column=0, sticky='w', padx=5, pady=1)

        row(pf, "Threshold (%):", 0)
        self.entry_threshold = tk.Entry(pf, width=7)
        self.entry_threshold.insert(0, str(self.config.PROFIT_THRESHOLD * 100))
        self.entry_threshold.grid(row=0, column=1, padx=3, pady=1)
        self.scale_threshold = tk.Scale(pf, from_=0.1, to=2.0, resolution=0.1,
                                         orient=tk.HORIZONTAL, length=90,   # ← 90px
                                         bg='#2d2d2d', fg='white',
                                         command=lambda v: self._sync_entry(self.entry_threshold, v))
        self.scale_threshold.set(self.config.PROFIT_THRESHOLD * 100)
        self.scale_threshold.grid(row=0, column=2, padx=3, pady=1)

        row(pf, "Forward Periods:", 1)
        self.spin_forward = tk.Spinbox(pf, from_=5, to=100, width=6, increment=5)
        self.spin_forward.delete(0, tk.END); self.spin_forward.insert(0, self.config.FORWARD_PERIODS)
        self.spin_forward.grid(row=1, column=1, padx=3, pady=1)

        row(pf, "Confidence (%):", 2)
        self.entry_confidence = tk.Entry(pf, width=7)
        self.entry_confidence.insert(0, str(self.config.MIN_CONFIDENCE * 100))
        self.entry_confidence.grid(row=2, column=1, padx=3, pady=1)
        self.scale_confidence = tk.Scale(pf, from_=50, to=90, resolution=5,
                                          orient=tk.HORIZONTAL, length=90,  # ← 90px
                                          bg='#2d2d2d', fg='white',
                                          command=lambda v: self._sync_entry(self.entry_confidence, v))
        self.scale_confidence.set(self.config.MIN_CONFIDENCE * 100)
        self.scale_confidence.grid(row=2, column=2, padx=3, pady=1)

        row(pf, "Pos. Risk (%):", 3)
        self.entry_risk = tk.Entry(pf, width=7)
        self.entry_risk.insert(0, str(self.config.POSITION_RISK_PCT * 100))
        self.entry_risk.grid(row=3, column=1, padx=3, pady=1)

        self.var_adaptive = tk.BooleanVar(value=self.config.USE_ADAPTIVE_STOPS)
        tk.Checkbutton(pf, text="Adaptive SL/TP", variable=self.var_adaptive,
                       bg='#2d2d2d', fg='white', selectcolor='#1e1e1e',
                       font=('Arial', 8)).grid(row=4, column=0, columnspan=2, sticky='w', padx=5, pady=1)
        self.var_htf_atr = tk.BooleanVar(value=True)
        tk.Checkbutton(pf, text="H4 ATR SL/TP", variable=self.var_htf_atr,
                       bg='#2d2d2d', fg='#00BCD4', selectcolor='#1e1e1e',
                       font=('Arial', 8)).grid(row=4, column=2, sticky='w', padx=3, pady=1)

        btn_row = tk.Frame(pf, bg='#2d2d2d'); btn_row.grid(row=5, column=0, columnspan=3,
                                                              sticky='ew', padx=5, pady=3)
        tk.Button(btn_row, text="✅ Uygula", command=self.apply_parameters,
                  bg='#4CAF50', fg='white', font=('Arial', 8, 'bold')).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        tk.Button(btn_row, text="🔄 Reset", command=self.reset_parameters,
                  bg='#607D8B', fg='white', font=('Arial', 8, 'bold')).pack(side=tk.LEFT, padx=2)

        # ── Symbol & Date ──
        sf = tk.LabelFrame(parent, text="Sembol & Tarih", bg='#2d2d2d', fg='white',
                           font=('Arial', 9, 'bold'))
        sf.pack(fill=tk.X, **pad)

        row_s = tk.Frame(sf, bg='#2d2d2d'); row_s.pack(fill=tk.X, padx=5, pady=1)
        tk.Label(row_s, text="Sembol:", bg='#2d2d2d', fg='white', font=('Arial', 8)).pack(side=tk.LEFT)
        self.combo_symbol = ttk.Combobox(row_s, width=10)
        self.combo_symbol.set("EURUSD"); self.combo_symbol.pack(side=tk.LEFT, padx=3)

        row_ms = tk.Frame(sf, bg='#2d2d2d'); row_ms.pack(fill=tk.X, padx=5, pady=1)
        tk.Label(row_ms, text="Multi:", bg='#2d2d2d', fg='white', font=('Arial', 8)).pack(side=tk.LEFT)
        self.entry_multi_symbols = tk.Entry(row_ms, width=22, bg='#3d3d3d', fg='#00BCD4',
                                             insertbackground='white', font=('Arial', 8))
        self.entry_multi_symbols.pack(side=tk.LEFT, padx=3)

        today = datetime.now().date()
        default_start = today - timedelta(days=60)
        for label_txt, attr_name, y, m, d in [
            ("Start:", 'date_start', default_start.year, default_start.month, default_start.day),
            ("End:",   'date_end',   today.year, today.month, today.day)]:
            row_d = tk.Frame(sf, bg='#2d2d2d'); row_d.pack(fill=tk.X, padx=5, pady=1)
            tk.Label(row_d, text=label_txt, bg='#2d2d2d', fg='white',
                     font=('Arial', 8), width=5).pack(side=tk.LEFT)
            de = DateEntry(row_d, width=12, background='darkblue', foreground='white',
                           borderwidth=2, date_pattern='yyyy-mm-dd', maxdate=today,
                           year=y, month=m, day=d)
            de.pack(side=tk.LEFT, padx=3)
            setattr(self, attr_name, de)

        cap_row = tk.Frame(sf, bg='#2d2d2d'); cap_row.pack(fill=tk.X, padx=5, pady=1)
        tk.Label(cap_row, text="Sermaye($):", bg='#2d2d2d', fg='white',
                 font=('Arial', 8)).pack(side=tk.LEFT)
        self.entry_capital = tk.Entry(cap_row, width=12)
        self.entry_capital.insert(0, "10000"); self.entry_capital.pack(side=tk.LEFT, padx=3)

        # ── Actions ──
        af = tk.LabelFrame(parent, text="İşlemler", bg='#2d2d2d', fg='white',
                           font=('Arial', 9, 'bold'))
        af.pack(fill=tk.X, **pad)
        for txt, cmd, clr in [
            ("📥 Veri Çek",   self.fetch_data,   '#2196F3'),
            ("🤖 Model Eğit", self.train_model,  '#FF9800'),
            ("📊 Backtest",   self.run_backtest,  '#9C27B0')]:
            tk.Button(af, text=txt, command=cmd, bg=clr, fg='white',
                      font=('Arial', 8, 'bold')).pack(pady=2, padx=5, fill=tk.X)

        # Model save/load/launcher — compact row
        mio = tk.Frame(af, bg='#2d2d2d'); mio.pack(fill=tk.X, padx=5, pady=1)
        for txt, cmd, clr in [
            ("💾 Kaydet",  self.save_model_dialog, '#00796B'),
            ("📂 Yükle",   self.load_model_dialog, '#00796B'),
            ("🚀 Launcher",self.create_launcher_files, '#455A64')]:
            tk.Button(mio, text=txt, command=cmd, bg=clr, fg='white',
                      font=('Arial', 8, 'bold')).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)

        tk.Button(af, text="📈 Walk-Forward", command=self.run_walk_forward,
                  bg='#FF5722', fg='white', font=('Arial', 8, 'bold')).pack(pady=2, padx=5, fill=tk.X)

        # Paper Trading
        self.var_paper_trading = tk.BooleanVar(value=False)
        tk.Checkbutton(parent, text="📄 Paper Trading", variable=self.var_paper_trading,
                       bg='#2d2d2d', fg='#4CAF50', selectcolor='#1e1e1e',
                       font=('Arial', 8, 'bold'), command=self._toggle_paper_trading).pack(
                           pady=1, padx=5, anchor='w')

        # ── 3-Tier Architecture Settings ──
        tier_frame = tk.LabelFrame(parent, text="🎯 3-Tier Architecture",
                                   bg='#2d2d2d', fg='#FFD700', font=('Arial', 9, 'bold'))
        tier_frame.pack(fill=tk.X, **pad)

        t1 = tk.Frame(tier_frame, bg='#2d2d2d'); t1.pack(fill=tk.X, padx=5, pady=1)
        self.var_h4_filter = tk.BooleanVar(value=True)
        tk.Checkbutton(t1, text="Tier-1 H4 Direction Filter",
                       variable=self.var_h4_filter, bg='#2d2d2d', fg='#FFD700',
                       selectcolor='#1e1e1e', font=('Arial', 8, 'bold'),
                       command=self._on_h4_filter_change).pack(side=tk.LEFT)
        tk.Label(t1, text="Min:", bg='#2d2d2d', fg='white', font=('Arial', 7)).pack(side=tk.LEFT, padx=(8,0))
        self.entry_h4_score = tk.Entry(t1, width=2, bg='#3d3d3d', fg='white',
                                        insertbackground='white')
        self.entry_h4_score.insert(0, "4"); self.entry_h4_score.pack(side=tk.LEFT, padx=2)
        tk.Label(t1, text="/5", bg='#2d2d2d', fg='white', font=('Arial', 7)).pack(side=tk.LEFT)

        t2_lbl = tk.Label(tier_frame, text="Tier-2 ML (entry timing) ← always ON",
                          bg='#2d2d2d', fg='#4CAF50', font=('Arial', 7))
        t2_lbl.pack(anchor='w', padx=5, pady=0)

        t3 = tk.Frame(tier_frame, bg='#2d2d2d'); t3.pack(fill=tk.X, padx=5, pady=1)
        self.var_m15_confirm = tk.BooleanVar(value=True)
        tk.Checkbutton(t3, text="Tier-3 M15 Pullback+Candle",
                       variable=self.var_m15_confirm, bg='#2d2d2d', fg='#00BCD4',
                       selectcolor='#1e1e1e', font=('Arial', 8, 'bold'),
                       command=self._on_m15_confirm_change).pack(side=tk.LEFT)

        # ── Feature / Label / Model settings ──
        fl_frame = tk.LabelFrame(parent, text="Model Ayarları", bg='#2d2d2d', fg='white',
                                 font=('Arial', 9, 'bold'))
        fl_frame.pack(fill=tk.X, **pad)

        self.var_feature_selection = tk.BooleanVar(value=True)
        tk.Checkbutton(fl_frame, text="🎯 Feature Selection (RFE)",
                       variable=self.var_feature_selection, bg='#2d2d2d', fg='white',
                       selectcolor='#1e1e1e', font=('Arial', 8),
                       command=self._toggle_feature_selection).pack(pady=0, padx=5, anchor='w')

        lm_row = tk.Frame(fl_frame, bg='#2d2d2d'); lm_row.pack(fill=tk.X, padx=5, pady=1)
        tk.Label(lm_row, text="Label:", bg='#2d2d2d', fg='white', font=('Arial', 8)).pack(side=tk.LEFT)
        self.var_label_mode = tk.StringVar(value='PERCENTILE')
        for mt, mv in [('%ile', 'PERCENTILE'), ('ATR', 'ATR'),
                       ('Fixed', 'FIXED'), ('🔄Swing', 'SWING')]:
            tk.Radiobutton(lm_row, text=mt, variable=self.var_label_mode, value=mv,
                           bg='#2d2d2d', fg='white', selectcolor='#1e1e1e',
                           font=('Arial', 7), command=self._on_label_mode_change).pack(side=tk.LEFT, padx=1)

        at_row = tk.Frame(fl_frame, bg='#2d2d2d'); at_row.pack(fill=tk.X, padx=5, pady=1)
        tk.Label(at_row, text="Pct%:", bg='#2d2d2d', fg='white', font=('Arial', 7)).pack(side=tk.LEFT)
        self.entry_label_pct = tk.Entry(at_row, width=3, bg='#3d3d3d', fg='white', insertbackground='white')
        self.entry_label_pct.insert(0, "25"); self.entry_label_pct.pack(side=tk.LEFT, padx=2)
        tk.Label(at_row, text="ATR×:", bg='#2d2d2d', fg='white', font=('Arial', 7)).pack(side=tk.LEFT)
        self.entry_atr_mult = tk.Entry(at_row, width=4, bg='#3d3d3d', fg='white', insertbackground='white')
        self.entry_atr_mult.insert(0, "1.0"); self.entry_atr_mult.pack(side=tk.LEFT, padx=2)

        sw_row = tk.Frame(fl_frame, bg='#2d2d2d'); sw_row.pack(fill=tk.X, padx=5, pady=1)
        for lbl, ent_attr, def_val in [('LB:', 'entry_swing_lb', '5'),
                                        ('ATR:', 'entry_swing_atr', '1.0'),
                                        ('Win:', 'entry_swing_win', '3')]:
            tk.Label(sw_row, text=lbl, bg='#2d2d2d', fg='#FFD700', font=('Arial', 7)).pack(side=tk.LEFT)
            ent = tk.Entry(sw_row, width=3, bg='#3d3d3d', fg='white', insertbackground='white')
            ent.insert(0, def_val); ent.pack(side=tk.LEFT, padx=1)
            setattr(self, ent_attr, ent)

        mp_row = tk.Frame(fl_frame, bg='#2d2d2d'); mp_row.pack(fill=tk.X, padx=5, pady=1)
        tk.Label(mp_row, text="🧠:", bg='#2d2d2d', fg='white', font=('Arial', 8)).pack(side=tk.LEFT)
        self.var_model_preset = tk.StringVar(value='CONSERVATIVE')
        for pt, pv, pc in [('Agresif', 'AGGRESSIVE', '#f44336'),
                            ('Dengeli', 'BALANCED', '#FF9800'),
                            ('Güvenli', 'CONSERVATIVE', '#4CAF50')]:
            tk.Radiobutton(mp_row, text=pt, variable=self.var_model_preset, value=pv,
                           bg='#2d2d2d', fg=pc, selectcolor='#1e1e1e',
                           font=('Arial', 7, 'bold'), command=self._on_model_preset_change).pack(side=tk.LEFT, padx=2)

        wf_row = tk.Frame(fl_frame, bg='#2d2d2d'); wf_row.pack(fill=tk.X, padx=5, pady=1)
        tk.Label(wf_row, text="WF(gün):", bg='#2d2d2d', fg='white', font=('Arial', 7)).pack(side=tk.LEFT)
        self.entry_wf_window = tk.Entry(wf_row, width=4, bg='#3d3d3d', fg='white', insertbackground='white')
        self.entry_wf_window.insert(0, "60"); self.entry_wf_window.pack(side=tk.LEFT, padx=2)
        tk.Label(wf_row, text="Step:", bg='#2d2d2d', fg='white', font=('Arial', 7)).pack(side=tk.LEFT)
        self.entry_wf_step = tk.Entry(wf_row, width=4, bg='#3d3d3d', fg='white', insertbackground='white')
        self.entry_wf_step.insert(0, "20"); self.entry_wf_step.pack(side=tk.LEFT, padx=2)

        # ── Position Management ──
        pm_frame = tk.LabelFrame(parent, text="📋 Pozisyon Yönetimi",
                                 bg='#2d2d2d', fg='#FFD700', font=('Arial', 9, 'bold'))
        pm_frame.pack(fill=tk.X, **pad)

        pm1 = tk.Frame(pm_frame, bg='#2d2d2d'); pm1.pack(fill=tk.X, padx=5, pady=1)
        self.var_pos_mgmt = tk.BooleanVar(value=True)
        tk.Checkbutton(pm1, text="Aktif", variable=self.var_pos_mgmt, bg='#2d2d2d', fg='#4CAF50',
                       selectcolor='#1e1e1e', font=('Arial', 8, 'bold'),
                       command=self._on_pos_mgmt_change).pack(side=tk.LEFT)
        for txt, var_name in [('Trail', 'var_trailing'), ('BE', 'var_breakeven'),
                               ('Partial', 'var_partial'), ('TersKapat', 'var_reverse_close')]:
            v = tk.BooleanVar(value=True)
            setattr(self, var_name, v)
            tk.Checkbutton(pm1, text=txt, variable=v, bg='#2d2d2d', fg='white',
                           selectcolor='#1e1e1e', font=('Arial', 7)).pack(side=tk.LEFT, padx=2)

        pm2 = tk.Frame(pm_frame, bg='#2d2d2d'); pm2.pack(fill=tk.X, padx=5, pady=1)
        for lbl, ent_attr, def_val in [('BE R:R:', 'entry_be_rr', '0.5'),
                                        ('Trail×:', 'entry_trail_mult', '2.0'),
                                        ('Part%:', 'entry_partial_pct', '50')]:
            tk.Label(pm2, text=lbl, bg='#2d2d2d', fg='white', font=('Arial', 7)).pack(side=tk.LEFT)
            ent = tk.Entry(pm2, width=4, bg='#3d3d3d', fg='white', insertbackground='white')
            ent.insert(0, def_val); ent.pack(side=tk.LEFT, padx=1)
            setattr(self, ent_attr, ent)

        # ── Signal Quality Filter ──
        sq_frame = tk.LabelFrame(parent, text="🔍 Sinyal Kalite Filtresi",
                                 bg='#2d2d2d', fg='#00BCD4', font=('Arial', 9, 'bold'))
        sq_frame.pack(fill=tk.X, **pad)

        sq1 = tk.Frame(sq_frame, bg='#2d2d2d'); sq1.pack(fill=tk.X, padx=5, pady=1)
        self.var_sq_filter = tk.BooleanVar(value=False)
        tk.Checkbutton(sq1, text="Rafine Mod", variable=self.var_sq_filter,
                       bg='#2d2d2d', fg='#00BCD4', selectcolor='#1e1e1e',
                       font=('Arial', 8, 'bold'), command=self._on_sq_filter_change).pack(side=tk.LEFT)
        for txt, var_name in [('H4+H1', 'var_sq_h4h1'), ('EMA', 'var_sq_ema_trend'),
                               ('Session', 'var_sq_session')]:
            v = tk.BooleanVar(value=True); setattr(self, var_name, v)
            tk.Checkbutton(sq1, text=txt, variable=v, bg='#2d2d2d', fg='white',
                           selectcolor='#1e1e1e', font=('Arial', 7)).pack(side=tk.LEFT, padx=2)

        sq2 = tk.Frame(sq_frame, bg='#2d2d2d'); sq2.pack(fill=tk.X, padx=5, pady=1)
        for lbl, ent_attr, def_val in [('Skor:', 'entry_sq_score', '4'),
                                        ('Conf:', 'entry_sq_conf', '0.72'),
                                        ('Cool:', 'entry_sq_cooldown', '60')]:
            tk.Label(sq2, text=lbl, bg='#2d2d2d', fg='white', font=('Arial', 7)).pack(side=tk.LEFT)
            ent = tk.Entry(sq2, width=4, bg='#3d3d3d', fg='white', insertbackground='white')
            ent.insert(0, def_val); ent.pack(side=tk.LEFT, padx=1)
            setattr(self, ent_attr, ent)

        # ── Progress ──
        prog_frame = tk.Frame(parent, bg='#2d2d2d'); prog_frame.pack(fill=tk.X, padx=5, pady=3)
        self.progress = ttk.Progressbar(prog_frame, mode='determinate', maximum=100,
                                         style='green.Horizontal.TProgressbar')
        self.progress.pack(fill=tk.X, side=tk.LEFT, expand=True)
        self.lbl_progress = tk.Label(prog_frame, text="0%", bg='#2d2d2d', fg='#FFD700',
                                      font=('Arial', 8, 'bold'), width=5)
        self.lbl_progress.pack(side=tk.RIGHT, padx=2)
        self.lbl_progress_detail = tk.Label(parent, text="", bg='#2d2d2d', fg='#aaa',
                                             font=('Arial', 7), anchor='w')
        self.lbl_progress_detail.pack(fill=tk.X, padx=5)

        # ── Config Management (2×2 grid) ──
        cfg_frame = tk.LabelFrame(parent, text="Konfigürasyon", bg='#2d2d2d', fg='white',
                                  font=('Arial', 9, 'bold'))
        cfg_frame.pack(fill=tk.X, **pad)
        cfg_r0 = tk.Frame(cfg_frame, bg='#2d2d2d'); cfg_r0.pack(fill=tk.X, padx=5, pady=1)
        cfg_r1 = tk.Frame(cfg_frame, bg='#2d2d2d'); cfg_r1.pack(fill=tk.X, padx=5, pady=1)
        for frm, btns in [
            (cfg_r0, [('💾 Kaydet', self.save_current_config, '#2196F3'),
                      ('📂 Yükle',  self.load_saved_config,   '#FF9800')]),
            (cfg_r1, [('⚡ Optimize', self.auto_optimize,       '#9C27B0'),
                      ('📋 Listele', self.show_saved_configs,   '#607D8B')])]:
            for txt, cmd, clr in btns:
                tk.Button(frm, text=txt, command=cmd, bg=clr, fg='white',
                          font=('Arial', 8)).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)


    def create_results_panel(self, parent):
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
        self.tab_summary = tk.Frame(self.notebook, bg='#2d2d2d')
        self.notebook.add(self.tab_summary, text='📊 Özet')
        self.create_summary_tab()
        self.tab_equity = tk.Frame(self.notebook, bg='#2d2d2d')
        self.notebook.add(self.tab_equity, text='📈 Equity')
        self.tab_trades = tk.Frame(self.notebook, bg='#2d2d2d')
        self.notebook.add(self.tab_trades, text='💰 Trades')
        self.tab_analysis = tk.Frame(self.notebook, bg='#2d2d2d')
        self.notebook.add(self.tab_analysis, text='📉 Analiz')
        self.tab_indicators = tk.Frame(self.notebook, bg='#2d2d2d')
        self.notebook.add(self.tab_indicators, text='🎯 İndikatör')
        self.tab_live = tk.Frame(self.notebook, bg='#2d2d2d')
        self.notebook.add(self.tab_live, text='🔴 CANLI')
        self.create_live_trading_tab()

    def create_log_panel(self, parent):
        log_frame = tk.LabelFrame(parent, text="📋 Log", bg='#2d2d2d', fg='white',
                                  font=('Arial', 10, 'bold'))
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.text_log = tk.Text(log_frame, bg='#1e1e1e', fg='#00ff00',
                                font=('Courier', 8), wrap=tk.WORD)
        self.text_log.pack(fill=tk.BOTH, expand=True, padx=3, pady=3, side=tk.LEFT)
        sb = tk.Scrollbar(log_frame, command=self.text_log.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_log.config(yscrollcommand=sb.set)
        tk.Button(parent, text="🗑️ Temizle", command=self.clear_log,
                  bg='#607D8B', fg='white', font=('Arial', 8)).pack(
                      pady=3, padx=5, fill=tk.X)

    def create_summary_tab(self):
        cards_frame = tk.Frame(self.tab_summary, bg='#2d2d2d')
        cards_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.metric_labels = {}
        metrics_config = [
            ('💰 Getiri', 'total_return', '%'), ('📊 Trades', 'total_trades', ''),
            ('✅ Kazanan', 'winning_trades', ''), ('❌ Kaybeden', 'losing_trades', ''),
            ('🎯 Win Rate', 'win_rate', '%'), ('💎 PF', 'profit_factor', ''),
            ('📉 Max DD', 'max_drawdown', '%'), ('📅 Günlük DD', 'daily_max_dd', '%'),
            ('⚡ Sharpe', 'sharpe_ratio', ''),
        ]
        row, col = 0, 0
        for title, key, unit in metrics_config:
            card = tk.Frame(cards_frame, bg='#3d3d3d', relief=tk.RAISED, bd=2)
            card.grid(row=row, column=col, padx=8, pady=8, sticky='nsew')
            ttk.Label(card, text=title, font=('Arial', 9)).pack(pady=3)
            lbl = tk.Label(card, text="--", font=('Arial', 16, 'bold'),
                           bg='#3d3d3d', fg='#4CAF50')
            lbl.pack(pady=8)
            self.metric_labels[key] = (lbl, unit)
            col += 1
            if col > 3: col = 0; row += 1
        for i in range(3): cards_frame.grid_rowconfigure(i, weight=1)
        for i in range(4): cards_frame.grid_columnconfigure(i, weight=1)

    def create_live_trading_tab(self):
        main_container = tk.Frame(self.tab_live, bg='#2d2d2d')
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Control
        ctrl = tk.LabelFrame(main_container, text="🎮 Kontrol", bg='#2d2d2d', fg='white',
                              font=('Arial', 10, 'bold'))
        ctrl.pack(fill=tk.X, pady=5)

        stat_row = tk.Frame(ctrl, bg='#2d2d2d'); stat_row.pack(fill=tk.X, padx=5, pady=3)
        tk.Label(stat_row, text="Durum:", bg='#2d2d2d', fg='white',
                 font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=3)
        self.lbl_live_status = tk.Label(stat_row, text="● Durdu", bg='#2d2d2d',
                                         fg='#f44336', font=('Arial', 11, 'bold'))
        self.lbl_live_status.pack(side=tk.LEFT, padx=8)

        # Buttons (compact — no height=2)
        btn_row = tk.Frame(ctrl, bg='#2d2d2d'); btn_row.pack(fill=tk.X, padx=5, pady=3)
        self.btn_live_start = tk.Button(btn_row, text="▶️ BAŞLAT",
            command=self.start_live_trading, bg='#4CAF50', fg='white',
            font=('Arial', 9, 'bold'), width=12)
        self.btn_live_start.pack(side=tk.LEFT, padx=3)
        self.btn_live_pause = tk.Button(btn_row, text="⏸️ DURAKLAT",
            command=self.pause_live_trading, bg='#FF9800', fg='white',
            font=('Arial', 9, 'bold'), width=12, state=tk.DISABLED)
        self.btn_live_pause.pack(side=tk.LEFT, padx=3)
        self.btn_live_stop = tk.Button(btn_row, text="⏹️ DURDUR",
            command=self.stop_live_trading, bg='#f44336', fg='white',
            font=('Arial', 9, 'bold'), width=12, state=tk.DISABLED)
        self.btn_live_stop.pack(side=tk.LEFT, padx=3)
        self.btn_emergency = tk.Button(btn_row, text="🚨 ACİL DUR",
            command=self.emergency_stop, bg='#b71c1c', fg='white',
            font=('Arial', 9, 'bold'), width=12)
        self.btn_emergency.pack(side=tk.LEFT, padx=3)

        mode_row = tk.Frame(ctrl, bg='#2d2d2d'); mode_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(mode_row, text="Mod:", bg='#2d2d2d', fg='white',
                 font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=3)
        self.var_auto_trade = tk.BooleanVar(value=False)
        tk.Radiobutton(mode_row, text="✋ Manuel", variable=self.var_auto_trade, value=False,
                       bg='#2d2d2d', fg='white', selectcolor='#FF9800',
                       font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(mode_row, text="🤖 Otomatik", variable=self.var_auto_trade, value=True,
                       bg='#2d2d2d', fg='white', selectcolor='#4CAF50',
                       font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        tk.Label(mode_row, text="⚠️  Otomatik = Gerçek para!",
                 bg='#2d2d2d', fg='#FF5722', font=('Arial', 8, 'italic')).pack(side=tk.LEFT, padx=10)

        # Lot size
        lot_frame = tk.LabelFrame(ctrl, text="📐 Pozisyon Büyüklüğü",
                                  bg='#2d2d2d', fg='#FFD700', font=('Arial', 9, 'bold'))
        lot_frame.pack(fill=tk.X, padx=5, pady=3)
        lr1 = tk.Frame(lot_frame, bg='#2d2d2d'); lr1.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(lr1, text="Mod:", bg='#2d2d2d', fg='white', font=('Arial', 8, 'bold')).pack(side=tk.LEFT)
        self.var_lot_mode = tk.StringVar(value=self.config.LOT_MODE)
        for txt, val, clr in [('Sabit', 'FIXED', '#FF9800'), ('Risk%', 'RISK_PCT', '#4CAF50'),
                               ('ATR', 'ATR_BASED', '#2196F3')]:
            tk.Radiobutton(lr1, text=txt, variable=self.var_lot_mode, value=val,
                           bg='#2d2d2d', fg='white', selectcolor=clr,
                           font=('Arial', 8), command=self._on_lot_mode_change).pack(side=tk.LEFT, padx=3)
        self.lbl_lot_mode_desc = tk.Label(lr1, text="", bg='#2d2d2d', fg='#aaa', font=('Arial', 7))
        self.lbl_lot_mode_desc.pack(side=tk.LEFT, padx=5)

        lr2 = tk.Frame(lot_frame, bg='#2d2d2d'); lr2.pack(fill=tk.X, padx=5, pady=2)
        self.frame_fixed_lot = tk.Frame(lr2, bg='#2d2d2d')
        tk.Label(self.frame_fixed_lot, text="Lot:", bg='#2d2d2d', fg='white',
                 font=('Arial', 8)).pack(side=tk.LEFT)
        self.entry_fixed_lot = tk.Entry(self.frame_fixed_lot, width=7, bg='#3d3d3d', fg='white',
                                         insertbackground='white', justify='center')
        self.entry_fixed_lot.insert(0, str(self.config.FIXED_LOT)); self.entry_fixed_lot.pack(side=tk.LEFT, padx=2)

        self.frame_risk_pct = tk.Frame(lr2, bg='#2d2d2d')
        tk.Label(self.frame_risk_pct, text="Risk%:", bg='#2d2d2d', fg='white',
                 font=('Arial', 8)).pack(side=tk.LEFT)
        self.entry_live_risk = tk.Entry(self.frame_risk_pct, width=5, bg='#3d3d3d', fg='white',
                                         insertbackground='white', justify='center')
        self.entry_live_risk.insert(0, str(self.config.LIVE_RISK_PCT * 100)); self.entry_live_risk.pack(side=tk.LEFT, padx=2)

        self.frame_atr_mult = tk.Frame(lr2, bg='#2d2d2d')
        tk.Label(self.frame_atr_mult, text="ATR×:", bg='#2d2d2d', fg='white',
                 font=('Arial', 8)).pack(side=tk.LEFT)
        self.entry_atr_sl_mult = tk.Entry(self.frame_atr_mult, width=4, bg='#3d3d3d', fg='white',
                                           insertbackground='white', justify='center')
        self.entry_atr_sl_mult.insert(0, str(self.config.ATR_SL_MULTIPLIER)); self.entry_atr_sl_mult.pack(side=tk.LEFT, padx=2)

        lr3 = tk.Frame(lot_frame, bg='#2d2d2d'); lr3.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(lr3, text="Max Lot:", bg='#2d2d2d', fg='white', font=('Arial', 8)).pack(side=tk.LEFT)
        self.entry_max_lot = tk.Entry(lr3, width=6, bg='#3d3d3d', fg='white',
                                       insertbackground='white', justify='center')
        self.entry_max_lot.insert(0, str(self.config.MAX_LOT)); self.entry_max_lot.pack(side=tk.LEFT, padx=2)
        tk.Button(lr3, text="🔍 Önizle", command=self._preview_lot_calculation,
                  bg='#555', fg='white', font=('Arial', 7, 'bold')).pack(side=tk.RIGHT, padx=3)

        self._on_lot_mode_change()

        # Manual approval
        self.approval_frame = tk.LabelFrame(main_container, text="✋ Manuel Onay",
                                             bg='#2d2d2d', fg='white', font=('Arial', 10, 'bold'))
        self.lbl_signal_info = tk.Label(self.approval_frame, text="Sinyal bekleniyor...",
                                         bg='#2d2d2d', fg='#FFD700', font=('Courier', 9), justify='left')
        self.lbl_signal_info.pack(padx=10, pady=10)
        abf = tk.Frame(self.approval_frame, bg='#2d2d2d'); abf.pack(pady=5)
        self.btn_approve = tk.Button(abf, text="✅ ONAYLA", command=self.approve_signal,
                                      bg='#4CAF50', fg='white', font=('Arial', 10, 'bold'),
                                      width=16, state=tk.DISABLED)
        self.btn_approve.pack(side=tk.LEFT, padx=8)
        self.btn_reject = tk.Button(abf, text="❌ REDDET", command=self.reject_signal,
                                     bg='#f44336', fg='white', font=('Arial', 10, 'bold'),
                                     width=16, state=tk.DISABLED)
        self.btn_reject.pack(side=tk.LEFT, padx=8)

        # Bottom — positions & stats
        bottom = tk.Frame(main_container, bg='#2d2d2d'); bottom.pack(fill=tk.BOTH, expand=True, pady=5)
        pf = tk.LabelFrame(bottom, text="📊 Açık Pozisyonlar", bg='#2d2d2d', fg='white',
                           font=('Arial', 9, 'bold'))
        pf.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)
        self.text_positions = tk.Text(pf, height=12, bg='#1e1e1e', fg='#00ff00', font=('Courier', 8))
        self.text_positions.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
        sf = tk.LabelFrame(bottom, text="📈 Canlı İstatistikler", bg='#2d2d2d', fg='white',
                           font=('Arial', 9, 'bold'))
        sf.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=3)
        self.text_live_stats = tk.Text(sf, height=12, bg='#1e1e1e', fg='#FFD700', font=('Courier', 8))
        self.text_live_stats.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)


    # ─────────────────────────────────────────────────────────────────
    # Utility / sync helpers
    # ─────────────────────────────────────────────────────────────────
    def _sync_entry(self, var, entry, fmt="{:.4f}"):
        try:
            val = float(var.get())
            entry.delete(0, tk.END)
            entry.insert(0, fmt.format(val))
        except Exception:
            pass

    def set_progress(self, val: int, detail: str = ""):
        self.progress['value'] = val
        if detail:
            self.lbl_progress.config(text=detail)
        self.window.update_idletasks()

    def reset_progress(self):
        self.progress['value'] = 0
        self.lbl_progress.config(text="")

    # ─────────────────────────────────────────────────────────────────
    # Toggle callbacks
    # ─────────────────────────────────────────────────────────────────
    def _toggle_paper_trading(self):
        self.config.PAPER_TRADING = bool(self.var_paper.get())
        mode = "PAPER" if self.config.PAPER_TRADING else "LIVE"
        self.log(f"Trading mode set to: {mode}")

    def _toggle_feature_selection(self):
        self.config.USE_FEATURE_SELECTION = bool(self.var_feature_sel.get())

    def _on_label_mode_change(self, *args):
        mode = self.var_label_mode.get()
        self.config.LABEL_MODE = mode

    def _on_model_preset_change(self, *args):
        preset = self.var_model_preset.get()
        presets = {
            "Balanced":    dict(N_ESTIMATORS=200, MAX_DEPTH=6, MIN_SAMPLES_LEAF=30, CONFIDENCE_THRESHOLD=0.58),
            "Conservative":dict(N_ESTIMATORS=300, MAX_DEPTH=5, MIN_SAMPLES_LEAF=50, CONFIDENCE_THRESHOLD=0.62),
            "Aggressive":  dict(N_ESTIMATORS=100, MAX_DEPTH=8, MIN_SAMPLES_LEAF=15, CONFIDENCE_THRESHOLD=0.54),
            "Custom":      None,
        }
        p = presets.get(preset)
        if p:
            for k, v in p.items():
                setattr(self.config, k, v)
            self.load_mode_parameters()

    def _on_pos_mgmt_change(self, *args):
        mode = self.var_pos_mgmt.get()
        self.config.POSITION_MANAGEMENT_MODE = mode

    def _on_sq_filter_change(self):
        self.config.USE_SIGNAL_QUALITY_FILTER = bool(self.var_sq_filter.get())
        self._sync_sq_settings()

    def _sync_sq_settings(self):
        state = tk.NORMAL if self.config.USE_SIGNAL_QUALITY_FILTER else tk.DISABLED
        for w in getattr(self, '_sq_widgets', []):
            try:
                w.config(state=state)
            except Exception:
                pass

    def _on_h4_filter_change(self):
        self.config.USE_H4_DIRECTION_FILTER = bool(self.var_h4_filter.get())
        self.log(f"H4 Direction Filter: {'ON' if self.config.USE_H4_DIRECTION_FILTER else 'OFF'}")

    def _on_m15_confirm_change(self):
        self.config.USE_M15_ENTRY_CONFIRM = bool(self.var_m15_confirm.get())
        self.log(f"M15 Entry Confirm: {'ON' if self.config.USE_M15_ENTRY_CONFIRM else 'OFF'}")

    # ─────────────────────────────────────────────────────────────────
    # Lot mode
    # ─────────────────────────────────────────────────────────────────
    def _on_lot_mode_change(self, *args):
        mode = getattr(self, 'var_lot_mode', None)
        if mode is None:
            return
        m = mode.get()
        for attr, frame in [('frame_fixed_lot', None), ('frame_risk_pct', None), ('frame_atr_mult', None)]:
            w = getattr(self, attr, None)
            if w:
                w.pack_forget()
        if m == "Fixed":
            self.frame_fixed_lot.pack(fill=tk.X, padx=5, pady=1)
        elif m == "Risk%":
            self.frame_risk_pct.pack(fill=tk.X, padx=5, pady=1)
        elif m == "ATR":
            self.frame_atr_mult.pack(fill=tk.X, padx=5, pady=1)

    def _apply_lot_settings_to_config(self):
        try:
            mode = self.var_lot_mode.get()
            if mode == "Fixed":
                self.config.LOT_SIZE = float(self.entry_fixed_lot.get())
                self.config.LOT_MODE = "fixed"
            elif mode == "Risk%":
                self.config.LIVE_RISK_PCT = float(self.entry_live_risk.get()) / 100.0
                self.config.LOT_MODE = "risk_pct"
            elif mode == "ATR":
                self.config.ATR_SL_MULTIPLIER = float(self.entry_atr_sl_mult.get())
                self.config.LOT_MODE = "atr"
            self.config.MAX_LOT = float(self.entry_max_lot.get())
        except Exception as e:
            self.log(f"Lot settings error: {e}")

    def _preview_lot_calculation(self):
        try:
            self._apply_lot_settings_to_config()
            sym = self.config.SYMBOL
            price = 1.1
            atr = 0.001
            if self.live_traders:
                lt = list(self.live_traders.values())[0]
                p = lt.get_current_price(sym)
                if p:
                    price = p
            lot = self.live_traders[sym].calculate_position_size(sym, atr, price) if sym in self.live_traders else self.config.LOT_SIZE
            self.log(f"Preview lot for {sym} @ {price:.5f}: {lot:.2f}")
        except Exception as e:
            self.log(f"Preview error: {e}")

    # ─────────────────────────────────────────────────────────────────
    # Log helpers
    # ─────────────────────────────────────────────────────────────────
    def log(self, message: str):
        try:
            import datetime
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            full = f"[{ts}] {message}\n"
            self.text_log.config(state=tk.NORMAL)
            self.text_log.insert(tk.END, full)
            self.text_log.see(tk.END)
            self.text_log.config(state=tk.DISABLED)
        except Exception:
            print(message)

    def clear_log(self):
        self.text_log.config(state=tk.NORMAL)
        self.text_log.delete(1.0, tk.END)
        self.text_log.config(state=tk.DISABLED)

    def update_mode_description(self):
        mode = self.var_label_mode.get() if hasattr(self, 'var_label_mode') else self.config.LABEL_MODE
        descs = {
            "triple_barrier": "Triple Barrier: Sabit TP/SL oranına göre etiketler.",
            "trend_following": "Trend Following: EMA crossover + ATR filtresi.",
            "mean_reversion": "Mean Reversion: BB + RSI aşırı bölge.",
            "adaptive": "Adaptive: Volatiliteye göre dinamik.",
            "swing": "Swing: Yerel min/max pivot noktaları.",
        }
        desc = descs.get(mode, "")
        if hasattr(self, 'lbl_mode_desc'):
            self.lbl_mode_desc.config(text=desc)

    def load_mode_parameters(self):
        if hasattr(self, 'slider_n_est'):
            self.slider_n_est.set(self.config.N_ESTIMATORS)
        if hasattr(self, 'slider_depth'):
            self.slider_depth.set(self.config.MAX_DEPTH)
        if hasattr(self, 'slider_leaf'):
            self.slider_leaf.set(self.config.MIN_SAMPLES_LEAF)
        if hasattr(self, 'slider_conf'):
            self.slider_conf.set(self.config.CONFIDENCE_THRESHOLD)
        if hasattr(self, 'slider_tp'):
            self.slider_tp.set(self.config.TP_MULTIPLIER)
        if hasattr(self, 'slider_sl'):
            self.slider_sl.set(self.config.SL_MULTIPLIER)


    # ─────────────────────────────────────────────────────────────────
    # MT5 Connect
    # ─────────────────────────────────────────────────────────────────
    def connect_mt5(self):
        try:
            import MetaTrader5 as mt5
            if not mt5.initialize():
                self.log(f"MT5 init failed: {mt5.last_error()}")
                return
            info = mt5.terminal_info()
            self.log(f"MT5 connected: {info.name} build {info.build}")
            self.btn_connect.config(bg='#4CAF50', text="✅ Bağlı")
        except ImportError:
            self.log("MetaTrader5 package not installed — demo mode.")
            self.btn_connect.config(bg='#FF9800', text="⚠️ Demo")
        except Exception as e:
            self.log(f"MT5 error: {e}")

    def change_trading_mode(self):
        new_mode = self.trading_mode.get()
        self.config.set_mode(new_mode)
        self.log(f"Mode: {new_mode}")
        self.update_mode_description()
        self.load_mode_parameters()

    def apply_parameters(self):
        try:
            self.config.PROFIT_THRESHOLD = float(self.scale_threshold.get()) / 100
            self.config.MIN_CONFIDENCE   = float(self.scale_confidence.get()) / 100
            self.config.FORWARD_PERIODS  = int(self.spin_forward.get())
            self.config.POSITION_RISK_PCT = float(self.entry_risk.get()) / 100
            self.config.USE_ADAPTIVE_STOPS = bool(self.var_adaptive.get())
            self._apply_lot_settings_to_config()
            self.log("Parameters applied.")
        except Exception as e:
            self.log(f"Apply params error: {e}")

    def reset_parameters(self):
        fresh = Config()
        for attr in ["N_ESTIMATORS","MAX_DEPTH","MIN_SAMPLES_LEAF","CONFIDENCE_THRESHOLD",
                     "TP_MULTIPLIER","SL_MULTIPLIER","LOT_SIZE","LIVE_RISK_PCT","MAX_LOT"]:
            setattr(self.config, attr, getattr(fresh, attr))
        self.load_mode_parameters()
        self.log("Parameters reset to defaults.")

    # ─────────────────────────────────────────────────────────────────
    # Data, Training, Backtest
    # ─────────────────────────────────────────────────────────────────
    def fetch_data(self):
        def _task():
            try:
                self.log("Fetching data...")
                self.set_progress(10, "Connecting...")
                fetcher = MT5DataFetcher(self.config)
                sym = self.config.SYMBOL
                tfs = self.config.TIMEFRAMES
                data = fetcher.fetch_multi_timeframe(sym, tfs, self.config.LOOKBACK_BARS)
                if not data:
                    self.log("No data returned.")
                    self.reset_progress()
                    return
                self.data_dict = data
                bars = {tf: len(df) for tf, df in data.items()}
                self.log(f"Data fetched: {bars}")
                self.set_progress(100, "Done")
                self.window.after(2000, self.reset_progress)
            except Exception as e:
                self.log(f"Fetch error: {e}")
                self.reset_progress()
        import threading
        threading.Thread(target=_task, daemon=True).start()

    def train_model(self):
        def _task():
            try:
                self.apply_parameters()
                if not hasattr(self, 'data_dict') or not self.data_dict:
                    self.log("No data — fetching first...")
                    fetcher = MT5DataFetcher()
                    self.data_dict = fetcher.fetch_multi_timeframe(
                        self.config.SYMBOL, self.config.TIMEFRAMES, self.config.LOOKBACK_BARS)
                self.log("Training model...")
                self.set_progress(5, "Preparing features...")
                self.model = MLModel(self.config)
                ind = TechnicalIndicators(self.config)

                def prog_cb(v, d=""):
                    self.set_progress(v, d)

                res = self.model.train(self.data_dict, progress_callback=prog_cb)
                self.set_progress(100, "Training complete")
                self.log(f"Train result: {res}")
                self.window.after(0, lambda: self._show_train_results(res))
                self.window.after(3000, self.reset_progress)
            except Exception as e:
                import traceback
                self.log(f"Train error: {e}\n{traceback.format_exc()}")
                self.reset_progress()
        import threading
        threading.Thread(target=_task, daemon=True).start()

    def _show_train_results(self, res: dict):
        try:
            lines = ["=== Training Results ==="]
            for k, v in res.items():
                if isinstance(v, float):
                    lines.append(f"  {k}: {v:.4f}")
                elif isinstance(v, list):
                    lines.append(f"  {k}: {v[:5]}...")
                else:
                    lines.append(f"  {k}: {v}")
            self.text_results.config(state=tk.NORMAL)
            self.text_results.delete(1.0, tk.END)
            self.text_results.insert(tk.END, "\n".join(lines))
            self.text_results.config(state=tk.DISABLED)
        except Exception:
            pass

    def run_backtest(self):
        def _task():
            try:
                if not hasattr(self, 'data_dict') or not self.data_dict:
                    self.log("No data for backtest.")
                    return
                if self.model is None:
                    self.log("No trained model.")
                    return
                self.log("Running backtest...")
                self.set_progress(10, "Backtesting...")
                bt = Backtester(self.config, self.model, capital=self.config.INITIAL_CAPITAL)
                res = bt.run(self.data_dict,
                             progress_callback=lambda v, d="": self.set_progress(v, d))
                self.backtest_results = res
                self.log(f"Backtest done: {res.get('total_trades',0)} trades, "
                         f"PF={res.get('profit_factor',0):.2f}, "
                         f"WR={res.get('win_rate',0):.1%}")
                self.set_progress(100, "Complete")
                self.window.after(0, self._update_results)
                self.window.after(3000, self.reset_progress)
            except Exception as e:
                import traceback
                self.log(f"Backtest error: {e}\n{traceback.format_exc()}")
                self.reset_progress()
        import threading
        threading.Thread(target=_task, daemon=True).start()

    def run_walk_forward(self):
        def _task():
            try:
                if not hasattr(self, 'data_dict') or not self.data_dict:
                    self.log("No data for walk-forward.")
                    return
                self.log("Running walk-forward analysis...")
                self.set_progress(5, "Walk-Forward...")
                wfa = WalkForwardAnalyzer(self.config)
                res = wfa.run(self.data_dict,
                              progress_callback=lambda v, d="": self.set_progress(v, d))
                self.wf_results = res
                self.log(f"Walk-Forward done: {len(res.get('windows',[]))} windows")
                self.set_progress(100, "Complete")
                self.window.after(0, lambda: self._show_walk_forward_results(res))
                self.window.after(3000, self.reset_progress)
            except Exception as e:
                import traceback
                self.log(f"Walk-Forward error: {e}\n{traceback.format_exc()}")
                self.reset_progress()
        import threading
        threading.Thread(target=_task, daemon=True).start()

    def _show_walk_forward_results(self, res: dict):
        try:
            windows = res.get("windows", [])
            lines = [f"=== Walk-Forward: {len(windows)} windows ==="]
            for i, w in enumerate(windows):
                lines.append(f"  W{i+1}: WR={w.get('win_rate',0):.1%} PF={w.get('profit_factor',0):.2f} "
                              f"Trades={w.get('total_trades',0)}")
            agg = res.get("aggregate", {})
            if agg:
                lines.append(f"Aggregate WR={agg.get('win_rate',0):.1%} "
                              f"PF={agg.get('profit_factor',0):.2f}")
            self.text_results.config(state=tk.NORMAL)
            self.text_results.delete(1.0, tk.END)
            self.text_results.insert(tk.END, "\n".join(lines))
            self.text_results.config(state=tk.DISABLED)
        except Exception:
            pass


    # ─────────────────────────────────────────────────────────────────
    # Model save / load
    # ─────────────────────────────────────────────────────────────────
    def save_model_dialog(self):
        from tkinter import filedialog
        if self.model is None or not self.model.is_trained:
            self.log("No trained model to save.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Pickle files","*.pkl"),("All files","*.*")],
            title="Model Kaydet")
        if path:
            try:
                self.model.save(path)
                self.log(f"Model saved: {path}")
            except Exception as e:
                self.log(f"Save error: {e}")

    def load_model_dialog(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            filetypes=[("Pickle files","*.pkl"),("All files","*.*")],
            title="Model Yükle")
        if path:
            self._load_model_from_path(path)

    def _load_model_from_path(self, path: str):
        try:
            self.model = MLModel(self.config)
            self.model.load(path)
            self.log(f"Model loaded: {path}")
        except Exception as e:
            self.log(f"Load error: {e}")

    def _auto_load_model(self):
        import os, glob
        patterns = [
            f"mt5_model_{self.config.SYMBOL}*.pkl",
            "mt5_model*.pkl",
            "model*.pkl",
        ]
        for pat in patterns:
            files = sorted(glob.glob(pat), key=os.path.getmtime, reverse=True)
            if files:
                self._load_model_from_path(files[0])
                return

    # ─────────────────────────────────────────────────────────────────
    # Config management
    # ─────────────────────────────────────────────────────────────────
    def save_current_config(self):
        from tkinter import filedialog, simpledialog
        import json
        name = simpledialog.askstring("Config Adı", "Konfigürasyon adı:", parent=self.window)
        if not name:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            initialfile=f"config_{name}.json",
            filetypes=[("JSON","*.json"),("All","*.*")])
        if path:
            try:
                data = {k: v for k, v in vars(self.config).items()
                        if isinstance(v, (int, float, str, bool, list))}
                with open(path, "w") as fp:
                    json.dump(data, fp, indent=2)
                self.log(f"Config saved: {path}")
            except Exception as e:
                self.log(f"Config save error: {e}")

    def load_saved_config(self):
        from tkinter import filedialog
        import json
        path = filedialog.askopenfilename(
            filetypes=[("JSON","*.json"),("All","*.*")],
            title="Config Yükle")
        if path:
            try:
                with open(path) as fp:
                    data = json.load(fp)
                for k, v in data.items():
                    if hasattr(self.config, k):
                        setattr(self.config, k, v)
                self.load_mode_parameters()
                self.log(f"Config loaded: {path}")
            except Exception as e:
                self.log(f"Config load error: {e}")

    def auto_optimize(self):
        def _task():
            try:
                if not hasattr(self, 'data_dict') or not self.data_dict:
                    self.log("No data for optimization.")
                    return
                self.log("Running auto-optimization...")
                self.set_progress(5, "Optimizing...")
                opt = ParameterOptimizer(self.config)
                best = opt.optimize(self.data_dict,
                                    progress_callback=lambda v, d="": self.set_progress(v, d))
                for k, v in best.items():
                    setattr(self.config, k, v)
                self.load_mode_parameters()
                self.log(f"Optimization done: {best}")
                self.set_progress(100, "Done")
                self.window.after(3000, self.reset_progress)
            except Exception as e:
                self.log(f"Optimize error: {e}")
                self.reset_progress()
        import threading
        threading.Thread(target=_task, daemon=True).start()

    def show_saved_configs(self):
        import glob, json, os
        files = glob.glob("config_*.json")
        if not files:
            self.log("No saved configs found.")
            return
        win = tk.Toplevel(self.window)
        win.title("Kayıtlı Konfigürasyonlar")
        win.configure(bg='#2d2d2d')
        win.geometry("400x300")
        lb = tk.Listbox(win, bg='#1e1e1e', fg='white', font=('Courier', 9))
        lb.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        for f in sorted(files):
            lb.insert(tk.END, f)
        def load_selected():
            sel = lb.curselection()
            if sel:
                self._load_model_from_path(files[sel[0]])
                win.destroy()
        tk.Button(win, text="Yükle", command=load_selected,
                  bg='#4CAF50', fg='white').pack(pady=5)

    # ─────────────────────────────────────────────────────────────────
    # Results / analysis display
    # ─────────────────────────────────────────────────────────────────
    def _update_results(self):
        res = getattr(self, 'backtest_results', None)
        if not res:
            return
        lines = [
            "=== Backtest Results ===",
            f"  Total Trades  : {res.get('total_trades',0)}",
            f"  Win Rate      : {res.get('win_rate',0):.1%}",
            f"  Profit Factor : {res.get('profit_factor',0):.2f}",
            f"  Sharpe Ratio  : {res.get('sharpe_ratio',0):.2f}",
            f"  Max Drawdown  : {res.get('max_drawdown',0):.2%}",
            f"  Net Profit    : {res.get('net_profit',0):.2f}",
            f"  Total Return  : {res.get('total_return',0):.2%}",
        ]
        self.text_results.config(state=tk.NORMAL)
        self.text_results.delete(1.0, tk.END)
        self.text_results.insert(tk.END, "\n".join(lines))
        self.text_results.config(state=tk.DISABLED)

        # update summary tab labels
        field_map = {
            'lbl_total_trades': ('total_trades', '{}'),
            'lbl_win_rate':     ('win_rate',     '{:.1%}'),
            'lbl_pf':           ('profit_factor','{:.2f}'),
            'lbl_sharpe':       ('sharpe_ratio', '{:.2f}'),
            'lbl_drawdown':     ('max_drawdown', '{:.2%}'),
            'lbl_net_profit':   ('net_profit',   '{:.2f}'),
        }
        for attr, (key, fmt) in field_map.items():
            lbl = getattr(self, attr, None)
            if lbl:
                try:
                    lbl.config(text=fmt.format(res.get(key, 0)))
                except Exception:
                    pass

    def _update_all_gui(self):
        self._update_results()

    def _log_indicator_summary(self, data_dict):
        try:
            ind = TechnicalIndicators(self.config)
            df = data_dict.get('M15', data_dict.get(list(data_dict.keys())[0]))
            df2 = ind.calculate_all(df)
            last = df2.iloc[-1]
            self.log(f"RSI={last.get('RSI',float('nan')):.1f} "
                     f"MACD={last.get('MACD',float('nan')):.5f} "
                     f"EMA9={last.get('EMA9',float('nan')):.5f}")
        except Exception:
            pass

    def _plot_equity_curve(self):
        res = getattr(self, 'backtest_results', None)
        if not res or 'equity_curve' not in res:
            self.log("No equity curve data.")
            return
        try:
            import matplotlib.pyplot as plt
            equity = res['equity_curve']
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(equity, color='cyan', linewidth=1)
            ax.set_title('Equity Curve'); ax.set_xlabel('Trade #'); ax.set_ylabel('Capital')
            ax.grid(True, alpha=0.3)
            fig.patch.set_facecolor('#1e1e1e'); ax.set_facecolor('#1e1e1e')
            ax.tick_params(colors='white'); ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white'); ax.title.set_color('white')
            plt.tight_layout(); plt.show()
        except Exception as e:
            self.log(f"Plot error: {e}")

    def _show_trades(self):
        res = getattr(self, 'backtest_results', None)
        if not res or 'trades' not in res:
            self.log("No trade data.")
            return
        trades = res['trades']
        win = tk.Toplevel(self.window)
        win.title("Trade Listesi"); win.configure(bg='#2d2d2d'); win.geometry("700x400")
        cols = ['entry_time','exit_time','type','entry_price','exit_price','pnl','pips']
        import tkinter.ttk as ttk2
        tree = ttk2.Treeview(win, columns=cols, show='headings', height=20)
        for c in cols:
            tree.heading(c, text=c); tree.column(c, width=90)
        sb = tk.Scrollbar(win, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        sb.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        for t in trades:
            row = [str(t.get(c,"")) for c in cols]
            tag = 'win' if t.get('pnl',0) > 0 else 'loss'
            tree.insert("", tk.END, values=row, tags=(tag,))
        tree.tag_configure('win', foreground='#4CAF50')
        tree.tag_configure('loss', foreground='#f44336')

    def _plot_analysis(self):
        res = getattr(self, 'backtest_results', None)
        if not res:
            self.log("No backtest results.")
            return
        try:
            import matplotlib.pyplot as plt
            trades = res.get('trades', [])
            pnls = [t.get('pnl', 0) for t in trades]
            if not pnls:
                self.log("No trades to plot.")
                return
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].bar(range(len(pnls)), pnls,
                        color=['green' if p > 0 else 'red' for p in pnls], alpha=0.7)
            axes[0].set_title('P&L per Trade'); axes[0].axhline(0, color='white', lw=0.5)
            cumulative = [sum(pnls[:i+1]) for i in range(len(pnls))]
            axes[1].plot(cumulative, color='cyan'); axes[1].set_title('Cumulative P&L')
            for ax in axes:
                ax.set_facecolor('#1e1e1e'); ax.tick_params(colors='white')
                ax.title.set_color('white')
            fig.patch.set_facecolor('#1e1e1e')
            plt.tight_layout(); plt.show()
        except Exception as e:
            self.log(f"Analysis plot error: {e}")

    def _analyze_indicators(self):
        if not hasattr(self, 'data_dict') or not self.data_dict:
            self.log("No data for indicator analysis.")
            return
        self._log_indicator_summary(self.data_dict)

    def _export_indicator_stats(self):
        if not hasattr(self, 'data_dict') or not self.data_dict:
            self.log("No data.")
            return
        try:
            from tkinter import filedialog
            path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                filetypes=[("CSV","*.csv")])
            if path:
                df = list(self.data_dict.values())[0]
                ind = TechnicalIndicators(self.config)
                df2 = ind.calculate_all(df)
                df2.describe().to_csv(path)
                self.log(f"Stats exported: {path}")
        except Exception as e:
            self.log(f"Export error: {e}")

    def _plot_indicator_distributions(self):
        if not hasattr(self, 'data_dict') or not self.data_dict:
            self.log("No data.")
            return
        try:
            import matplotlib.pyplot as plt
            df = list(self.data_dict.values())[0]
            ind = TechnicalIndicators(self.config)
            df2 = ind.calculate_all(df)
            cols = [c for c in ['RSI','MACD','ATR','BB_Upper'] if c in df2.columns]
            if not cols:
                self.log("No indicator columns found.")
                return
            fig, axes = plt.subplots(1, len(cols), figsize=(4*len(cols), 4))
            if len(cols) == 1:
                axes = [axes]
            for ax, c in zip(axes, cols):
                ax.hist(df2[c].dropna(), bins=40, color='cyan', alpha=0.7, edgecolor='none')
                ax.set_title(c); ax.set_facecolor('#1e1e1e'); ax.tick_params(colors='white')
                ax.title.set_color('white')
            fig.patch.set_facecolor('#1e1e1e')
            plt.tight_layout(); plt.show()
        except Exception as e:
            self.log(f"Distribution plot error: {e}")

    def create_launcher_files(self):
        try:
            import os
            sym = self.config.SYMBOL
            script = f'''#!/usr/bin/env python3
import subprocess, sys
subprocess.run([sys.executable, "{os.path.abspath(__file__)}", "--symbol", "{sym}", "--autostart"])
'''
            fname = f"launch_{sym}.py"
            with open(fname, "w") as fp:
                fp.write(script)
            os.chmod(fname, 0o755)
            self.log(f"Launcher created: {fname}")
        except Exception as e:
            self.log(f"Launcher error: {e}")


    # ─────────────────────────────────────────────────────────────────
    # Live Trading control
    # ─────────────────────────────────────────────────────────────────
    def start_live_trading(self):
        if self.is_trading:
            self.log("Already trading.")
            return
        if self.model is None or not self.model.is_trained:
            self.log("No trained model — attempting auto-load...")
            self._auto_load_model()
            if self.model is None or not self.model.is_trained:
                self.log("Cannot start: no model.")
                return
        self.apply_parameters()
        self._apply_lot_settings_to_config()

        symbols = [s.strip() for s in self.config.SYMBOL.split(",") if s.strip()]
        if not symbols:
            symbols = [self.config.SYMBOL]

        self.is_trading = True
        self.is_paused = False
        self.live_traders = {}

        for sym in symbols:
            cfg = Config()
            cfg.__dict__.update(self.config.__dict__)
            cfg.SYMBOL = sym
            lt = LiveTrader(cfg, self.model)
            lt.log_callback   = self.on_live_log
            lt.signal_callback = self.on_live_signal
            lt.trade_callback  = self.on_live_trade
            lt.error_callback  = self.on_live_error
            self.live_traders[sym] = lt

        self._trading_thread = threading.Thread(
            target=self._multi_trading_loop, daemon=True)
        self._trading_thread.start()

        self._monitor_thread = threading.Thread(
            target=self.monitor_positions_loop, daemon=True)
        self._monitor_thread.start()

        self.btn_start.config(state=tk.DISABLED)
        self.btn_pause.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.NORMAL)
        paper = "PAPER" if self.config.PAPER_TRADING else "LIVE"
        syms_str = ", ".join(symbols)
        self.log(f"Trading started [{paper}]: {syms_str}")

    def _multi_trading_loop(self):
        import time
        symbols = list(self.live_traders.keys())
        for sym, lt in self.live_traders.items():
            lt.start(sym, self.config.TIMEFRAMES, auto_trade=True)
        while self.is_trading:
            if not self.is_paused:
                for sym, lt in list(self.live_traders.items()):
                    try:
                        lt.trading_loop_iteration()
                    except Exception as e:
                        self.on_live_error(sym, str(e))
            time.sleep(self.config.LOOP_INTERVAL_SECONDS)

    def pause_live_trading(self):
        if not self.is_trading:
            return
        self.is_paused = not self.is_paused
        state = "PAUSED" if self.is_paused else "RESUMED"
        self.log(f"Trading {state}")
        txt = "▶ Devam" if self.is_paused else "⏸ Duraklat"
        self.btn_pause.config(text=txt)

    def stop_live_trading(self):
        self.is_trading = False
        self.is_paused = False
        for lt in self.live_traders.values():
            try:
                lt.stop()
            except Exception:
                pass
        self.live_traders.clear()
        self.btn_start.config(state=tk.NORMAL)
        self.btn_pause.config(state=tk.DISABLED, text="⏸ Duraklat")
        self.btn_stop.config(state=tk.DISABLED)
        self.log("Trading stopped.")

    def emergency_stop(self):
        self.stop_live_trading()
        # Close all open positions
        try:
            import MetaTrader5 as mt5
            positions = mt5.positions_get()
            if positions:
                for pos in positions:
                    order_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": pos.symbol,
                        "volume": pos.volume,
                        "type": order_type,
                        "position": pos.ticket,
                        "deviation": 20,
                        "magic": self.config.MAGIC_NUMBER,
                        "comment": "emergency_stop",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    mt5.order_send(request)
                self.log(f"Emergency stop: closed {len(positions)} positions.")
        except Exception as e:
            self.log(f"Emergency stop error: {e}")

    # ─────────────────────────────────────────────────────────────────
    # Position monitor
    # ─────────────────────────────────────────────────────────────────
    def monitor_positions_loop(self):
        import time
        while self.is_trading:
            try:
                self.window.after(0, self._update_positions_display)
                self.window.after(0, self._update_live_stats)
            except Exception:
                pass
            time.sleep(5)

    def _update_positions_display(self):
        try:
            lines = []
            for sym, lt in self.live_traders.items():
                positions = lt.get_open_positions(sym)
                if positions:
                    for pos in positions:
                        if self.config.PAPER_TRADING:
                            lines.append(f"[PAPER] {sym} | {pos.get('type','?')} "
                                         f"vol={pos.get('volume',0):.2f} "
                                         f"pnl={pos.get('profit',0):.2f}")
                        else:
                            lines.append(f"{pos.symbol} | {'BUY' if pos.type==0 else 'SELL'} "
                                         f"vol={pos.volume:.2f} profit={pos.profit:.2f}")
            if not lines:
                lines = ["Açık pozisyon yok"]
            self.text_positions.config(state=tk.NORMAL)
            self.text_positions.delete(1.0, tk.END)
            self.text_positions.insert(tk.END, "\n".join(lines))
            self.text_positions.config(state=tk.DISABLED)
        except Exception:
            pass

    def _update_live_stats(self):
        try:
            total_pnl = 0.0
            total_trades = 0
            for sym, lt in self.live_traders.items():
                total_pnl += getattr(lt, 'daily_pnl', 0)
                total_trades += getattr(lt, 'daily_trades', 0)
            lines = [
                f"Günlük P&L   : {total_pnl:+.2f}",
                f"Günlük Trade : {total_trades}",
                f"Semboller    : {len(self.live_traders)}",
                f"Mod          : {'PAPER' if self.config.PAPER_TRADING else 'LIVE'}",
                f"Durum        : {'TRADING' if self.is_trading else 'STOPPED'}",
            ]
            self.text_live_stats.config(state=tk.NORMAL)
            self.text_live_stats.delete(1.0, tk.END)
            self.text_live_stats.insert(tk.END, "\n".join(lines))
            self.text_live_stats.config(state=tk.DISABLED)
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────
    # Live callbacks
    # ─────────────────────────────────────────────────────────────────
    def on_live_log(self, message: str):
        self.window.after(0, lambda: self.log(message))

    def on_live_signal(self, symbol: str, signal: int, confidence: float,
                       h4_score: int = 0, tier3_ok: bool = True):
        direction = "BUY" if signal == 1 else "SELL"
        info = (f"Symbol   : {symbol}\n"
                f"Direction: {direction}\n"
                f"Confidence: {confidence:.1%}\n"
                f"H4 Score : {h4_score}/5\n"
                f"M15 Conf : {'✅' if tier3_ok else '❌'}")
        if self.config.REQUIRE_MANUAL_APPROVAL:
            self.pending_signal = {"symbol": symbol, "signal": signal,
                                   "confidence": confidence}
            self.window.after(0, lambda: self._show_signal_for_approval(info))
        else:
            self.window.after(0, lambda: self.log(f"Signal: {symbol} {direction} {confidence:.1%}"))

    def _show_signal_for_approval(self, info: str):
        self.lbl_signal_info.config(text=info)
        if not getattr(self, 'approval_frame', None):
            return
        self.approval_frame.pack(fill=tk.X, padx=5, pady=3)
        self.btn_approve.config(state=tk.NORMAL)
        self.btn_reject.config(state=tk.NORMAL)

    def on_live_trade(self, symbol: str, result: dict):
        msg = (f"Trade: {symbol} {result.get('type','?')} "
               f"lot={result.get('volume',0):.2f} "
               f"entry={result.get('price',0):.5f}")
        self.window.after(0, lambda: self.log(msg))

    def on_live_error(self, symbol: str, error: str):
        self.window.after(0, lambda: self.log(f"ERROR [{symbol}]: {error}"))

    def approve_signal(self):
        sig = getattr(self, 'pending_signal', None)
        if not sig:
            return
        sym = sig["symbol"]
        self.log(f"Signal APPROVED: {sym}")
        if sym in self.live_traders:
            self.live_traders[sym].approved_signal = sig
        self.pending_signal = None
        self.btn_approve.config(state=tk.DISABLED)
        self.btn_reject.config(state=tk.DISABLED)

    def reject_signal(self):
        sig = getattr(self, 'pending_signal', None)
        if sig:
            self.log(f"Signal REJECTED: {sig.get('symbol','?')}")
        self.pending_signal = None
        self.btn_approve.config(state=tk.DISABLED)
        self.btn_reject.config(state=tk.DISABLED)

    # ─────────────────────────────────────────────────────────────────
    # Run
    # ─────────────────────────────────────────────────────────────────
    def run(self):
        self.log("MT5 Trading System v6.5 başlatıldı.")
        self.log(f"Symbol: {self.config.SYMBOL} | Mode: {self.config.TRADING_MODE}")
        if self._init_model_path:
            self._load_model_from_path(self._init_model_path)
        else:
            self._auto_load_model()
        if self._init_autostart:
            self.window.after(1500, self.start_live_trading)
        self.window.mainloop()



# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="MT5 Trading System v6.5")
    parser.add_argument("--symbol",    default=None, help="Trading symbol (e.g. EURUSD)")
    parser.add_argument("--model",     default=None, help="Path to pre-trained model .pkl")
    parser.add_argument("--autostart", action="store_true", help="Auto-start live trading on launch")
    parser.add_argument("--paper",     action="store_true", help="Force paper trading mode")
    args = parser.parse_args()

    config = Config()
    if args.symbol:
        config.SYMBOL = args.symbol
    if args.paper:
        config.PAPER_TRADING = True

    app = TradingGUI(
        symbol=config.SYMBOL,
        model_path=args.model,
        autostart=args.autostart,
    )
    app.config = config
    app.run()


if __name__ == "__main__":
    main()
