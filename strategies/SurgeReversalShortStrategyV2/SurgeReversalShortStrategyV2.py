from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import numpy as np

class SurgeReversalShortStrategyV2(IStrategy):
    """
    增强版策略：
    在加密货币价格24小时内暴涨后，结合K线形态、成交量、MACD、RSI、StochRSI、布林带等指标，
    捕捉顶部趋势反转信号，进行短线做空交易。
    """
    timeframe = '15m'
    can_short = True
    stoploss = -0.05
    minimal_roi = {}
    process_only_new_candles = True
    startup_candle_count = 120

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        计算所有用于入场和出场的技术指标
        """
        if dataframe.empty:
            return dataframe

        # 计算24小时内最低价及当前涨幅百分比
        dataframe['min_24h'] = dataframe['close'].rolling(window=96, min_periods=1).min()
        dataframe['pct_change_24h'] = ((dataframe['close'] - dataframe['min_24h']) / dataframe['min_24h']) * 100

        # RSI指标
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # MACD指标
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macd_signal'] = macd['macdsignal']

        # 布林带
        boll = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe['bb_upper'] = boll['upperband']
        dataframe['bb_mid'] = boll['middleband']
        dataframe['bb_lower'] = boll['lowerband']

        # 成交量平均线
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()

        # Stochastic RSI
        stoch_rsi = ta.STOCHRSI(dataframe, timeperiod=14)
        dataframe['stochrsi_k'] = stoch_rsi['fastk']
        dataframe['stochrsi_d'] = stoch_rsi['fastd']

        # ATR
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        # dataframe.dropna(inplace=True)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        生成做空信号
        """
        dataframe['enter_short'] = 0
        dataframe['enter_long'] = 0  # 不使用做多

        # 条件1：暴涨确认
        cond_surge = dataframe['pct_change_24h'] > 30

        # 条件2：成交量放大
        cond_volume_confirm = dataframe['volume'] > 1.2 * dataframe['volume_mean']

        # 条件3：反转K线形态
        # body = abs(dataframe['close'] - dataframe['open'])
        # range_ = dataframe['high'] - dataframe['low'] + 1e-9
        # upper_wick = dataframe['high'] - dataframe[['open', 'close']].max(axis=1)
        # lower_wick = dataframe[['open', 'close']].min(axis=1) - dataframe['low']

        # cond_doji = (body / range_ < 0.1)
        # cond_shooting = (upper_wick > 0.6 * range_) & (body < 0.3 * range_)
        # cond_big_red = (dataframe['close'] < dataframe['open'] * 0.95)
        # cond_3_red = (
        #     (dataframe['close'] < dataframe['open']) &
        #     (dataframe['close'].shift(1) < dataframe['open'].shift(1)) &
        #     (dataframe['close'].shift(2) < dataframe['open'].shift(2))
        # )

        # cond_reversal_candle = cond_doji | cond_shooting | cond_big_red | cond_3_red
        cond_reversal_candle = True

        # 条件4：动能反转信号（至少满足两个）
        cond_macd_bear = (dataframe['macd'] < dataframe['macd_signal']) & (dataframe['macd'].shift(1) > dataframe['macd_signal'].shift(1))
        cond_rsi_bear = (dataframe['rsi'].shift(1) > 70) & (dataframe['rsi'] < dataframe['rsi'].shift(1))
        cond_stochrsi_revert = (dataframe['stochrsi_k'].shift(1) > 80) & (dataframe['stochrsi_k'] < dataframe['stochrsi_k'].shift(1))
        cond_bb_revert = (dataframe['close'].shift(1) > dataframe['bb_upper'].shift(1)) & (dataframe['close'] < dataframe['bb_upper'])

        momentum_signals = [cond_macd_bear, cond_rsi_bear, cond_stochrsi_revert, cond_bb_revert]
        signal_sum = sum([cond.astype(int) for cond in momentum_signals])
        cond_momentum_confirm = signal_sum >= 2

        # 合并所有条件
        entry_condition = cond_surge & cond_volume_confirm & cond_reversal_candle & cond_momentum_confirm
        dataframe.loc[entry_condition, 'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        生成做空平仓信号
        """
        dataframe['exit_short'] = 0
        dataframe['exit_long'] = 0

        cond_rsi_oversold = dataframe['rsi'] < 30
        cond_macd_bull = (dataframe['macd'] > dataframe['macd_signal']) & (dataframe['macd'].shift(1) < dataframe['macd_signal'].shift(1))
        cond_bb_mean_revert = dataframe['close'] < dataframe['bb_mid']
        cond_bounce_zone = (dataframe['close'] < dataframe['bb_lower']) & (dataframe['volume'] < dataframe['volume_mean'])

        exit_condition = cond_rsi_oversold | cond_macd_bull | cond_bb_mean_revert | cond_bounce_zone
        dataframe.loc[exit_condition, 'exit_short'] = 1

        return dataframe
