# Freqtrade 策略：在虚拟货币价格暴涨后做空，基于15分钟K线，捕捉趋势反转信号
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import numpy as np

class SurgeReversalShortStrategy(IStrategy):
    """
    该策略在虚拟货币经历24小时内超过40%的暴涨后，捕捉价格顶部的趋势反转信号并做空。
    使用15分钟K线，通过以下几个技术指标组合判断入场和出场时机：
    - MACD死叉
    - RSI超买回落
    - K线反转形态（十字星、长上影线、大阴线）
    - （可选）布林带回归信号
    """
    timeframe = '15m'                # 使用的时间周期为15分钟
    can_short = True                 # 启用做空模式（需支持合约交易）
    stoploss = -0.05                 # 最大止损为5%（做空时为负值）
    minimal_roi = {"0": 10}          # 不设置固定止盈（设置一个非常大的值）
    process_only_new_candles = True  # 仅处理新生成的K线
    startup_candle_count = 120       # 启动时需要的最少历史K线数量（约25小时）

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """计算所需的技术指标"""
        # 过去24小时内的最低价
        dataframe['min_24h'] = dataframe['close'].rolling(window=96).min()
        # 当前价相对于24小时最低价的涨幅（最大上涨幅度）
        dataframe['pct_change_24h'] = ((dataframe['close'] - dataframe['min_24h']) / dataframe['min_24h']) * 100

        # 计算14周期的 RSI 指标
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # 计算MACD（包含macd线、信号线）
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macd_signal'] = macd['macdsignal']

        # 计算布林带（可选，用于判断价格回归）
        # bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2, nbdevdn=2)
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe['bb_upper'] = bollinger['upperband']
        dataframe['bb_mid'] = bollinger['middleband']
        dataframe['bb_lower'] = bollinger['lowerband']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """定义做空信号的触发条件"""
        dataframe['enter_long'] = 0   # 明确不做多
        dataframe['enter_short'] = 0  # 初始化做空信号

        # 条件1：过去24小时涨幅超过20%
        cond_surge = dataframe['pct_change_24h'] > 20
        # cond_surge = True

        # 计算当前K线的K线结构参数：实体、高低范围、上下影线
        body = abs(dataframe['close'] - dataframe['open'])
        range_ = (dataframe['high'] - dataframe['low']) + 1e-9  # 防止除0
        upper_wick = dataframe['high'] - dataframe[['open', 'close']].max(axis=1)
        lower_wick = dataframe[['open', 'close']].min(axis=1) - dataframe['low']

        # 条件2-1：顶部十字星形态（实体非常小）
        cond_doji = (body / range_ < 0.1)

        # 条件2-2：上影线长、实体小，形成“射击之星”形态
        cond_shooting = (
            (upper_wick > 0.6 * range_) &
            (body < 0.3 * range_) &
            ((dataframe[['open', 'close']].max(axis=1) - dataframe['low']) < 0.5 * range_)
        )

        # 条件2-3：快速大阴线（单根K线下跌超过5%）
        cond_big_red = (dataframe['close'] < dataframe['open'] * 0.95)

        # 合并K线反转信号
        cond_reversal_candle = cond_doji | cond_shooting | cond_big_red

        # 条件3-1：MACD死叉（macd线下穿信号线）
        cond_macd_bear = (
            (dataframe['macd'] < dataframe['macd_signal']) &
            (dataframe['macd'].shift(1) > dataframe['macd_signal'].shift(1))
        )

        # 条件3-2：RSI从超买区域回落
        cond_rsi_bear = (
            (dataframe['rsi'].shift(1) > 70) &
            (dataframe['rsi'] < dataframe['rsi'].shift(1))
        )

        # 条件3-3（可选）：布林带回归（价格上穿上轨后回落）
        cond_bb_revert = (
            (dataframe['close'].shift(1) > dataframe['bb_upper'].shift(1)) &
            (dataframe['close'] < dataframe['bb_upper'])
        )

        # 最终做空入场条件：暴涨 + 反转形态 + 动能信号（满足其一）
        entry_condition = cond_surge & cond_reversal_candle & (cond_macd_bear | cond_rsi_bear | cond_bb_revert)

        dataframe.loc[entry_condition, 'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """定义做空平仓信号"""
        dataframe['exit_long'] = 0    # 不使用做多
        dataframe['exit_short'] = 0   # 初始化做空平仓信号

        # 条件1：RSI进入超卖区（可能下跌到位）
        cond_rsi_oversold = dataframe['rsi'] < 30

        # 条件2：MACD金叉（趋势可能反转向上）
        cond_macd_bull = (
            (dataframe['macd'] > dataframe['macd_signal']) &
            (dataframe['macd'].shift(1) < dataframe['macd_signal'].shift(1))
        )

        # 条件3（可选）：价格跌破布林中轨（可能进入支撑区）
        cond_bb_mean_revert = dataframe['close'] < dataframe['bb_mid']

        # 合并退出条件
        exit_condition = cond_rsi_oversold | cond_macd_bull | cond_bb_mean_revert

        dataframe.loc[exit_condition, 'exit_short'] = 1

        return dataframe