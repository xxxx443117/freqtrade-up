from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
from technical import qtpylib

class TrendFollowing15m(IStrategy):
    """
    趋势追踪策略（适用于 15 分钟K线）
    - 使用 EMA 判断当前价格趋势（短期 vs 长期）
    - 使用 ADX 判断趋势强度（过滤震荡）
    - 使用 MACD 判断进出场时机（动量变化）
    - 启用止损与追踪止盈保护
    """

    # 使用的时间周期为 15 分钟
    timeframe = '15m'

    # 不启用做空
    can_short = False

    # 最小收益率策略（为空时使用信号或追踪止盈控制出场）
    minimal_roi = {}

    # 固定止损：亏损达到 -10% 自动止损
    stoploss = -0.10

    # 启用追踪止盈
    trailing_stop = True
    trailing_stop_positive = 0.04              # 盈利回撤 4% 时止盈
    trailing_stop_positive_offset = 0.06       # 盈利达到 6% 才开始追踪止盈
    trailing_only_offset_is_reached = True     # 只有达到偏移后才启用追踪止盈

    # 需要的最少K线数（至少100根用于计算 EMA）
    startup_candle_count: int = 120

    # 只处理新的蜡烛（提高效率）
    process_only_new_candles = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        计算并添加所需的技术指标到 dataframe 中
        """

        # 计算短期与长期的 EMA，用于判断趋势方向
        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)

        # 计算 MACD 指标（macd主线、信号线、柱状图）
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # 计算 ADX 指标，用于识别趋势强度
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=10)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        定义买入逻辑（开多单）
        条件：
        - EMA20 > EMA100：短期在上方，趋势向上
        - ADX > 20：市场处于较强趋势
        - MACD 金叉（动能增强）
        """
        dataframe['enter_long'] = 0  # 初始化列

        buy_condition = (
            (dataframe['ema20'] > dataframe['ema100']) &                                   # 上升趋势
            (dataframe['adx'] > 20) &                                                      # 趋势强度足够
            (qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal'])) &          # MACD 金叉
            (dataframe['volume'] > 0)                                                      # 避免无量交易
        )

        dataframe.loc[buy_condition, 'enter_long'] = 1  # 设置买入信号
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        定义卖出逻辑（平多单）
        条件：
        - EMA20 < EMA100：趋势可能反转
        - 或 MACD 死叉：上涨动能衰退
        """
        dataframe['exit_long'] = 0  # 初始化列

        sell_condition = (
            (
                (dataframe['ema20'] < dataframe['ema100']) |                                # 趋势反转
                (qtpylib.crossed_below(dataframe['macd'], dataframe['macdsignal']))         # MACD 死叉
            ) &
            (dataframe['volume'] > 0)                                                       # 有成交量
        )

        dataframe.loc[sell_condition, 'exit_long'] = 1  # 设置卖出信号
        return dataframe
