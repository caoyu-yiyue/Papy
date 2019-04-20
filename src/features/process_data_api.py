"""
一些生成OLS 回归的features 和targets 所用的函数。
"""

import pandas as pd
import numpy as np
from src.data import preparing_data as predata


def obtain_feature_index(reverse_ret_dframe: pd.DataFrame):
    """
    返回reverse_ret_dframe 的index 中的Trddt 一列，作为下一步features 使用的index。
    Parameters:
    -----------
    reverse_ret_dframe:
        pd.DataFrame
        保存有反转组合收益的数据框，下一步作为ols 的Target 使用

    Results:
    --------
    pandas.Index:
        一列与输入的数据框的Trddt 相同的index
    """

    # 从reverse_ret_dframe 的index 中，抛去不需要的level，并去重，拿出日期那一列的level
    year_idx = reverse_ret_dframe.index.droplevel('cap_group').unique()
    return year_idx


def calcualte_market_exc_ret():
    """
    计算市场超额收益率序列
    Results:
        pd.Series
        市场超额收益率序列
    """
    ret_market: pd.Series = predata.read_rm_data()
    rf: pd.Series = predata.read_rf_data()

    rm_exc: pd.Series = ret_market - rf
    return rm_exc


def calculate_stds(std_roll_window: int = 20):
    """
    计算以过去std_roll_window 天滚动得到的市场历史波动率的对数值，以及相邻两天波动率的变动量
    Parameters:
    -----------
    std_roll_window:
        int
        滚动计算标准差时，选用的滚动窗口日期
    Results:
    --------
    tuple:
        返回一个tuple，第0 个值为滚动标准差的对数值，第1 个值为标准差的变动值（差分值）
    """

    # 读取市场指数文件
    market_index: pd.Series = predata.read_market_index_data()

    # 计算历史滚动波动率，以及其差分值（变动情况）
    rolling_std: pd.Series = market_index.rolling(window=std_roll_window).std()
    delta_std: pd.Series = rolling_std.diff()
    rolling_std_log: pd.Series = np.log(rolling_std)

    return (rolling_std_log, delta_std)


def shift_leading_gradually(benchmark: pd.Series,
                            col_name_prefix: str,
                            leading_time: int = 5):
    """
    输入一个列benchmark 作为基准，每一行依次添加未来t+1，...，t+leading_time 的值。
    最终返回一个benchmark 错位后的t+1, ..., t+leading_time 列组成的DataFrame。

    Parameters:
    -----------
    benchmark:
        pd.Series
        需要进行错位的列

    col_name_prefix:
        str
        返回的列名的前缀

    leading_time:
        int, default 5
        最大的错位时间

    Results:
    --------
    pd.DataFrame
        benchmark 错位后，以t+1，...,t+leading_time 为列组成的数据框
    """

    dframe_added_leading = pd.DataFrame()

    # 以benchmark 分别向前错位1，2，...，leading_time，然后组合到同一个DataFrame 中
    for index in range(1, (leading_time + 1)):
        # 每一天后面需要追加一个t+1,...,leading_time, 所以shift 中使用负的的index 值进行错位
        shifted_serie = benchmark.shift(-index)
        dframe_added_leading[col_name_prefix +
                             '_t+{}'.format(index)] = shifted_serie

    return dframe_added_leading


def generate_targets(reverse_ret_dframe: pd.DataFrame):
    """
    从反转收益组合数据，生成用于OLS 分组回归的targets 数据框
    Parameters:
    -----------
    reverse_ret_dframe:
        pd.DataFrame
        反转组合收益率的时间序列表格，index 为时间和规模，columns 为不同反转策略（如Lo-Hi)

    Results:
    --------
    pd.DataFrame
        用于OLS 回归的数据框，index 为时间、规模组、反转策略，columns 为收益率
    """

    # 按照所有的index 分组
    grouped_df = reverse_ret_dframe.groupby(reverse_ret_dframe.index.names)

    # 每组去掉index 后转置
    target_df = grouped_df.apply(lambda df: df.reset_index(drop=True).T)

    # 为target_df 数据框的index 和列命名
    target_df.rename_axis(['Trddt', 'cap_group', 'rev_group'], inplace=True)
    target_df.rename(columns={0: 'rev_ret'}, inplace=True)

    return target_df


def get_rm_features(file='data/processed/rm_features.pickle'):
    """
    从保存的文件中读取市场超额收益率features（市场超额收益率的t+1,...t+5 期）
    Parameters:
    -----------
    file:
        str(path for rm_features file)
        保存rm_features 的文件路径

    Results:
    --------
    pandas.DataFrame:
        未来五日市场超额收益率组成的一个pandas.DataFrame
    """

    dframe = pd.read_pickle(file)
    return dframe


def _get_std_features_dframe(file='data/processed/std_features.pickle'):
    """
    从保存的文件中读取波动率、波动率变动未来五期值组成的features 数据框。
    Parameters:
    -----------
    file:
        str(path for std_features file)
        保存有和波动率有关的数据框的路径

    Results:
    --------
    pandas.DataFrame:
        收益率和未来五天波动率变动共同组成的pandas.DataFrame
    """

    dframe = pd.read_pickle(file)
    return dframe


def get_rolling_std_features(file='data/processed/std_features.pickle'):
    """
    从保存的波动率features 文件中，获得滚动波动率（roling_std）这一项
    Parameters:
    -----------
    file:
        str(path to std_feautres file)

    Results:
    --------
    pandas.DataFrame:
        滚动的波动率数据框
    """

    std_dframe = pd.read_pickle(file)
    return std_dframe['rolling_std'].to_frame()


def get_delta_std_features(file='data/processed/std_features.pickle'):
    """
    从保存的波动率features 文件中，获得五列波动率变动（delta_std）的数据列
    Parameters:
    -----------
    file:
        str(path to std_feautres file)

    Results:
    --------
    pandas.DataFrame:
        未来五天波动率变动值的数据框
    """

    std_dframe = pd.read_pickle(file)
    delta_std_col = [col for col in std_dframe if col.startswith('delta_')]
    return std_dframe[delta_std_col]


def get_targets(file='data/processed/targets.pickle'):
    """
    从保存的文件中读取OLS 回归所用的targets，即超额收益率计算的反转组合收益
    Parameters:
    -----------
    file:
        str(path to the targets file)

    Results:
    --------
    pandas.DataFrame:
        读取的targets 数据框
    """

    dframe = pd.read_pickle(file)
    return dframe
