"""
一些生成OLS 回归的features 和targets 所用的函数。
"""

import pandas as pd
import numpy as np
from src.data import preparing_data as predata
from src.features import reverse_port_ret as rpt


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
    year_idx = reverse_ret_dframe.index.get_level_values('Trddt').unique()
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


def calculate_stds(std_roll_window: int = 20, forward_window: int = 5):
    """
    计算以过去std_roll_window 天滚动得到的市场历史波动率的对数值，以及相邻两天波动率的变动量、
    相邻forward_window 天波动率的变动量。
    Parameters:
    -----------
    std_roll_window:
        int
        滚动计算标准差时，选用的滚动窗口日期
    forward_window:
        int
        计算未来某些天的波动率的变动量所有的天数
    Results:
    --------
    tuple:
        返回一个tuple，第0 个值为滚动标准差的对数值，第1 个值为相邻两天标准差的变动值（差分值），
        第二个值为相邻forward_window 天标准差的变动值。
    """

    # 读取市场指数文件
    market_index: pd.Series = predata.read_market_index_data()

    # 计算历史滚动波动率，以及其差分值（变动情况）
    rolling_std: pd.Series = market_index.rolling(window=std_roll_window).std()
    delta_std_1day: pd.Series = rolling_std.diff()
    delta_std_full_forward: pd.Series = rolling_std.diff(forward_window)
    rolling_std_log: pd.Series = np.log(rolling_std)

    return (rolling_std_log, delta_std_1day, delta_std_full_forward)


def calculate_turnover(backward_window, forward_window):
    """
    根据输入的向前和向后窗口，计算组合的turnover。

    Parameters：
    -----------
    backward_window, forward_window:
        int, 向前和向后计算组合的窗口长度
    """
    prepared_data: pd.DataFrame = predata.read_prepared_data()
    turnover_df: pd.DataFrame = predata.read_turnover_data()
    turnover_merged: pd.DataFrame = prepared_data.join(turnover_df,
                                                       on=['Stkcd', 'Trddt'])

    rev_port_turnover: pd.Series = rpt.reverse_port_ret_quick(
        turnover_merged,
        backward_window=backward_window,
        forward_window=forward_window,
        backward_method=rpt._normalize_last,
        forward_method=sum,
        col_for_backward_looking='log_ret',
        col_for_forward_looking='turnOver',
        average_in='reverse_group')

    return rev_port_turnover


def calculate_amihud(backward_window, forward_window):
    """
    以backward_window 和forward_window 为参数计算组合的amihud 指标。

    Return:
    -------
        pd.Series
    """
    # 计算amihud 指标的主函数
    prepared_data: pd.DataFrame = predata.read_prepared_data()

    # 前一部分和计算反转组合收益的步骤一样
    # add a column for nomolized return for each stock
    prepared_data["norm_ret"] = rpt.backward_rolling_apply(
        df=prepared_data,
        window=backward_window,
        method=rpt._normalize_last,
        calcu_column="log_ret",
    )

    # drop na values
    prepared_data.dropna(inplace=True)

    # add a captain group sign
    prepared_data["cap_group"] = rpt.creat_group_signs(
        df=prepared_data,
        column_to_cut="Dsmvosd",
        groupby_column="Trddt",
        quntiles=5,
        labels=["Small", "2", "3", "4", "Big"],
    )

    # add a column for return group
    prepared_data["ret_group"] = rpt.creat_group_signs(
        df=prepared_data,
        column_to_cut="norm_ret",
        groupby_column="Trddt",
        quntiles=10,
        labels=["Lo", "2", "3", "4", "5", "6", "7", "8", "9", "Hi"],
    )

    # 加入一组表示反转组合组别的代号
    prepared_data["rev_group"] = prepared_data["ret_group"].apply(
        rpt._creat_rev_group)

    # %%
    # 计算每天的Amihud 指标
    prepared_data['dollar_volume_today'] = prepared_data[
        'Clsprc'] * prepared_data['Dnshrtrd']
    prepared_data['amihud'] = (prepared_data['Dretwd'].abs() /
                               prepared_data['dollar_volume_today'])

    # %%
    # 计算组合加权平均的Amihud 指标
    rev_port_amihud: pd.Series = rpt.weighted_average_by_group(
        prepared_data,
        groupby_columns=['Trddt', 'cap_group', 'rev_group'],
        calcu_column='amihud',
        weights_column='dollar_volume_today')

    # %%
    # 在未来五天内进行平均，得到最终的Amihud 指标
    rev_port_amihud.name = 'amihud'
    amihud_time_ave: pd.Series = rpt.forward_rolling_apply(
        df=rev_port_amihud.to_frame(),
        window=forward_window,
        method=np.average,
        groupby_column=['cap_group', 'rev_group'],
        calcu_column='amihud')
    amihud_time_ave.dropna(inplace=True)

    # 更改index 顺序
    amihud_reindex: pd.Series = amihud_time_ave.reindex(
        ['Lo-Hi', '2-9', '3-8', '4-7', '5-6'], level=2)

    return amihud_reindex


def calculate_ret_sign(port_ret: pd.Series):
    """
    输入组合反转收益率，返回与其维度相同的、表示其为正或负的虚拟变量
    Parameters:
    -----------
    port_ret:
        pd.Series 组合收益率的Series

    Return:
        pd.Series 表示收益正或负的虚拟变量
    """

    return port_ret.gt(0).astype(float).rename('ret_sign')


def features_mul_dummy(features, dummy, broadcast_level='Trddt'):
    """
    输入一个features 与一个dummy 变量，返回一个按照index 使二者相乘后得到的变量

    Parameters:
    -----------
    features:
        pd.Series or pd.DataFrame
        需要与dummy 相乘的features 变量数据
    dummy:
        pd.Series
        与features 进行相乘的dummy
    broadcast_level:
        str or list, default 'Trddt'
        二者相乘时进行broadcast 依据的column 或index 名

    Return:
    -------
        pd.Series or pd.DataFrame
        dummy 与features 相乘后的变量。其类型与features 一致，长度应该和二者最长的相等
    """

    features_with_dummy = features.mul(dummy, axis=0, level=broadcast_level)
    if isinstance(features_with_dummy, pd.Series):
        s_name = features.name + '_' + dummy.name
        features_with_dummy.rename(s_name, inplace=True)
    elif isinstance(features_with_dummy, pd.DataFrame):
        features_with_dummy.rename(
            columns=lambda col_name: col_name + '_' + dummy.name, inplace=True)
    return features_with_dummy


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
                             '_t_{}'.format(index)] = shifted_serie

    return dframe_added_leading


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
    return std_dframe['rolling_std_log']


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
    delta_std_col = [
        col for col in std_dframe if col.startswith('delta_std_t')
    ]
    return std_dframe[delta_std_col]


def get_delta_std_forward_interval(file='data/processed/std_features.pickle'):
    """
    读取在整个forward 区间中（比如五日内）的std 差值数据

    Return:
    -------
        pd.Series
    """
    std_dframe = pd.read_pickle(file)
    delta_std_full = std_dframe['delta_std_full']
    return delta_std_full


def get_turnover_features(file='data/processed/turnover_features.pickle'):
    """
    读取组合的turnover 数据

    Returns:
    --------
        pd.Series
    """
    dframe = pd.read_pickle(file)
    return dframe


def get_amihud_features(file='data/processed/amihud_features.pickle'):
    """
    读取保存的amihud 指标

    Return:
    -------
        pd.Series
    """
    return pd.read_pickle(file)


def get_ret_sign(file='data/processed/ret_sign.pickle'):
    return pd.read_pickle(file)


def get_targets(file='data/processed/targets.pickle'):
    """
    从保存的文件中读取OLS 回归所用的targets，即超额收益率计算的反转组合收益
    Parameters:
    -----------
    file:
        str(path to the targets file)

    Results:
    --------
    pandas.Series:
        读取的targets 数据框
    """

    dframe = pd.read_pickle(file)
    return dframe
