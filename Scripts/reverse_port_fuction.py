import pandas as pd
import numpy as np


# %%
# function for normalized return in past rolling window.
def normalize_ret_rolling_past(df: pd.DataFrame,
                               window: int,
                               ret_column: str = 'log_ret',
                               groupby_column: str = 'Stkcd'):
    """
    传入一个pandas.DataFrame 对象，对其中的一列收益率以过去的window 区间为窗口做滚动标准化。

    Parameters
    ----------
    df:
        输入的pandas.DataFrame 对象
    window:
        滚动窗口长度
    ret_column:
        需要进行标准化的收益率列
    groupby_column:
        分组依据的列。例如按股票代码分组。

    Returns
    -------
    pandas.Series
        按过去window 期标准化后的收益率。类型为一列与原pandas.DataFrame 的index 相同的Series 对象。
        每组的前(window - 1) 个位置为NaN。
    """
    normalized_ret_serie: pd.Series = df[ret_column].groupby(
        level=groupby_column).rolling(window=window).apply(
            lambda serie: (serie[-1] - serie.mean()) / serie.std())
    return normalized_ret_serie.reset_index(level=0, drop=True)


# def cumulative_ret(data_serie: pd.Series):
#     """
#     输入一列数字，计算累积收益率。如：
#     输入[0.4, 0.3, 0.2]，返回（1 * 1.2 * 1.3 * 1.4 - 1）

#     Parameters:
#     -----------
#     data_serie: array like nums
#     """
#     cumul_ret = 1
#     for num in data_serie:
#         cumul_ret = (1 + num) * cumul_ret
#     return (cumul_ret - 1)


# %%
def cumulative_ret_rolling_forward(df: pd.DataFrame,
                                   window: int,
                                   ret_column: str = 'Dretwd',
                                   groupby_column: str = 'Stkcd',
                                   shift: int = 2):
    """
    计算pd.DataFrame 对象某一列，针对长度为window 的窗口期内滚动的累积收益率。

    Parameters:
    -----------
    df:
        pd.DataFrame
        进行计算的DataFrame 对象
    window:
        int
        计算累积收益率的窗口期长度
    ret_column:
        str or list of str 'Dretwd' Default
        本期收益率列名，即用于滚动计算累计收益率的列
    groupby_column:
        str 'Stkcd' Default
        用于将数据分组的列名。默认为'Stkcd'，即股票代码
    shift:
        int 2 Default
        位移参数，确定每个时间点，从何时开始累积收益率区间。若shift=0, 从本期开始；shift=1，从下一期开始。
        默认为2，因为项目中在t 期排序，(t + 1) 期开始持有，那么(t + 2) 期的收益率是从第(t + 1) 期持有所得。

    Returns:
    --------
    pandas.Series
        滚动按未来window 计算得到的累积收益率Series。
    """
    # 由于pandas v0.24.1 还没有向前滚动的接口，这里先将数据倒置过来，然后向后apply 函数，达到目的。
    reverse_order: pd.DataFrame = df[::-1].loc[:, ret_column].shift(shift)
    applied_series: pd.Series = reverse_order.groupby(
        groupby_column, sort=False).rolling(window).apply(
            lambda serie: (serie + 1).product() - 1)
    return applied_series.reset_index(
        level=0, drop=True).sort_index(level=df.index.names)


# %%
def creat_group_signs(df: pd.Series, column_to_cut: str, groupby_column: str,
                      quntiles: int, labels: list):
    """
    输入一个pandas.DataFrame，按照groupby_column 分组后，将column_to_cut 成quntiles 组。
    最终标识为lables。

    Parameters:
    -----------
    df:
        pd.DataFrame
        进行分类标识的数据框
    column_to_cut:
        str
        准备进行分类的列名
    groupby_column:
        str
        进行分类的列名
    quntiles:
        int
        最终分为的组数
    lables:
        list
        各组的标识名。长度需要等于quntiles
    """
    grouped_df = df.loc[:, column_to_cut].groupby(groupby_column)
    return grouped_df.transform(
        lambda serie: pd.qcut(serie, q=quntiles, labels=labels))


# %%
def port_ret_mini(df: pd.DataFrame,
                  groupby_columns: list = ['Trddt', 'cap_group', 'ret_group'],
                  ret_column: str = 'cum_ret',
                  weights_column: str = 'dollar_volumn'):
    """
    输入一个数据框，计算每天、每个规模组、每个收益组的加权平均回报率。
    输出一列pd.Series，index 即为时间、规模组、收益组，值为加权平均值。

    Parameters:
    -----------
    df:
        pd.DataFrame
        将要计算的数据框本身
    groupby_columns:
        list of str, Default ['Trddt', 'cap_group', 'ret_group']
        用于分组的列名。默认为时间、规模组、收益组
    ret_columns:
        str, Default 'cum_ret'
        用于计算加权收益率的列名
    weights_column:
        str, Default 'dollar_volumn'
        计算加权平均时的权重
    """
    ret_df_grouped = df.groupby(groupby_columns)
    portfolie_ret_serie = ret_df_grouped.apply(
        lambda df: np.average(df[ret_column], weights=df[weights_column]))
    return portfolie_ret_serie


# port_ret_mini(df=ret_df_final)
