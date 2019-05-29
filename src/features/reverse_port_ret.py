import pandas as pd
import numpy as np
import numba
import click
from src.data import preparing_data as predata

# def read_prepared_data(file='data/interim/prepared_data.h5'):
#     dframe = pd.read_hdf(file)
#     return dframe


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

    print('Calculating the past window normalized return for column \'{}\'...'.
          format(ret_column))

    @numba.jit
    def _normalize_ret(serie):
        return (serie[-1] - serie.mean()) / serie.std()

    normalized_ret_serie: pd.Series = df.loc[:, ret_column].groupby(
        level=groupby_column, group_keys=False).rolling(window=window).apply(
            _normalize_ret, raw=True)
    return normalized_ret_serie


@numba.jit
def _cumulative_ret(ndarray: np.ndarray):
    """
    计算ndarray 的累积收益率的函数。
    """
    return (ndarray + 1).prod() - 1


def forward_rolling_apply(df: pd.DataFrame,
                          window: int,
                          method,
                          calcu_column: str = 'Dretwd',
                          groupby_column: str = 'Stkcd',
                          shift: int = 2):
    """
    计算pd.DataFrame 对象某一列，在长度为window 的每个窗口期内进行method 函数的计算。

    Parameters:
    -----------
    df:
        pd.DataFrame
        进行计算的DataFrame 对象
    window:
        int
        进行滚动的窗口期长度
    method:
        function
        每个滚动窗口中进行计算使用的函数。
    calcu_column:
        str or list of str 'Dretwd' Default
        传入method 进行计算的列名，即用于滚动计算的数值列
    groupby_column:
        str 'Stkcd' Default
        用于将数据分组的列名。默认为'Stkcd'，即股票代码
    shift:
        int 2 Default
        位移参数，确定每个时间点，从何时开始累积区间。若shift=0, 从本期开始；shift=1，从下一期开始。
        默认为2，因为项目中在t 期排序，(t + 1) 期开始持有，那么(t + 2) 期的收益率是从第(t + 1) 期持有所得。

    Returns:
    --------
    pandas.Series
        滚动按未来Window 使用method 计算得到的一列Series 结果。
    """

    print('Calculating the forward window using method ' + method.__name__ +
          ' for column \'{}\'...'.format(calcu_column))

    # 由于pandas v0.24.1 还没有向前滚动的接口，这里先将数据倒置过来，然后向后apply 函数，达到目的。
    reverse_order: pd.Series = df[::-1].loc[:, calcu_column]
    applied_series: pd.Series = reverse_order.groupby(
        groupby_column, group_keys=False,
        sort=False).shift(shift).rolling(window).apply(
            method, raw=True)
    return applied_series.sort_index(level=df.index.names)


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

    print('Creating the group signs for {}...'.format(column_to_cut))

    grouped_df = df.loc[:, column_to_cut].groupby(groupby_column)
    return grouped_df.transform(
        lambda serie: pd.qcut(serie, q=quntiles, labels=labels))


def reverse_port_ret_mini(
        df: pd.DataFrame,
        groupby_columns: list = ['Trddt', 'cap_group', 'ret_group'],
        ret_column: str = 'cum_ret',
        weights_column: str = 'dollar_volume'):
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
        str, Default 'dollar_volume'
        计算加权平均时的权重

    Returns:
    --------
    pandas.Series
        以分组变量为MutipleIndex 的一列pandas.Series
    """

    print('Calculating the weighted average return for mini group...')

    @numba.jit(nopython=True)
    def _weighted_mean(data_col, weights_col):
        return (data_col * weights_col).sum() / weights_col.sum()

    return df.groupby(groupby_columns).apply(
        lambda x: _weighted_mean(
            x[ret_column].to_numpy(), x[weights_column].to_numpy()))


def _reverse_port_one(serie: pd.Series):
    '''
    输入不同标准收益率分组的一列序列，输家-赢家获得反转组合收益

    Parameters:
    ------------
    serie:
    pandas.Series
        一列分组后的收益值，输家在前，赢家在后。（默认情况下，以时间、规模组和收益组为Index）

    Returns:
    --------
    list
        一个包含了len(serie)/ 2 个反转组合收益率的list
    '''
    result = []
    for index in range(int(len(serie) / 2)):
        high_group = serie.iloc[9 - index]
        low_group = serie.iloc[index]
        result.append(low_group - high_group)
    return result


def reverse_port_ret_all(serie: pd.Series):
    """
    传入port_ret_mini() 计算所得的细分组加权平均收益率序列，计算出所有日期的反转收益率数据框。

    Parameters:
    -----------
    serie:
        pandas.Seires
        以时间、规模、收益分组作为MutipleIndex 的pandas.Series

    Returns:
    --------
    pandas.DataFrame
        以日期、规模为MultiIndex，以反转组合标签（"Lo-Hi"）为列名的DataFrame
    """

    print('Calculating the reverse portfolie return...')

    # 按照前两组分类，然后每组计算反转收益。
    grouped_serie = serie.groupby(level=[0, 1])
    reverse_ret_in_serie: pd.Series = grouped_serie.apply(_reverse_port_one)
    # 将Series of list 转换为DataFrame
    reverse_ret_each_day = pd.DataFrame(
        (item for item in reverse_ret_in_serie),
        index=reverse_ret_in_serie.index,
        columns=['Lo-Hi', '2-9', '3-8', '4-7', '5-6'])
    return reverse_ret_each_day


def reverse_port_ret_aver(df: pd.DataFrame,
                          groupby_column: str = 'cap_group',
                          categories: list = ['Small', '2', '3', '4', 'Big']):
    """
    输入每天的反转组合收益率DataFrame，求出所有日期的反转组合平均收益率。
    Parameters:
    -----------
    df:
        pandas.DataFrame
        进行计算的数据，包含了每日的（5*5）反转组合收益率
    groupby_column:
        str
        进行分组的列名，这里默认为'cap_group'。按此分组后，每组中就是不同日期的数据了。
    categories:
        list of str
        求平均后'cap_group' 的category 属性丢失，重新赋予CategoryIndex 属性时制定顺序用。

    Returns:
    --------
    pandas.DataFrame
        返回平均过后的反转组合收益率
    """
    reverse_ret_aver = df.groupby(groupby_column).mean()
    reverse_ret_aver.set_axis(
        pd.CategoricalIndex(reverse_ret_aver.index, categories=categories),
        inplace=True)
    return reverse_ret_aver.sort_index()


def reverse_port_ret_quick(dframe: pd.DataFrame,
                           backward_window: int = 60,
                           forward_window: int = 5,
                           forward_method=_cumulative_ret,
                           col_for_backward_looking: str = 'log_ret',
                           col_for_forward_looking: str = 'Dretwd'):
    """
    将如上计算反转组合收益率的步骤组合在一起，
    快速计算一个按前backward_window 排序，持有forward_window 的反转组合收益时间序列。

    Parameters:
    ------------
    dframe:
        pd.DataFrame
        用于计算反转收益的整体数据框

    backward_window:
        int, default 60
        排序期时长，默认为60

    forward_window:
        int, default 5
        持有期时长，默认为5

    Rerurns:
        pd.DataFrame
        一个按时间和市值组别为index，反转组合标示（如Lo-Hi）为column 的数据框
    """
    # add a column for normaliezed return
    dframe['norm_ret'] = normalize_ret_rolling_past(
        df=dframe, window=backward_window, ret_column=col_for_backward_looking)

    # add a column for cumulative return for each stosk
    dframe['cum_ret'] = forward_rolling_apply(
        df=dframe,
        window=forward_window,
        method=forward_method,
        calcu_column=col_for_forward_looking)

    # drop na values
    dframe.dropna(inplace=True)

    # add a captain group sign
    dframe['cap_group'] = creat_group_signs(
        df=dframe,
        column_to_cut='Dsmvosd',
        groupby_column='Trddt',
        quntiles=5,
        labels=['Small', '2', '3', '4', 'Big'])

    # add a column for return group
    dframe['ret_group'] = creat_group_signs(
        df=dframe,
        column_to_cut='norm_ret',
        groupby_column='Trddt',
        quntiles=10,
        labels=['Lo', '2', '3', '4', '5', '6', '7', '8', '9', 'Hi'])

    # portfolie return for every day, cap_group and ret_group
    portfolie_ret_serie = reverse_port_ret_mini(df=dframe)

    # Low group substract high group to form reverse portfolie.
    reverse_ret_time_series: pd.DataFrame = reverse_port_ret_all(
        serie=portfolie_ret_serie)

    return reverse_ret_time_series


def read_reverse_port_ret_data(fname='data/interim/reverse_port_ret.pickle'):
    dframe = pd.read_pickle(fname)
    return dframe


@click.command()
@click.argument(
    'input_file', type=click.Path(exists=True, readable=True, dir_okay=True))
@click.argument('output_file', type=click.Path(writable=True, dir_okay=True))
def main(input_file, output_file):

    print('calculating reverse portfolie return for\
         backward_window = 60 and forward_window = 5')

    dframe = predata.read_prepared_data()
    reverse_ret_time_series = reverse_port_ret_quick(dframe)

    # save the reverse_ret_time_series
    reverse_ret_time_series.to_pickle(output_file)


if __name__ == "__main__":
    main()
