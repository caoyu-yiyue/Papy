"""计算反转组合的换手率情况"""

import pandas as pd
from src.data import preparing_data as preda
from src.features import reverse_port_ret as rpt
import click
import numba


def read_reverse_port_turnover_data(
        fname='data/interim/reverse_port_turnover.pickle'):
    dframe = pd.read_pickle(fname)
    return dframe


# 创建反转组合的标签的私有函数
@numba.jit(nopython=True)
def _creat_ret_group(row):
    if row == 'Lo' or row == 'Hi':
        return 'Lo-Hi'
    elif row == '2' or row == '9':
        return '2-9'
    elif row == '3' or row == '8':
        return '3-8'
    elif row == '4' or row == '7':
        return '4-7'
    elif row == '5' or row == '6':
        return '5-6'


# %%
@click.command()
@click.argument('output_file', type=click.Path(writable=True))
def main(output_file):
    prepared_data: pd.DataFrame = preda.read_prepared_data()
    turnover_df: pd.DataFrame = preda.read_turnover_data()
    turnover_merged: pd.DataFrame = prepared_data.join(
        turnover_df, on=['Stkcd', 'Trddt'])

    backward_window, forward_window = 60, 5

    # 前一部分和计算反转组合收益的步骤一样
    # add a column for nomolized return for each stock
    turnover_merged['norm_ret'] = rpt.backward_rolling_apply(
        df=turnover_merged,
        window=backward_window,
        method=rpt._normalize_last,
        calcu_column='log_ret')

    # add a column for cumulative return for each stosk
    turnover_merged['forward_cum_col'] = rpt.forward_rolling_apply(
        df=turnover_merged,
        window=forward_window,
        method=sum,
        calcu_column='turnOver')

    # drop na values
    turnover_merged.dropna(inplace=True)

    # add a captain group sign
    turnover_merged['cap_group'] = rpt.creat_group_signs(
        df=turnover_merged,
        column_to_cut='Dsmvosd',
        groupby_column='Trddt',
        quntiles=5,
        labels=['Small', '2', '3', '4', 'Big'])

    # add a column for return group
    turnover_merged['ret_group'] = rpt.creat_group_signs(
        df=turnover_merged,
        column_to_cut='norm_ret',
        groupby_column='Trddt',
        quntiles=10,
        labels=['Lo', '2', '3', '4', '5', '6', '7', '8', '9', 'Hi'])

    # 加入一组表示反转组合组别的代号
    turnover_merged['rev_group'] = turnover_merged['ret_group'].apply(
        _creat_ret_group)

    # 直接按照时间、规模、反转组合组别进行分组，计算加权平均的换手率
    rev_port_turnover: pd.Series = rpt.weighted_average_by_group(
        turnover_merged,
        groupby_columns=['Trddt', 'cap_group', 'rev_group'],
        calcu_column='turnOver',
        weights_column='dollar_volume')

    # 这里将rev_port_turnover 先转成宽表，以和组合反转收益率一致。
    rev_port_turnover: pd.DataFrame = rev_port_turnover.unstack(
        level='rev_group')
    # 整理一下列的顺序
    rev_port_turnover.columns = ['Lo-Hi', '2-9', '3-8', '4-7', '5-6']

    rev_port_turnover.to_pickle(output_file)


if __name__ == "__main__":
    main()
