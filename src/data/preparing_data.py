# """从原始数据读取，去掉宣发日及后一日的数据，加入dollar_volume 列、log_ret 列和cap_group 列"""
# %%
import pandas as pd
import datetime
import numpy as np
import click


###############################################################################
# 0. 定义一些读取原始数据的函数
###############################################################################
def read_ret_df(file='data/raw/raw_data.h5'):
    """
    读取保存回报率、交易量等数据的主要数据表。直接返回该pd.DataFrame
    """
    dframe = pd.read_hdf(file, key='ret_df').sort_index()
    return dframe


def read_rf_data(file='data/raw/raw_data.h5'):
    """
    读取无风险收益率，直接返回以时间为index 的无风险收益率的pd.Series
    """
    ret_f: pd.Series = pd.read_hdf(file, key='ret_Rf').loc[:, 'ret_f']
    return ret_f.sort_index()


def read_rm_data(file='data/raw/raw_data.h5'):
    """
    读取市场收益率数据，直接返回以时间为index 的市场收益率pd.Series
    """
    ret_market: pd.Series = pd.read_hdf(file, key='ret_market').loc[:, 'ret_m']
    return ret_market.sort_index()


def read_annodt_data(file='data/raw/raw_data.h5'):
    """
    读取宣发日数据框，返回一个pd.DataFrame
    """
    annodt_df: pd.DataFrame = pd.read_hdf(file, key='annodt_df')
    return annodt_df


def read_turnover_data(file='data/raw/raw_data.h5'):
    """
    读取换手率数据，返回一个pd.DataFrame
    """
    turnover_df: pd.Series = pd.read_hdf(file, key='turnOver_df')
    return turnover_df


def read_market_index_data(file='data/raw/raw_data.h5'):
    """
    读取市场指数数据，返回一个以时间为index 的pd.Series
    """

    market_index: pd.Series = pd.read_hdf(
        file, key="market_index").loc[:, 'Clsindex']
    return market_index.sort_index()


###############################################################################
# 1. 删除宣发日以及后一日的股票
###############################################################################


def _delete_annodt(ret_df: pd.DataFrame, annodt_df: pd.DataFrame):
    """
    从ret_df 中删掉annodt_df 中的宣发日和后一日的行。返回一个删除掉这些数据的pd.DataFrame
    """
    # %% 为annodt_df 增加index 用于后面的合并；同时留下原时间列、增加一列滞后的列，用于接下来的判断。
    annodt_df.set_index('Annodt', drop=False, append=True, inplace=True)
    annodt_df.index.names = ['Stkcd', 'Trddt']
    annodt_df['Annodt_lag'] = annodt_df['Annodt'] + datetime.timedelta(days=1)

    # delete the data on annount date and the day after it.
    # 合并收益数据和宣告日数据，同时为了按代码分组。
    ret_df_with_annodt: pd.DataFrame = ret_df.join(
        annodt_df, on=['Stkcd', 'Trddt'], how='outer')
    ret_df_grouped = ret_df_with_annodt.groupby(level='Stkcd')
    # 删除宣发日和其后一日的行。逻辑为：交易日既不在宣发日，同时也不在后一日的list 里。
    ret_df_without_annodt = ret_df_grouped.apply(
        lambda df: df.query('Trddt not in Annodt and Trddt not in Annodt_lag'))
    # 通过reset_index() 来去掉分组。同时删掉不需要的宣发日期列。
    ret_df_final: pd.DataFrame = ret_df_without_annodt.reset_index(
        level=0, drop=True).drop(
            labels=['Annodt', 'Annodt_lag'], axis=1)

    return ret_df_final


#############################################################################
# 2.add some column for future use
#############################################################################
# %% add a column for dollar_volume
def _add_dollar_volume(dframe: pd.DataFrame):
    """
    计算每组的组合权重dollar_volume，加到dframe 中
    """
    ret_df_grouped = dframe.groupby('Stkcd')
    dframe['dollar_volume'] = ret_df_grouped.apply(
        lambda df: df['Clsprc'].shift() * df['Dnshrtrd']).reset_index(
            level=0, drop=True)

    return dframe


def _add_log_ret(dframe: pd.DataFrame):
    """
    为dframe 增加一列对数收益率
    """
    dframe['log_ret'] = np.log1p(dframe['Dretwd'])
    return dframe


#############################################################################
# 3. 读取准备完全的数据
#############################################################################
def read_prepared_data(file='data/interim/prepared_data.pickle'):
    """
    读取data prepare 之后的数据
    """
    dframe = pd.read_pickle(file)
    return dframe


#############################################################################
# 3.save the prepared data.
#############################################################################


@click.command()
@click.argument('input_file', type=click.Path(exists=True, readable=True))
@click.argument('output_file', type=click.Path(writable=True))
def main(input_file, output_file):
    # read the return file and the annount date file
    ret_df = read_ret_df(input_file)
    annodt_df: pd.DataFrame = read_annodt_data(input_file)

    # delete the annount date and the day after it,
    # add dollar volume and log return
    ret_df_final: pd.DataFrame = _delete_annodt(
        ret_df=ret_df,
        annodt_df=annodt_df).pipe(_add_dollar_volume).pipe(_add_log_ret)

    # save the file to output path
    ret_df_final.to_pickle(output_file)


if __name__ == "__main__":
    main()
