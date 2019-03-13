"""从原始数据读取，去掉宣发日及后一日的数据，加入dollar_volume 列、log_ret 列和cap_group 列"""
# %%
import pandas as pd
import datetime
import numpy as np
import modules.reverse_func as rfunc

###############################################################################
# 1. 删除宣发日以及后一日的股票
###############################################################################

# %% read the HDF5 files
store = pd.HDFStore('data/raw_data.h5')
ret_df: pd.DataFrame = store.get('ret_df').sort_index()
annodt_df: pd.DataFrame = store.get('annodt_df')
store.close()

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

#############################################################################
# 2.add some column for future use
#############################################################################
# %% add a column for dollar_volume
# 计算每组的组合权重dollar_volume
ret_df_grouped = ret_df_final.groupby('Stkcd')
ret_df_final['dollar_volume'] = ret_df_grouped.apply(
    lambda df: df['Clsprc'].shift() * df['Dnshrtrd']).reset_index(
        level=0, drop=True)

# add a column for log return
ret_df_final['log_ret'] = np.log1p(ret_df_final['Dretwd'])

# add a captain group sign
ret_df_final['cap_group'] = rfunc.creat_group_signs(
    df=ret_df_final,
    column_to_cut='Dsmvosd',
    groupby_column='Trddt',
    quntiles=5,
    labels=['Small', '2', '3', '4', 'Big'])

#############################################################################
# 3.save the prepared data.
#############################################################################
# %%
ret_df_final.to_hdf(
    'data/prepared_data.h5', key='prepared_data', format='table')
