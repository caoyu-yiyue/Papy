# do some data cleaning.

# %% import modules.
import pandas as pd
import datetime

# %% read the HDF5 files
store = pd.HDFStore('data/raw_data.h5')
ret_df: pd.DataFrame = store.get('ret_df').sort_index()
annodt_df: pd.DataFrame = store.get('annodt_df')
store.close()

# %%
# 为annodt_df 增加index 用于后面的合并；同时留下原时间列、增加一列滞后的列，用于接下来的判断。
annodt_df.set_index('Annodt', drop=False, append=True, inplace=True)
annodt_df.index.names = ['Stkcd', 'Trddt']
annodt_df['Annodt_lag'] = annodt_df['Annodt'] + datetime.timedelta(days=1)

# %% delete the data on annount date and the day after it.
# 合并收益数据和宣告日数据，同时为了按代码分组。
ret_df_with_annodt: pd.DataFrame = ret_df.join(
    annodt_df, on=['Stkcd', 'Trddt'], how='outer')
ret_df_grouped = ret_df_with_annodt.groupby(level='Stkcd')

# 删除宣发日和其后一日的行。逻辑为：交易日既不在宣发日，同时也不在后一日的list 里。
ret_df_without_annodt = ret_df_grouped.apply(
    lambda df: df.query('Trddt not in Annodt and Trddt not in Annodt_lag'))

# 通过reset_index() 来去掉分组。同时删掉不需要的宣发日期列。
ret_df_final = ret_df_without_annodt.reset_index(
    level=0, drop=True).drop(
        labels=['Annodt', 'Annodt_lag'], axis=1)
