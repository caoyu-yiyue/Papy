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
ret_df_final: pd.DataFrame = ret_df_without_annodt.reset_index(
    level=0, drop=True).drop(
        labels=['Annodt', 'Annodt_lag'], axis=1)

# %%
# add a column for normalized return
ret_df_grouped = ret_df_final.groupby(level='Stkcd')

# %% 计算标准化收益率
normalied_df: pd.Series = ret_df_grouped['Dretwd'].rolling(window=60).apply(
    lambda window: (window[59] - window.mean()) / window.std())

# %% 标准化收益率合并到原数据
normalied_df.reset_index(level=0, drop=True, inplace=True)
ret_df_final['Norm_ret'] = normalied_df
ret_df_final.dropna(subset=['Norm_ret'], inplace=True)

# %% group by cap
ret_df_grouped = ret_df_final.groupby(
    'Trddt', as_index=False, group_keys=False)
ret_df_final['cap_group'] = ret_df_grouped['Dsmvosd'].apply(
    lambda series: pd.qcut(series, q=5, labels=['Small', '2', '3', '4', 'Big'])
        ).reset_index(level=0, drop=True)

# %% group by normalize return
ret_df_final['ret_group'] = ret_df_grouped['Norm_ret'].apply(
    lambda series: pd.qcut(
        series, q=10,
        labels=['Lo', '2', '3', '4', '5', '6', '7', '8', '9', 'Hi'])
            ).reset_index(level=0, drop=True)


# %% a function for calculate the cumulative return.
def cumulative_ret(data_serie: pd.Series):
    '''
    data_serie: array like nums
    输入一列数字，计算累积收益率。如：
    输入[0.4, 0.3, 0.2]，返回（1 * 1.2 * 1.3 * 1.4 - 1）
    '''
    cumul_ret = 1
    for num in data_serie:
        cumul_ret = (1 + num) * cumul_ret
    return (cumul_ret - 1)


# cumulative_ret([0.4, 0.3, 0.2, 0.1])

# %% calculate cumulative return for each stock.
# 将时间倒序，解决下一步rolling 对象没有向前rolling 的问题。
ret_df_grouped = ret_df_final.groupby('Stkcd')
tem = ret_df_grouped['Dretwd'].apply(
    lambda serie: serie.sort_index(level='Trddt', ascending=False).shift(1)
).reset_index(
    level=0, drop=True)

# 将上面倒叙过后的数据框，使用cumulative_ret 函数计算未来五天的累积收益率。最后把顺序转回来
ret_df_final['cum_ret'] = tem.groupby('Stkcd').rolling(5).apply(
    cumulative_ret).reset_index(
        level=0, drop=True).sort_index(level=['Stkcd', 'Trddt'])
ret_df_final.dropna(subset=['cum_ret'], inplace=True)
