# do some data cleaning.

# %% import modules.
import pandas as pd
from modules import reverse_func as rfunc

# %%
# read the prepared data
ret_df_final: pd.DataFrame = pd.read_hdf('data/prepared_data.h5')

# %%
# add a column for normaliezed return
ret_df_final['norm_ret'] = rfunc.normalize_ret_rolling_past(
    df=ret_df_final, window=60)
# 清理空值，防止因为两列空值重叠时多删数据。
ret_df_final.dropna(subset=['norm_ret', 'dollar_volume'], inplace=True)

# %%
# add a column for cumulative return for each stosk
ret_df_final['cum_ret'] = rfunc.cumulative_ret_rolling_forward(
    df=ret_df_final, window=5)
ret_df_final.dropna(subset=['cum_ret'], inplace=True)

# %%
# add a captain group sign
ret_df_final['cap_group'] = rfunc.creat_group_signs(
    df=ret_df_final,
    column_to_cut='Dsmvosd',
    groupby_column='Trddt',
    quntiles=5,
    labels=['Small', '2', '3', '4', 'Big'])

# %%
# add a column for return group
ret_df_final['ret_group'] = rfunc.creat_group_signs(
    df=ret_df_final,
    column_to_cut='norm_ret',
    groupby_column='Trddt',
    quntiles=10,
    labels=['Lo', '2', '3', '4', '5', '6', '7', '8', '9', 'Hi'])

# %% portfolie return for every day, cap_group and ret_group
# 计算每组按照dollar_volume 加权得到的组合收益率
portfolie_ret_serie = rfunc.reverse_port_ret_mini(df=ret_df_final)

# %% calculate every day, in every cap_group, the reverse portfolie
# 低减高形成的反转收益率
reverse_ret_each_day: pd.DataFrame = rfunc.reverse_port_ret_all(
    serie=portfolie_ret_serie)

# 各个日期的反转收益求均值
reverse_ret_aver: pd.DataFrame = rfunc.reverse_port_ret_aver(
    df=reverse_ret_each_day)

# %%
# store = pd.HDFStore('data/reverse_portfolie.h5')
# key = 'reverse' + str(backward_window) + '_' + str(forward_window)
# store[key] = reverse_ret_aver
# # store.get('reverse20_5')
# store.close()
