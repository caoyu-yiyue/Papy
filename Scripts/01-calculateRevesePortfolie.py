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
# add a column for return group
ret_df_final['ret_group'] = rfunc.creat_group_signs(
    df=ret_df_final,
    column_to_cut='norm_ret',
    groupby_column='Trddt',
    quntiles=10,
    labels=['Lo', '2', '3', '4', '5', '6', '7', '8', '9', 'Hi'])

# %%
# add a column for cumulative return for each stosk
ret_df_final['cum_ret'] = rfunc.cumulative_ret_rolling_forward(
    df=ret_df_final, window=5)
ret_df_final.dropna(subset=['cum_ret'], inplace=True)

# %% portfolie return for every day, cap_group and ret_group
# 计算每组按照dollar_volume 加权得到的组合收益率
portfolie_ret_serie = rfunc.reverse_port_ret_mini(df=ret_df_final)


# %% calculate every day, in every cap_group, the reverse portfolie


# “输家-赢家”，计算反转收益的函数。
def reverse_port(serie: pd.Series):
    '''
    输入不同标准收益率分组的一列序列，输家-赢家获得反转组合收益
    serie: pd.Series 一列分组后的收益值，输家在前，赢家在后。
    '''
    result = []
    for index in range(5):
        high_group = serie.iloc[9 - index]
        low_group = serie.iloc[index]
        result.append(low_group - high_group)
    return result


# 按照日期和cap_group 分组后，每组内应用以上的函数求出反转收益。
portfolie_ret_grouped = portfolie_ret_serie.groupby(['Trddt', 'cap_group'])
reverse_ret_in_serie: pd.Series = portfolie_ret_grouped.apply(reverse_port)

# 上面输出一个serie of list，把它转化为一个数据框。
reverse_ret_each_day = pd.DataFrame(
    (item for item in reverse_ret_in_serie),
    index=reverse_ret_in_serie.index,
    columns=['Lo-Hi', '2-9', '3-8', '4-7', '5-6'])
reverse_ret_aver = reverse_ret_each_day.groupby('cap_group', sort=False).mean()

# 计算均值后cap_group 失去了Categories 类型，重新规定为CategoricalIndex 并排序
reverse_ret_aver.set_index(
    pd.CategoricalIndex(
        reverse_ret_aver.index, categories=['Small', '2', '3', '4', 'Big']),
    inplace=True)
reverse_ret_aver.sort_index()

# %%
# store = pd.HDFStore('data/reverse_portfolie.h5')
# key = 'reverse' + str(backward_window) + '_' + str(forward_window)
# store[key] = reverse_ret_aver
# # store.get('reverse20_5')
# store.close()
