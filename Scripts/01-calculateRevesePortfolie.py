# do some data cleaning.

# %% import modules.
import pandas as pd
import numpy as np

# %%
# window length for backward and forward
backward_window = 60
forward_window = 5

# %%
# read the prepared data
ret_df_final = pd.read_hdf('data/prepared_data.h5')

# %% add a column for normalized return
ret_df_grouped = ret_df_final.groupby(level='Stkcd')
# 计算标准化收益率
normalied_df: pd.Series = ret_df_grouped['log_ret'].rolling(
    window=backward_window).apply(
        lambda window: (window[-1] - window.mean()) / window.std())
# 标准化收益率合并到原数据
normalied_df.reset_index(level=0, drop=True, inplace=True)
ret_df_final['Norm_ret'] = normalied_df

# 在最后清理空值，防止因为两列空值重叠时多删数据。
ret_df_final.dropna(subset=['Norm_ret', 'dollar_volume'], inplace=True)

##############################################################################
# group the data.


# %% group by normalize return
ret_df_final['ret_group'] = ret_df_grouped['Norm_ret'].transform(
    lambda series: pd.qcut(
        series, q=10,
        labels=['Lo', '2', '3', '4', '5', '6', '7', '8', '9', 'Hi'])
            )


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
    lambda serie: serie.sort_index(level='Trddt', ascending=False).shift(2)
).reset_index(
    level=0, drop=True)

# 将上面倒叙过后的数据框，使用cumulative_ret 函数计算未来五天的累积收益率。最后把顺序转回来
ret_df_final['cum_ret'] = tem.groupby('Stkcd').rolling(forward_window).apply(
    cumulative_ret).reset_index(
        level=0, drop=True).sort_index(level=['Stkcd', 'Trddt'])
ret_df_final.dropna(subset=['cum_ret'], inplace=True)

# %% portfolie return for every day, cap_group and ret_group
# 计算每组按照dollar_volume 加权得到的组合收益率
ret_df_grouped = ret_df_final.groupby(['Trddt', 'cap_group', 'ret_group'])
portfolie_ret_serie = ret_df_grouped.apply(
    lambda df: np.average(df['cum_ret'], weights=df['dollar_volume']))

# %%
# 试用agg 函数。性能表现不好暂时放在这里。
# ret_df_grouped.agg({
#     'cum_ret':
#     lambda x: np.average(x, weights=ret_df_grouped['dollar_volume'])
# })

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
