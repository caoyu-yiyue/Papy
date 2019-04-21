# %% [markdown]
# # 论文结果报告
# 初始的项目报告，用于对数据结果进行初步查看。

# %%
import pandas as pd
from src.models import ols_model as olsm

# %% [markdown]
# ## 使用未来五天的市场超额收益率做回归
# 下面是使用未来五天的市场收益率进行回归时，得到的$\alpha$ 和相应的$p-value$

# %%
# alpha 值
ols_on_rm: pd.DataFrame = olsm.read_ols_results_df('market_ret')
ols_on_rm.applymap(lambda ols_result: ols_result.params['const'])

# %%
# p 值
ols_on_rm.applymap(lambda ols_result: ols_result.pvalues['const'].round(4))

# %% [markdown]
# ## 使用过去20 天得到的滚动波动率进行回归
# 下面是使用过去20 天的滚动波动率进行回归所得到的$\beta$ 和相应的$p-value$

# %%
ols_on_std: pd.DataFrame = olsm.read_ols_results_df('rolling_std_log')
ols_on_std.applymap(
    lambda ols_result: ols_result.params['rolling_std_log'] * 100)

# %%
# p 值
ols_on_std.applymap(
    lambda ols_result: ols_result.pvalues['rolling_std_log'].round(4))

# %% [markdown]
# ## 使用delta_std 进行回归
# 下面是使用delta_std 进行回归的结果

# %%
# ols on delta_std
ols_on_delta_std: pd.DataFrame = olsm.read_ols_results_df('delta_std')
ols_on_delta_std.applymap(lambda ols_result: ols_result.params[1:].sum())

# %%
# p 值
# ols_on_delta_std.applymap(
# lambda ols_result: ols_result.pvalues['const'].round(4))

# %% [markdown]
# 使用delta_std 进行回归的同时，控制住市场超额收益

# %%
ols_on_delta_std_and_rm: pd.DataFrame = olsm.read_ols_results_df(
    'delta_std_and_rm')
ols_on_delta_std_and_rm.applymap(
    lambda ols_result: ols_result.params[6:].sum())
