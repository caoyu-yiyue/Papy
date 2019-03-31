# %%
import pandas as pd
from src.data import preparing_data as predata
from src.features import reverse_exc_ret as rxr
import statsmodels.api as sm

# %%
port_exc_ret: pd.DataFrame = rxr.read_reverse_exc_data()
year_idx = port_exc_ret.index.droplevel('cap_group').unique()

# read market return file and risk free return file
ret_market: pd.Series = predata.read_rm_data()
rf: pd.Series = predata.read_rf_data()

# %%
# 市场超额收益率
rm_exc = ret_market - rf

# =========================================
# 波动率
# =========================================
# %%
market_index = predata.read_market_index_data()

# %%
rolling_std: pd.Series = market_index.rolling(window=20).std()
delta_std = rolling_std.diff()

# %%
features_df = pd.DataFrame(
    data={
        'rm_exc': rm_exc,
        'rolling_std': rolling_std,
        'delta_std': delta_std
    },
    index=year_idx)

# =====================
# targets
# =====================
# %%
group_df = port_exc_ret.groupby(['Trddt', 'cap_group'])
# first_group: pd.DataFrame = group_df.get_group((('2007-04-04', 'Small')))
# first_group.reset_index(drop=True).T
target_df = group_df.apply(lambda df: df.reset_index(drop=True).T)

# %%
target_df.rename_axis(['Trddt', 'cap_group', 'rev_group'], inplace=True)
target_df.rename(columns={0: 'rev_ret'}, inplace=True)

# =====================
# OLS 下面的代码对象目前不对，以后需要重写一下
# =====================
# %%
# 反转组合超额收益率 ～ 市场超额收益率回归
Y = features_df['rm_exc']
X = port_exc_ret.loc[:, 'rm_exc'].xs('Small', level=1)

features_df = pd.DataFrame()
for index in range(5):
    x = X.shift(index)
    features_df['x_{}'.format(index + 1)] = x

# %%
features = features_df[['x_1', 'x_2', 'x_3', 'x_4', 'x_5']]
features = sm.add_constant(features)
model: sm.regression.linear_model.RegressionResults = sm.OLS(
    Y, features, missing='drop').fit(
        cov_type='HAC', cov_kwds={'maxlags': 5})
model.summary()

# %%
# 反转组合超额收益率 ～ 波动率
index = port_exc_ret.index
year_idx = index.droplevel(level=1).unique()
rolling_std.filter(year_idx)
