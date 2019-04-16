"""explore for ols regression"""
# %%
import pandas as pd
from src.features import process_data_api as proda
from statsmodels.regression import linear_model

# %%
features_df: pd.DataFrame = proda.read_features_data()
target_df: pd.DataFrame = proda.read_targets_data()

# %%
# 某一组反转组合的OLS 回归测试
# 以'Small', 'Lo-Hi' 为Y，市场超额收益率为X
# target = target_df.xs(('Small', 'Lo-Hi'), level=[1, 2])

# 循环出五个市场超额收益率
rm_exc = features_df['rm_exc']
features = pd.DataFrame()
for index in range(1, 6):
    # 每一天后面需要追加一个t + 1,...,5, 所以shift 中使用负的的index 值进行错位
    x = rm_exc.shift(-index)
    features['rm_exc_t+{}'.format(index)] = x

# %%
# ols
# features = linear_model.add_constant(features)
# model: linear_model.RegressionModel = linear_model.OLS(
#     endog=target, exog=features, missing='drop')
# fit: linear_model.RegressionResults = model.fit(
#     cov_type='HAC', cov_kwds={'maxlags': 5})
# fit.summary()

# 需要的回归结果
# alpha = fit.params.loc['const']
# p_value = fit.pvalues.loc['const']

# %%
# 为target 分组，features 加入常数
target_grouped = target_df.groupby(['cap_group', 'rev_group'], as_index=False)
features = linear_model.add_constant(features)

# %%
# 对每个组进行回归模型的设定
ols_on_rm = target_grouped.apply(
    lambda dframe: linear_model.OLS(
        dframe.droplevel([1, 2]), exog=features, missing='drop'))

# %%
# 为模型做fit，得出回归中的值
fit_on_rm: pd.Series = ols_on_rm.apply(
    lambda model: model.fit(cov_type='HAC', cov_kwds={'maxlags': 5}))

# %%
# 取出回归后的alpha 值，同时去掉一个没用的index 名字
alpha_series = fit_on_rm.apply(lambda fit: fit.params['const'])
alpha_series.rename_axis(index={'rev_group': None}, inplace=True)
alpha_series.head()

# %%
# 将上面的alpha series 转置成表格
alpha_reverse = alpha_series.groupby('cap_group').apply(
    lambda df: df.droplevel(0).to_frame().T).droplevel(1)
alpha_reverse.set_axis(
    pd.CategoricalIndex(
        alpha_reverse.index, categories=['Small', '2', '3', '4', 'Big']),
    inplace=True)
alpha_reverse.sort_index()
