"""explore for ols regression"""
# %%
import pandas as pd
from src.features import process_data_api as proda
from statsmodels.regression import linear_model

# %%
rm_features: pd.DataFrame = proda.get_rm_features()
target_df: pd.DataFrame = proda.get_targets()

# %%
# 某一组反转组合的OLS 回归测试
# 以'Small', 'Lo-Hi' 为Y，市场超额收益率为X
# target = target_df.xs(('Small', 'Lo-Hi'), level=[1, 2])

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
features = linear_model.add_constant(rm_features)

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
# 取出回归后的alpha 值
alpha_series = fit_on_rm.apply(lambda fit: fit.params['const'])
# alpha_series.rename_axis(index={'rev_group': None}, inplace=True)
alpha_series.head()

# %%
# 长表转置为表格，并对index 排序
alpha_series.unstack(level='rev_group').reindex(
    ['Small', '2', '3', '4', 'Big'])
