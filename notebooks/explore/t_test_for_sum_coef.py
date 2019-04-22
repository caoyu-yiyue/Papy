# %%
from src.models import ols_model as olsm
from src.features import process_data_api as proda
import pandas as pd

# %%
delta_std_results = olsm.read_ols_results_df(ols_features_type='delta_std')
pvalue_auto = delta_std_results.applymap(
    lambda ols_result: ols_result.t_test('delta_std_t_1 + delta_std_t_2 + delta_std_t_3 + delta_std_t_4\
     + delta_std_t_5 = 0').pvalue.item())


# %%
# 手撕
# 使用后面四列的值减掉第一列的值，然后五列一起与targets 进行回归。
# 是普通的系数和检验的做法。
delta_std_features: pd.DataFrame = proda.get_delta_std_features()
delta_std_features.head()

t_1 = delta_std_features['delta_std_t_1']

diff_features = delta_std_features.subtract(t_1, axis=0)
diff_features['delta_std_t_1'] = t_1

# %%
# ols
targets: pd.DataFrame = proda.get_targets()
targets_grouped = targets.groupby(['cap_group', 'rev_group'])
models_setted: pd.Series = targets_grouped.agg(
    olsm.ols_setting, features=diff_features)
models_trained: pd.Series = models_setted.apply(olsm.ols_train)

# %%
# 整理格式
ols_on_diff: pd.DataFrame = models_trained.unstack(level='rev_group')
pvalue_manually: pd.DataFrame = ols_on_diff.applymap(
    lambda ols_result: ols_result.pvalues['delta_std_t_1'])
pvalue_manually = pvalue_manually.reindex(['Small', '2', '3', '4', 'Big'])

# %%
# 两个计算结果是否相同
pvalue_auto.round(6).eq(pvalue_manually.round(6)).all().all()
