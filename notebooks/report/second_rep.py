"""使用第二次的一些表格做的总结"""
# %%
import pandas as pd
from src.models import ols_model as olm
from src.features import reverse_port_ret as rpt
from src.features import process_data_api as proda
from IPython.display import display
import warnings

# %%
# 设置jupyter 的显示
warnings.filterwarnings('ignore')

# %% [markdown]
# # 论文结果报告
# ## 检验原论文中的结果
# ### 描述性统计

# 首先按照原论文中的方法，使用过去60 天作为排序期，并以未来5 天作为持有期组成反转组合。反转组合的平均收益与标准差如下：

# %%
# 反转组合的平均收益率
rev_ret: pd.Series = rpt.read_reverse_port_ret_data()
aver_ret: pd.DataFrame = rpt.reverse_port_ret_aver(rev_ret.unstack())
display(aver_ret.rename_axis(index=None, columns=None))

# 反转组合收益的标准差
std_ret: pd.DataFrame = rev_ret.groupby(['cap_group',
                                         'rev_group']).std().unstack()
std_ret.reindex(['Small', '2', '3', '4', 'Big']).rename_axis(index=None,
                                                             columns=None)

# %% [markdown]
# ### 使用五天的市场收益率回归
# 使用持有期五天内的市场超额收益率对反转组合收益率进行回归，回归的截距项与其t 值如下。
# 可见，仅使用capm 模型不能解释大部分的反转组合定价误差。

# %%
target = proda.get_targets()
on_mkt: pd.Series = olm.ols_quick(features_type=olm.OLSFeatures.market_ret)
display(olm.look_up_ols_detail(on_mkt, detail='param', column='const'))
display(
    olm.look_up_ols_detail(on_mkt,
                           detail='t_test_star',
                           t_test_str='const = 0'))

# %% [markdown]
# ### 使用过去20 天的历史波动率回归
# 使用过去一个月内交易日（20 天）的历史波动率进行回归，观察反转收益是否与市场的波动有关，五天vol 的系数和与其t 值如下
# 从表格中可以看出，大部分的系数在这里并不显著。

# %%
on_std: pd.Series = olm.ols_quick(
    features_type=olm.OLSFeatures.rolling_std_log)

display(olm.look_up_ols_detail(on_std, 'param', column='rolling_std_log'))
display(
    olm.look_up_ols_detail(on_std,
                           't_test_star',
                           t_test_str='rolling_std_log = 0'))

# %% [markdown]
# ### 使用持有期内的波动率变动进行回归
# #### 使用每天的波动率变动
# 分别使用持有期内每天的波动率变动以及整个持有期内波动率的变动作为x 值进行回归，并观察回归系数与显著性。
# 首先使用每天的波动率变动值，panel 1 仅使用变动本身，panel 2 加入了五天的市场收益率进行控制。
# 可以看到，系数并无显著表现。

# %%
# 使用每天的std 变动
on_delta_std: pd.DataFrame = olm.ols_quick(olm.OLSFeatures.delta_std)
test_str = '+'.join(list(on_delta_std[0].params.index[1:])) + ' = 0'
display(
    olm.look_up_ols_detail(on_delta_std, 't_test_star', t_test_str=test_str))

# 使用每天的std 变动加上每天的市场超额收益进行控制
on_delta_std_rm: pd.DataFrame = olm.ols_quick(olm.OLSFeatures.delta_std_and_rm)
display(
    olm.look_up_ols_detail(on_delta_std_rm, 't_test_star',
                           t_test_str=test_str))

# %% [markdown]
# #### 使用五天的整体波动率变动
# 下面直接使用五天的波动率变动进行回归，观察其系数和t 值。可见系数依然并无明显的显著性特征

on_delta_full: pd.DataFrame = olm.ols_quick(olm.OLSFeatures.delta_std_full)
display(
    olm.look_up_ols_detail(on_delta_full,
                           't_test_star',
                           t_test_str='delta_std_full = 0'))
on_delta_full_rm = olm.ols_quick(olm.OLSFeatures.delta_std_full_rm)

# 加入市场进行控制
display(
    olm.look_up_ols_detail(on_delta_full_rm,
                           't_test_star',
                           t_test_str='delta_std_full=0'))
