"""
将该脚本在jupyter 环境中运行，传入processed 数据存储的位置，可以快速生成一个报告
"""
# %%
from src.models.grouped_ols import GroupedOLS, OLSFeatures
from IPython.display import display, HTML
import warnings
import sys

warnings.filterwarnings('ignore')

# %%
# 生成所有需要的对象
if __name__ == "__main__":
    proc_data_in = sys.argv[1]
else:
    proc_data_in = 'data/processed/'

# %%
all_GroupedOLS_obj = dict.fromkeys([e.name for e in OLSFeatures])

for fea_type in list(OLSFeatures):
    obj = GroupedOLS(processed_dir=proc_data_in, ols_features=fea_type)
    all_GroupedOLS_obj[fea_type.name] = obj


# %%
# 直接查看结果的函数
def t_test_single_col(for_which, col, param=False):
    """
    查看某一列的系数与t 值
    """
    for key in for_which:
        display(HTML('<h3>{}</h3>'.format(key)))
        obj: GroupedOLS = all_GroupedOLS_obj[key]
        if isinstance(col, str):
            msg_col = col
        elif isinstance(col, int):
            fea_name = obj.look_up_ols_detail(detail='params_name')
            msg_col = fea_name[col]

        if param:
            display(key + ': ' + "param for '{}'".format(msg_col))
            display(obj.look_up_ols_detail(detail='param', column=col))

        display(key + ': ' + "t_value for '{}'".format(msg_col))
        display(obj.look_up_ols_detail(detail='t_test_star', column=col))


# 快速查看多列的系数和为0 的检验t 值
def t_test_multi_col(for_which, cols):
    for key in for_which:
        display(HTML('<h3>{}</h3>'.format(key)))
        delta_std_obj: GroupedOLS = all_GroupedOLS_obj[key]
        fea_name: list = delta_std_obj.look_up_ols_detail(detail='params_name')
        cols_slc = slice(cols[0], cols[1])
        test_str = ' + '.join(fea_name[cols_slc]) + ' = 0'
        display(key + ': ' + test_str)
        display(
            delta_std_obj.look_up_ols_detail(detail='t_test_star',
                                             t_test_str=test_str))


# %% [markdown]
# ## rm and rolling_std_log
t_test_single_col(for_which=['market_ret'], col='const', param=True)

# %% [markdown]
# ## rolling_std_log
t_test_single_col(for_which=['rolling_std_log'], col='rolling_std_log')

# ## delta_std and with rm
t_test_multi_col(for_which=['delta_std', 'delta_std_and_rm'], cols=(1, 6))

# %% [markdown]
# ## delta_std_full and with rm
t_test_single_col(for_which=['delta_std_full', 'delta_std_full_rm'], col=1)

# %% [markdown]
# ## std_with_sign, delta_std_full_sign, delta_std_full_sign_rm
t_test_multi_col(for_which=[
    'std_with_sign', 'delta_std_full_sign', 'delta_std_full_sign_rm'
],
                 cols=(2, 4))

# %%
# ## std_amihud, delta_std_full_amihud
for fea in ['std_amihud', 'delta_std_full_amihud']:
    t_test_single_col(for_which=[fea], col=1)
    t_test_single_col(for_which=[fea], col=2)

# %%
# ## std_amihud_sign, delta_std_full_amihud_sign, and those with rm or 3f
for fea in [
        'std_amihud_sign', 'std_amihud_sign_rm', 'std_amihud_sign_3f',
        'delta_std_full_amihud_sign', 'delta_std_full_amihud_sign_rm',
        'delta_std_full_amihud_sign_3f'
]:
    t_test_multi_col(for_which=[fea], cols=(2, 4))
    t_test_multi_col(for_which=[fea], cols=(4, 6))
