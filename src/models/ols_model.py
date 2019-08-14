"""一些进行OLS 线性回归的函数和方法"""

import pandas as pd
from src.features import process_data_api as proda
import statsmodels.api as sm
import click
from enum import Enum


class FeatureType(Enum):
    """
    用于枚举回归feature 类型的Enum
    """
    # 原论文中的部分
    market_ret = 'mkt'
    rolling_std_log = 'std'
    delta_std = 'delta_std'
    delta_std_and_rm = 'delta_std_rm'

    # 新增加的部分
    delta_std_full = 'delta_std_full'
    amihud = 'amihud'
    turnover = 'turnover'

    # 加上sign 的部分
    std_with_sign = 'std_with_sign'
    delta_std_full_sign = 'delta_std_full_sign'
    delta_std_full_sign_rm = 'delta_std_full_sign_rm'


def select_features(features_type: FeatureType):
    """
    输入一种features 类型，返回相应类型的features
    Parameters:
    -----------
    features_type:
        FeatureType
        指定想要返回的features 类型

    Results:
    --------
    pd.Series, pd.DataFrame or tuple of them.
        返回的features，如果是多个则返回为一个tuple
    """

    if features_type == FeatureType.market_ret:
        features = proda.get_rm_features()
    elif features_type == FeatureType.rolling_std_log:
        features = proda.get_rolling_std_features()
    elif features_type == FeatureType.delta_std:
        features = proda.get_delta_std_features()
    elif features_type == FeatureType.delta_std_and_rm:
        # 指定该类型时，使用未来的市场收益率数据、合并上未来的波动率变动数据，一起返回
        rm_features: pd.DataFrame = proda.get_rm_features()
        delta_std = proda.get_delta_std_features()
        features = rm_features.merge(delta_std, on='Trddt')

    elif features_type == FeatureType.delta_std_full:
        # 返回整个未来区间内的std 变动量
        features = proda.get_delta_std_forward_interval()
    elif features_type == FeatureType.amihud:
        # 返回amihud 值
        features = proda.get_amihud_features()
    elif features_type == FeatureType.turnover:
        # 返回turnover 值
        features = proda.get_turnover_features()

    elif features_type == FeatureType.std_with_sign:
        # 返回波动率(std)、组合收益率虚拟变量、及二者交互项
        std_features = proda.get_rolling_std_features()
        ret_sign = proda.get_ret_sign()
        std_with_sign = proda.features_mul_dummy(std_features, ret_sign)
        features = (ret_sign, std_features, std_with_sign)
    elif features_type == FeatureType.delta_std_full_sign:
        # 返回整个区间中波动率(std)变动、组合收益率虚拟变量、及二者交互
        delta_std_full: pd.Series = proda.get_delta_std_forward_interval()
        ret_sign: pd.Series = proda.get_ret_sign()
        delta_full_with_sign: pd.Series = proda.features_mul_dummy(
            delta_std_full, ret_sign)
        features = (ret_sign, delta_std_full, delta_full_with_sign)
    elif features_type == FeatureType.delta_std_full_sign_rm:
        # 返回整个区间中波动率（std）变动、组合收益率正负虚拟变量、二者交互、市场过去五天收益做控制
        ret_sign = proda.get_ret_sign()
        delta_std_full = proda.get_delta_std_forward_interval()
        delta_full_with_sign = proda.features_mul_dummy(
            delta_std_full, ret_sign)
        mkt_5day = proda.get_rm_features()
        features = (ret_sign, delta_std_full, delta_full_with_sign, mkt_5day)
    else:
        print('Unknown features type passed.')

    return features


# 用于单组内的OLS 回归设定，在在每组内apply
def _each_group_ols_setting(targets: pd.DataFrame, features: pd.DataFrame):
    """
    根据一个targets 和一组features，设定一个OLS model 类
    Parameters:
    -----------
    targets:
        pd.DataFrame
        用于OLS 模型设定的targets（Y 值）

    features:
        pd.DataFrame
        用于OLS 模型设定的features（X 值）

    Results:
    --------
    statsmodels.regression.linear_models.OLS
        一个statsmodels 下的OLS 类
    """

    features = sm.add_constant(features)
    ols_model = sm.OLS(endog=targets, exog=features, missing='drop')
    return ols_model


def _each_ols_train(model: sm.OLS):
    """
    对输入的OLS model 进行拟合，返回拟合的结果。
    默认使用修正了异方差、自相关、多重共线性后的协方差矩阵，滞后阶数为5

    Parameters:
    -----------
    model:
        statsmodels.regression.linear_models.OLS
        一个statsmodels 下的OLS 类

    Results:
    --------
    statsmodels.regression.linear_models.OLSResults
        statsmodels 下的OLSResults，即OLS 拟合后的结果类
    """
    fit = model.fit(cov_type='HAC', cov_kwds={'maxlags': 5})
    return fit


# 用于将target 与features 组合后进行分组设定回归模型的函数
def ols_in_group(target: pd.Series,
                 features: list,
                 merge_on: list = None,
                 groupby_col=['cap_group', 'rev_group']):
    """
    为一个target 和一个或一组features 进行分组ols 拟合。结果返回按照groupby_col 为index 的DataFrame

    Parameter:
    ----------
    target:
        pd.Series, pd.DataFrame
        用于回归使用的target(Y 值)

    features:
        pd.DataFrame, pd.Series, list of pd.DataFrame or pd.Series
        用于回归使用的features(X 值)

    merge_on:
        str, list of str, default None
        用于合并target 和features 时所依赖的列的名字，为str 或list of str
        当为None 时，默认使用features 的index

    groupby_col:
        str, list of str, default ['cap_group', 'rev_group']
        用于分组进行OLS 回归时的组别列

    Return:
    -------
    pd.DataFrame
        分组回归后，以groupby_col 为index（或mutiIndex）为index 的DataFrame
        每一项都为statsmodels 下的OLSResult 对象

    """

    # 输入的merge_on 如果不是None，则应该与features 序列的长度相同
    assert (merge_on is None or len(merge_on) == len(features)
            ), 'Parameter \'merge_on\' must be None or as long as \'features\''

    # 判定输入的类型，统一转置为DataFrame 然后合并
    if isinstance(features, (list, tuple)):
        combine_list = list(features)
    elif isinstance(features, pd.DataFrame) or isinstance(features, pd.Series):
        combine_list = [features]
    else:
        raise TypeError(
            'features type must be DataFrame, Series or list of them.')
    combine_list.insert(0, target)

    combine_df_list = [
        pd.DataFrame(item) if not isinstance(item, pd.DataFrame) else item
        for item in combine_list
    ]

    # 依次将features 数据框合并进用于ols 回归的总数据框中，用于下一步的回归。
    df_for_ols = combine_df_list[0]
    for i, feature_df in enumerate(combine_df_list[1:]):
        # 如果传入的merge_on 为空值，那么将其设定为features 的index 的name
        if merge_on is None or merge_on[i] is None:
            print('Did not specify on which to merge Y and X, \
                using X\'s index instead.')
            perhap_index_name = [feature_df.index.name, feature_df.index.names]
            merger = [item for item in perhap_index_name
                      if item is not None][0]
        else:
            merger = merge_on[i]

        # 将targets 与features 合并为一个数据框，用于下一步分组OLS
        df_for_ols = df_for_ols.join(feature_df,
                                     how='left',
                                     on=merger,
                                     lsuffix='target')

    # 使用传入的分组参数groupby_col 对组合完的数据框分组，每组内应用前面的函数设定OLS 模型对象
    ols_setted: pd.DataFrame = df_for_ols.groupby(groupby_col).apply(
        lambda df: _each_group_ols_setting(df.iloc[:, 0], df.iloc[:, 1:]))

    # 对上面设定完的每个OLS 对象进行拟合
    ols_trained: pd.Series = ols_setted.apply(_each_ols_train)

    # reindex the Series for the ols results
    ols_series_reindexed = ols_trained.reindex(
        index=['Small', '2', '3', '4', 'Big'], level=0)

    return ols_series_reindexed


def read_ols_results_df(ols_features_type: FeatureType,
                        style: str = 'landscape'):
    """
    根据ols_features_type 指定的ols 模型features 类型，返回使用该features 拟合得到的一组OLSRsults

    Parameters:
    -----------
    ols_features_type:
        FeatureType
        指定想要读取的OLS 模型建立时使用的features 类型
    style:
        {'landscape', 'portrait'}, default 'landscape'
        指定返回DataFrame 的样式：
        'landscape': 返回横表（规模分组做index，反转组合做column）
        'portrait': 返回长表（规模和反转组合一起，作为MutiIndex）

    Results:
    --------
    pandas.DataFrame:
        相应的ols_features_type 生成的一组OLS Results 数据框
    """

    # 确定返回的模式在需要的两种之一
    if style not in {'portrait', 'landscape'}:
        msg = "style must be 'portrait' or 'landscape',{} is invalied.".format(
            style)
        raise ValueError(msg)

    # 从保存的文件中读取ols_results DataFrame
    file_path = 'models/ols_on_' + ols_features_type.value + '.pickle'
    ols_results_df: pd.DataFrame = pd.read_pickle(file_path)

    # 根据style 的要求，返回所需的表格样式
    if style == 'portrait':
        # 当为portrait 时，直接返回长表
        returned_ols_results = ols_results_df
    elif style == 'landscape':
        # 当为landscape 时，将长表变为横表并返回
        returned_ols_results = ols_results_df.unstack(level='rev_group')

    return returned_ols_results


# 查看回归结果的函数
def _star_df(pvalue):
    """
    float -> str
    输入一个p 值，返回对应的星号。用于apply 中调用。
    """
    if pvalue <= 0.01:
        return '***'
    elif pvalue <= 0.05:
        return '**'
    elif pvalue <= 0.1:
        return '*'
    else:
        return ''


def look_up_ols_detail(ols_result_df: pd.DataFrame,
                       detail,
                       column=None,
                       t_test_str=None):
    """
    返回一个OLSRsults 对象组成的DataFrame 的系数、pvalue、tvalue 等细节

    Parameters:
    -----------
    ols_result_df:
        pd.DataFrame or pd.Series
        存储OLSRsults 对象的DataFrame
    detail:
        str
        需要的结果细节的类型
        可选范围是{'param', 'pvalue', 'pvalue_star', 't_test', 't_test_star'}
    column:
        str or int
        需要返回的结果所在的列名，或它在回归结果中的index 数
    t_test_str:
        若需要返回的是t 检验相关的结果，则需要指定检验公式，以str 提供如'const = 0'

    Returns:
    --------
        pd.DataFrame
        存储所需的OLS 回归结果细节的数据框
    """

    # 检查detail 在需要的范围内
    if detail not in {
            'param', 'pvalue', 'pvalue_star', 't_test', 't_test_star'
    }:
        raise ValueError(
            "detail must be on of 'param', 'pvalue', 'pvalue_star',\
            't_test', 't_test_star'")

    # 检查detail 为t 检验时，t_test_str 必须有值
    if detail.startswith('t_test') and not isinstance(t_test_str, str):
        msg = "Must provide t_test_str when ask for t test"
        raise ValueError(msg)

    # 如果传入的结果类型是Series，则要将其变为DataFrame
    if isinstance(ols_result_df, pd.Series):
        ols_result_df: pd.DataFrame = ols_result_df.unstack()

    if detail == 'param':
        result_df = ols_result_df.applymap(
            lambda ols_result: ols_result.params[column].round(4))

    # pvalues
    elif detail == 'pvalue':
        # 返回系数p 值本值
        result_df = ols_result_df.applymap(
            lambda ols_result: ols_result.pvalues[column].round(4))
    elif detail == 'pvalue_star':
        # 返回带星号的系数p 值
        pvalue_df: pd.DataFrame = look_up_ols_detail(ols_result_df,
                                                     'pvalue',
                                                     column=column)
        star_df: pd.DataFrame = pvalue_df.applymap(_star_df)
        df_add_star: pd.DataFrame = pvalue_df.applymap(
            lambda pvalue: format(pvalue, '.4f')) + star_df
        result_df = df_add_star

    # t_test
    elif detail == 't_test':
        # 返回t 检验的t 值本值
        result_df = ols_result_df.applymap(
            lambda ols_result: ols_result.t_test(t_test_str).tvalue.item())
    elif detail == 't_test_star':
        # 返回带星号的t 值
        pvalue_df = ols_result_df.applymap(
            lambda ols_result: ols_result.t_test(t_test_str).pvalue.item())
        star_df: pd.DataFrame = pvalue_df.applymap(_star_df)
        tvalue_df: pd.DataFrame = look_up_ols_detail(ols_result_df,
                                                     detail='t_test',
                                                     column=column,
                                                     t_test_str=t_test_str)
        result_df = tvalue_df.applymap(
            lambda pvalue: format(pvalue, '.4f')) + star_df

    return result_df.rename_axis(index=None, columns=None)


def ols_quick(features_type: FeatureType, targets=None):
    """
    使用不同的features，对超额收益率计算的反转组合收益，进行OLS 回归。
    最终将不同组（规模 * 反转策略）的OLS 回归结果组成的pd.DataFrame 保存，
    存储路径为'models/ + featurestype + 'feautured_ols_results.pickle'

    **这里使用features 的index name 作为合并target 与features 的依据列
    注意在data process 的过程中，最终features 对象的index 为用于将二者合并的列**


    Parameters:
    -----------
    featurestype:
        FeaturesType
        进行OLS 模型设定时，使用的features 的类型
        如[market_ret, rolling_std_log, delta_std, delta_std_and_rm]

    Returns:
    --------
    pd.Series:
        回归过后得出的ols 结果列
    """

    # 获取到targets 和features
    if targets is None:
        # 如果没有传入targets，则读取保存的targets
        targets: pd.Series = proda.get_targets()
    features: pd.DataFrame = select_features(features_type)

    # 对targets 和features 进行回归。其中，merger_on_col 为None，默认使用features 的index
    ols_results_series: pd.Series = ols_in_group(targets, features)

    return ols_results_series


@click.command()
@click.option('--featurestype',
              type=click.Choice([e.value for e in FeatureType]),
              help='select the features\' type being to use')
@click.argument('output_file', type=click.Path(writable=True))
def main(featurestype, output_file):
    """
    调用ols_quick() 计算分组ols 的结果

    Parameters:
    -----------
    featurestype:
        str
        进行OLS 模型设定时，使用的features 的类型
        如[market_ret, rolling_std_log, delta_std, delta_std_and_rm]
    """

    # 使用ols_quick 进行回归。
    ols_results_series = ols_quick(features_type=FeatureType(featurestype))

    # 保存结果
    ols_results_series.to_pickle(output_file)


if __name__ == "__main__":
    main()
