"""一些进行OLS 线性回归的函数和方法"""

import pandas as pd
from src.features import process_data_api as proda
import statsmodels.api as sm
import click


def select_features(features_type: str):
    """
    输入一种features 类型，返回相应类型的features
    Parameters:
    -----------
    features_type:
        str
        指定想要返回的features 类型

    Results:
    --------
    pd.DataFrame
    返回的features 数据框
    """

    if features_type == 'market_ret':
        features = proda.get_rm_features()
    elif features_type == 'rolling_std_log':
        features = proda.get_rolling_std_features()
    elif features_type == 'delta_std':
        features = proda.get_delta_std_features()
    elif features_type == 'delta_std_and_rm':
        # 指定该类型时，使用未来的市场收益率数据、合并上未来的波动率变动数据，一起返回
        rm_features: pd.DataFrame = proda.get_rm_features()
        delta_std = proda.get_delta_std_features()
        features = rm_features.merge(delta_std, on='Trddt')
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
                 features: pd.DataFrame,
                 merge_on: str = None,
                 groupby_col=['cap_group', 'rev_group']):
    """
    为一个target 和一个或一组features 进行分组ols 拟合。结果返回按照groupby_col 为index 的DataFrame

    Parameter:
    ----------
    target:
        pd.Series
        用于回归使用的target(Y 值)

    features:
        pd.DataFrame 或pd.Series
        用于回归使用的features(X 值)

    merge_on:
        str, list of str, default None
        用于合并target 和features 时所依赖的列的名字，为str 或list of str

    groupby_col:
        str, list of str, default ['cap_group', 'rev_group']
        用于分组进行OLS 回归时的组别列

    Return:
    -------
    pd.DataFrame
        分组回归后，以groupby_col 为index（或mutiIndex）为index 的DataFrame
        每一项都为statsmodels 下的OLSResult 对象

    """

    # 判定输入的类型，统一转置为DataFrame 然后合并
    combine_list = [target, features]

    combine_df_list = [
        pd.DataFrame(item) if not isinstance(item, pd.DataFrame) else item
        for item in combine_list
    ]

    # 将targets 与features 合并为一个数据框，用于下一步分组OLS
    df_for_ols = combine_df_list[0].join(combine_df_list[1],
                                         how='left',
                                         on=merge_on,
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


def read_ols_results_df(ols_features_type: str, style: str = 'landscape'):
    """
    根据ols_features_type 指定的ols 模型features 类型，返回使用该features 拟合得到的一组OLSRsults

    Parameters:
    -----------
    ols_features_type:
        str
        指定想要读取的OLS 模型建立时使用的features 类型
        包括market_ret, rolling_std_log, delta_std, delta_std_and_rm
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

    # 断言返回的模式在需要的两种之一
    assert style in ['portrait', 'landscape'
                     ], "Ivalid data frame construction {}".format(style)

    # 从保存的文件中读取ols_results DataFrame
    file_path = 'models/ols_with_' + ols_features_type + '.pickle'
    ols_results_df: pd.DataFrame = pd.read_pickle(file_path)

    # 根据style 的要求，返回所需的表格样式
    if style == 'portrait':
        # 当为portrait 时，直接返回长表
        returned_ols_results = ols_results_df
    elif style == 'landscape':
        # 当为landscape 时，将长表变为横表并返回
        returned_ols_results = ols_results_df.unstack(level='rev_group')

    return returned_ols_results


@click.command()
@click.option('--featurestype',
              type=click.Choice([
                  'market_ret', 'rolling_std_log', 'delta_std',
                  'delta_std_and_rm'
              ]),
              help='select the features\' type being to use')
@click.argument('output_file', type=click.Path(writable=True))
def main(featurestype, output_file):
    """
    使用不同的features，对超额收益率计算的反转组合收益，进行OLS 回归。
    最终将不同组（规模 * 反转策略）的OLS 回归结果组成的pd.DataFrame 保存，
    存储路径为'models/ + featurestype + 'feautured_ols_results.pickle'

    **这里使用features 的index name 作为合并target 与features 的依据列
    注意在data process 的过程中，最终features 对象的index 为用于将二者合并的列**


    Parameters:
    -----------
    featurestype:
        str
        进行OLS 模型设定时，使用的features 的类型
        如[market_ret, rolling_std_log, delta_std, delta_std_and_rm]
    """

    # 获取到targets 和features
    targets_series: pd.Series = proda.get_targets()
    features: pd.DataFrame = select_features(features_type=featurestype)

    # 对targets 和features 进行回归。其中，merger_on_col 为用以组合二者的共同列名
    # 这里为避免在命令行处传参，定义为features 的index 名(str 或list)
    perhap_index_name = [features.index.name, features.index.names]
    merge_on_col = next(item for item in perhap_index_name if item is not None)

    ols_results_df = ols_in_group(targets_series, features, merge_on_col)

    # 保存结果
    ols_results_df.to_pickle(output_file)


if __name__ == "__main__":
    main()
