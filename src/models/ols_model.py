"""一些进行OLS 线性回归的函数和方法"""

import pandas as pd
from src.features import process_data_api as proda
from statsmodels.regression import linear_model as lm
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


def ols_setting(targets: pd.DataFrame, features: pd.DataFrame):
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
    features = lm.add_constant(features)
    ols_model = lm.OLS(
        endog=targets.droplevel([1, 2]), exog=features, missing='drop')
    return ols_model


def ols_train(model: lm.OLS):
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


def read_ols_results_df(ols_features_type: str):
    """
    根据ols_features_type 指定的ols 模型features 类型，返回使用该features 拟合得到了一组OLSRsults

    Parameters:
    -----------
    ols_features_type:
        str
        指定想要读取的OLS 模型建立时使用的features 类型
        包括market_ret, rolling_std_log, delta_std, delta_std_and_rm

    Results:
    --------
    pandas.DataFrame:
        相应的ols_features_type 生成的一组OLS Results 数据框
    """

    file_path = 'models/ols_with_' + ols_features_type + '.pickle'
    return pd.read_pickle(file_path)


@click.command()
@click.option(
    '--featurestype',
    type=click.Choice(
        ['market_ret', 'rolling_std_log', 'delta_std', 'delta_std_and_rm']),
    help='select the features\' type being to use')
@click.argument('output_file', type=click.Path(writable=True))
def main(featurestype, output_file):
    """
    使用不同的features，对超额收益率计算的反转组合收益，进行OLS 回归。
    最终将不同组（规模 * 反转策略）的OLS 回归结果组成的pd.DataFrame 保存，
    存储路径为'models/ + featurestype + 'feautured_ols_results.pickle'

    Parameters:
    -----------
    featurestype:
        str
        进行OLS 模型设定时，使用的features 的类型
        如[market_ret, rolling_std_log, delta_std, delta_std_and_rm]
    """

    # 获取到targets 和features
    targets_dframe: pd.DataFrame = proda.get_targets()
    features = select_features(features_type=featurestype)

    # 对targets 进行分组后，分别与features 进行OLS 模型设定，并在接下来进行拟合
    targets_grouped = targets_dframe.groupby(['cap_group', 'rev_group'])
    models_setted: pd.DataFrame = targets_grouped.agg(
        ols_setting, features=features)
    models_trained: pd.DataFrame = models_setted.applymap(ols_train)
    models_trained.rename(columns={'rev_ret': 'ols_results'}, inplace=True)

    models_trained.to_pickle(output_file)


if __name__ == "__main__":
    main()
