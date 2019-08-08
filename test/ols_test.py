import pytest
import pandas as pd
from src.features import process_data_api as proda
from src.models import ols_model as olm
from statsmodels.regression.linear_model import RegressionResultsWrapper


@pytest.fixture
def target_for_test():
    targets = proda.get_targets()
    return targets


def test_no_error_each_feature(target_for_test):
    for features_type in list(olm.FeatureType):
        targets_series: pd.Series = target_for_test
        features: pd.DataFrame = olm.select_features(
            features_type=features_type)

        # 对targets 和features 进行回归。其中，merger_on_col 为None，默认使用features 的index
        olm.ols_in_group(targets_series, features)


def test_result_shape(target_for_test):
    features = olm.select_features(
        features_type=olm.FeatureType.delta_std_full_sign)
    ols_result_df: pd.DataFrame = olm.ols_in_group(target_for_test, features)
    assert ols_result_df.shape == (25, )


def test_result_type(target_for_test):
    features = olm.select_features(
        features_type=olm.FeatureType.rolling_std_log)
    ols_result_df: pd.DataFrame = olm.ols_in_group(target_for_test, features)
    assert (ols_result_df.apply(type) == RegressionResultsWrapper).all()


def test_read_ols_result_success():
    # 选择几种FeatureType 查看读取过程中是否有错误
    olm.read_ols_results_df(olm.FeatureType.market_ret)
    olm.read_ols_results_df(olm.FeatureType.delta_std_full_sign_rm)


def test_read_ols_result_shape():
    # 测试不同style 参数传入后的结果
    portrait_df: pd.DataFrame = olm.read_ols_results_df(
        ols_features_type=olm.FeatureType.delta_std_full_sign,
        style='portrait')
    assert portrait_df.shape == (25, )

    landscape_df: pd.DataFrame = olm.read_ols_results_df(
        ols_features_type=olm.FeatureType.delta_std_full_sign,
        style='landscape')
    assert landscape_df.shape == (5, 5)
