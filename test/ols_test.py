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
