import pytest
import pandas as pd
from src.features import process_data_api as proda
from src.models import ols_model as olm
from statsmodels.regression.linear_model import RegressionResultsWrapper


class Test_ols_process(object):
    @pytest.fixture()
    def target_for_test(self):
        targets = proda.get_targets()
        return targets

    def test_no_error_each_feature(self, target_for_test):
        for features_type in list(olm.OLSFeatures):
            targets_series: pd.Series = target_for_test
            features: pd.DataFrame = olm.select_features(
                features_type=features_type)

            # 对targets 和features 进行回归。
            # 其中，merger_on_col 为None，默认使用features 的index
            olm.ols_in_group(targets_series, features)

    def test_result_shape(self, target_for_test):
        features = olm.select_features(
            features_type=olm.OLSFeatures.delta_std_full_sign)
        ols_result_df: pd.DataFrame = olm.ols_in_group(target_for_test,
                                                       features)
        assert ols_result_df.shape == (25, )

    def test_result_type(self, target_for_test):
        features = olm.select_features(
            features_type=olm.OLSFeatures.rolling_std_log)
        ols_result_df: pd.DataFrame = olm.ols_in_group(target_for_test,
                                                       features)
        assert (ols_result_df.apply(type) == RegressionResultsWrapper).all()

    def test_ols_quick(self, target_for_test):
        result_quick = olm.ols_quick(
            features_type=olm.OLSFeatures.delta_std_full_sign)
        olm.ols_quick(features_type=olm.OLSFeatures.delta_std_full_sign,
                      targets=target_for_test)

        # 与单独指定targets 与features 时的结果对比
        features = olm.select_features(
            features_type=olm.OLSFeatures.delta_std_full_sign)
        result_origin = olm.ols_in_group(target=target_for_test,
                                         features=features)
        assert result_quick[0].params.equals(result_origin[0].params)


class Test_read_ols_result(object):
    def test_read_ols_result_success(self):
        # 选择几种FeatureType 查看读取过程中是否有错误
        olm.read_ols_results_df(olm.OLSFeatures.market_ret)
        olm.read_ols_results_df(olm.OLSFeatures.delta_std_full_sign_rm)

    def test_read_ols_result_shape(self):
        # 测试不同style 参数传入后的结果
        portrait_df: pd.DataFrame = olm.read_ols_results_df(
            ols_features_type=olm.OLSFeatures.delta_std_full_sign,
            style='portrait')
        assert portrait_df.shape == (25, )

        landscape_df: pd.DataFrame = olm.read_ols_results_df(
            ols_features_type=olm.OLSFeatures.delta_std_full_sign,
            style='landscape')
        assert landscape_df.shape == (5, 5)

        with pytest.raises(ValueError) as e_info:
            olm.read_ols_results_df(
                ols_features_type=olm.OLSFeatures.delta_std_full_sign,
                style='test')
            assert str(
                e_info.value
            ) == "style must be 'portait' or 'landscape',test is invalied."


class Test_look_result_detail(object):
    """
    测试读取ols result df 参数的函数
    """
    @pytest.fixture()
    def get_ols_results(self):
        portrait_result: pd.Series = olm.read_ols_results_df(
            olm.OLSFeatures.delta_std_full_sign, style='portrait')
        landscape_result: pd.DataFrame = olm.read_ols_results_df(
            olm.OLSFeatures.delta_std_full_sign)
        result_dict = {
            'portrait': portrait_result,
            'landscape': landscape_result
        }
        return result_dict

    def test_each_detail(self, get_ols_results):
        result_df: pd.DataFrame = get_ols_results['landscape']
        details = ['param', 'pvalue', 'pvalue_star', 't_test', 't_test_star']
        for detail in details:
            olm.look_up_ols_detail(result_df,
                                   detail=detail,
                                   column=0,
                                   t_test_str='const = 0')

    def test_detail_type(self, get_ols_results):
        # 测试detail 参数传入int 和str 会返回相同的结果
        ols_result_df: pd.DataFrame = get_ols_results['landscape']
        use_int = olm.look_up_ols_detail(ols_result_df,
                                         detail='pvalue',
                                         column=2)
        use_str = olm.look_up_ols_detail(ols_result_df,
                                         detail='pvalue',
                                         column='delta_std_full')

        assert use_int.eq(use_str).all().all()

    def test_ols_result_type(self, get_ols_results):
        # 测试传入的ols_result 为pd.Series 时与返回pd.DataFrame 结果一致
        ols_result_series: pd.Series = get_ols_results['portrait']
        ols_result_df: pd.DataFrame = get_ols_results['landscape']

        use_series = olm.look_up_ols_detail(ols_result_series,
                                            detail='t_test',
                                            t_test_str='const = 0',
                                            column=0)
        use_df = olm.look_up_ols_detail(ols_result_df,
                                        detail='t_test',
                                        t_test_str='const = 0',
                                        column=0)
        assert use_series.eq(use_df).all().all()
