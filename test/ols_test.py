import pytest
import pandas as pd
from src.features import process_data_api as proda
from src.features.process_data_api import ProcessedType
from src.models import ols_model as olm
from src.models.grouped_ols import OLSFeatures, GroupedOLS
from statsmodels.regression.linear_model import RegressionResultsWrapper
import re


class Test_ols_process(object):
    @pytest.fixture()
    def target_for_test(self):
        targets = proda.get_targets()
        return targets

    def test_no_error_each_feature(self, target_for_test):
        for features_type in list(OLSFeatures):
            targets_series: pd.Series = target_for_test
            features: pd.DataFrame = olm.select_features(
                features_type=features_type)

            # 对targets 和features 进行回归。
            # 其中，merger_on_col 为None，默认使用features 的index
            olm.ols_in_group(targets_series, features)

    def test_result_shape(self, target_for_test):
        features = olm.select_features(
            features_type=OLSFeatures.delta_std_full_sign)
        ols_result_df: pd.DataFrame = olm.ols_in_group(target_for_test,
                                                       features)
        assert ols_result_df.shape == (25, )

    def test_result_type(self, target_for_test):
        features = olm.select_features(
            features_type=OLSFeatures.rolling_std_log)
        ols_result_df: pd.DataFrame = olm.ols_in_group(target_for_test,
                                                       features)
        assert (ols_result_df.apply(type) == RegressionResultsWrapper).all()

    def test_ols_quick(self, target_for_test):
        result_quick = olm.ols_quick(
            features_type=OLSFeatures.delta_std_full_sign)
        olm.ols_quick(features_type=OLSFeatures.delta_std_full_sign,
                      targets=target_for_test)

        # 与单独指定targets 与features 时的结果对比
        features = olm.select_features(
            features_type=OLSFeatures.delta_std_full_sign)
        result_origin = olm.ols_in_group(target=target_for_test,
                                         features=features)
        assert result_quick[0].params.equals(result_origin[0].params)


class Test_read_ols_result(object):
    def test_read_ols_result_success(self):
        # 选择几种FeatureType 查看读取过程中是否有错误
        olm.read_ols_results_df(OLSFeatures.market_ret)
        olm.read_ols_results_df(OLSFeatures.delta_std_full_sign_rm)

    def test_read_ols_result_shape(self):
        # 测试不同style 参数传入后的结果
        portrait_df: pd.DataFrame = olm.read_ols_results_df(
            ols_features_type=OLSFeatures.delta_std_full_sign,
            style='portrait')
        assert portrait_df.shape == (25, )

        landscape_df: pd.DataFrame = olm.read_ols_results_df(
            ols_features_type=OLSFeatures.delta_std_full_sign,
            style='landscape')
        assert landscape_df.shape == (5, 5)

        with pytest.raises(ValueError) as e_info:
            olm.read_ols_results_df(
                ols_features_type=OLSFeatures.delta_std_full_sign,
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
            OLSFeatures.delta_std_full_sign, style='portrait')
        landscape_result: pd.DataFrame = olm.read_ols_results_df(
            OLSFeatures.delta_std_full_sign)
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


class Test_objective(object):
    @pytest.fixture
    def grouped_ols_obj(self):
        obj = GroupedOLS(processed_dir='data/processed/',
                         ols_features=OLSFeatures.market_ret)
        return obj

    def test_get_attr(self, grouped_ols_obj):
        """
        测试获取属性，同时其属性的值是否符合预期
        """
        obj: GroupedOLS = grouped_ols_obj
        assert obj.forward_window == 5
        assert obj.targets.eq(
            proda.get_processed(which=ProcessedType.targets,
                                from_dir='data/processed')).all()
        assert obj.ols_features[0].dropna().eq(
            proda.get_processed(
                which=ProcessedType.market_ret,
                from_dir='data/processed/').dropna()).all().all()

    def test_ols_in_group(self, grouped_ols_obj):
        obj: GroupedOLS = grouped_ols_obj
        obj.ols_in_group()

    def test_lookup_detail(self, grouped_ols_obj):
        """
        验证对象化接口算出来的detail 和使用function 计算的结果相同
        """
        obj: GroupedOLS = grouped_ols_obj
        detail_df_from_obj = obj.ols_in_group().look_up_ols_detail(
            'param', 'const')

        targets = proda.get_processed(which=ProcessedType.targets,
                                      from_dir='data/processed/')
        mkt_ret = proda.get_processed(which=ProcessedType.market_ret,
                                      from_dir='data/processed/')
        ols_series = olm.ols_in_group(targets, mkt_ret)
        detail_df_from_fun = olm.look_up_ols_detail(ols_result_df=ols_series,
                                                    detail='param',
                                                    column='const')
        detail_df_from_fun.eq(detail_df_from_obj).all().all()

    def test_pass_in_target_manually(self):
        tar = proda.get_processed(which=ProcessedType.targets,
                                  from_dir='data/processed/')
        fea = proda.get_processed(
            which=ProcessedType.market_ret,
            from_dir='data/processed/',
        )
        obj = GroupedOLS(ols_features=fea, targets=tar, forward_window=5)
        obj.ols_in_group()


class Test_obj_select_fea(object):
    def test_every_olsFeatures(self):
        """测试每一个OLSFeatures 传入时不会出现对象创建错误"""
        for ols_fea in list(OLSFeatures):
            GroupedOLS(processed_dir='data/processed/', ols_features=ols_fea)

    def test_single_feature(self):
        """测试获取一个单个的features"""
        obj: GroupedOLS = GroupedOLS(processed_dir='data/processed/',
                                     ols_features=OLSFeatures.rolling_std_log)
        std_features = proda.get_processed(which=ProcessedType.rolling_std_log)
        assert (obj.ols_features[0] == std_features).all()

    def test_mutiple_features(self):
        """测试获取两个main features"""
        obj: GroupedOLS = GroupedOLS(processed_dir='data/processed/',
                                     ols_features=OLSFeatures.std_amihud)
        std_features = proda.get_processed(which=ProcessedType.rolling_std_log)
        amihud = proda.get_processed(which=ProcessedType.amihud)
        for idx, fea in enumerate([std_features, amihud]):
            assert fea.eq(obj.ols_features[idx]).all()

    def test_with_control(self):
        """测试获取一种main 和一种control features"""
        rm_features: pd.DataFrame = proda.get_processed(
            ProcessedType.market_ret)
        delta_std = proda.get_processed(ProcessedType.delta_std)
        features = (delta_std, rm_features)
        obj: GroupedOLS = GroupedOLS(processed_dir='data/processed/',
                                     ols_features=OLSFeatures.delta_std_and_rm)
        assert (
            obj.ols_features[0].dropna() == features[0].dropna()).all().all()
        assert (
            obj.ols_features[1].dropna() == features[1].dropna()).all().all()

    def test_with_dummy(self):
        """测试获取一个main 和一个dummy 配合"""
        std_features = proda.get_processed(ProcessedType.rolling_std_log)
        ret_sign = proda.get_processed(ProcessedType.ret_sign)
        std_with_sign = proda.features_mul_dummy(std_features, ret_sign)
        features = (ret_sign, std_features, std_with_sign)

        obj: GroupedOLS = GroupedOLS(processed_dir='data/processed/',
                                     ols_features=OLSFeatures.std_with_sign)
        for idx, fea in enumerate(features):
            assert fea.eq(obj.ols_features[idx]).all()


class Test_obj_look_detail(object):
    @pytest.fixture
    def gols_obj(self):
        """创建一个GroupedOLS 对象"""
        obj: GroupedOLS = GroupedOLS(processed_dir='data/processed/',
                                     ols_features=OLSFeatures.std_with_sign)
        return obj.ols_in_group()

    @pytest.fixture
    def first_result(self, gols_obj):
        """获取第一个OLS result"""
        return gols_obj.ols_dframe.iloc[0, 0]

    def test_each_detail(self, gols_obj):
        """测试每一个details 传入后没有问题"""
        details = [
            'param', 'params_name', 'pvalue', 'pvalue_star', 't_test',
            't_test_star'
        ]
        for detail in details:
            gols_obj.look_up_ols_detail(detail=detail,
                                        column=0,
                                        t_test_str='const = 0')

    def test_param_details(self, gols_obj, first_result):
        """测试params 相关功能"""
        # params_name
        assert gols_obj.look_up_ols_detail(
            detail='params_name') == first_result.params.index.tolist()

        # param
        assert gols_obj.look_up_ols_detail(
            detail='param',
            column=1).iloc[0, 0] == first_result.params[1].round(4)

    def test_pvalue_detail(self, gols_obj, first_result):
        """测试pvalue 相关功能"""
        # pvalue
        gols_obj.look_up_ols_detail(
            'pvalue', column=1).iloc[0, 0] == first_result.pvalues[0]

        # pvalue with star
        with_star = gols_obj.look_up_ols_detail('pvalue_star',
                                                column=2).iloc[0, 0]
        float(re.sub(r'\**', '', with_star)) == first_result.pvalues.round(4)

    def test_t_test_detail(self, gols_obj, first_result):
        """测试t test 相关功能"""
        # 指定列
        detail_df = gols_obj.look_up_ols_detail(detail='t_test', column=1)
        assert detail_df.iloc[0, 0] == first_result.tvalues[1].round(4)

        # 指定t_test_str
        detail_df = gols_obj.look_up_ols_detail(
            detail='t_test', t_test_str='const + rolling_std_log = 0')
        assert detail_df.iloc[0, 0] == first_result.t_test(
            'const + rolling_std_log = 0').tvalue.item()

        # 同时指定二者，返回warning，并按照t_test_str 进行测试
        with pytest.warns(UserWarning):
            detail_df = gols_obj.look_up_ols_detail(
                detail='t_test',
                column=1,
                t_test_str='const + rolling_std_log = 0')

            assert detail_df.iloc[0, 0] == first_result.t_test(
                'const + rolling_std_log = 0').tvalue.item()

        # 测试t_test_star 功能无错
        with_star = gols_obj.look_up_ols_detail('t_test_star',
                                                column=2).iloc[0, 0]
        float(re.sub(r'\**', '', with_star)) == first_result.tvalues.round(4)

    def test_both_column_and_testStr(self, gols_obj):
        with pytest.raises(ValueError):
            gols_obj.look_up_ols_detail('pvalue')
