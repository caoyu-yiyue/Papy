import pytest
import pandas as pd
import numpy as np
from src.features import process_data_api as proda
from src.data import preparing_data as preda

targets: pd.Series = pd.read_pickle('data/processed/targets.pickle')


def test_no_error_get_processed():
    """
    测试proda.get_processed() 函数针对每种ProcessedType 有无错误
    """
    for ftype in list(proda.ProcessedType):
        proda.get_processed(from_dir='data/processed/', which=ftype)


class Test_std_features(object):
    @pytest.fixture()
    def get_std_features(self):
        mkt_index: pd.Series = preda.read_market_index_data()
        std_rolling_log, delta_std, delta_std_full = proda.calculate_stds(
            std_roll_window=20, forward_window=5)
        result_dict = {
            'mkt_index': mkt_index,
            'std': std_rolling_log,
            'delta_std': delta_std,
            'delta_std_full': delta_std_full
        }
        return result_dict

    def test_std_rolling(self, get_std_features):
        '''测试滚动标准差是否符合规定'''
        mkt_index: pd.Series = get_std_features['mkt_index']
        std_rolling_log: pd.Series = get_std_features['std']
        assert np.log(
            mkt_index.iloc[0:20].std()) == std_rolling_log.dropna()[0]

    def test_delta_std_day1(self, get_std_features):
        """测试每日间的标准差的差值是否符合手动计算结果"""
        delta_std_day1: pd.Series = get_std_features['delta_std']
        mkt_index: pd.Series = get_std_features['mkt_index']

        # 因为滚动窗口计算，导致delta_std_day1 前面的（0-19）是空值，第20 个才是第一个有数字的
        assert mkt_index.iloc[1:21].std() - mkt_index.iloc[0:20].std(
        ) == delta_std_day1[20]

    def test_delta_std_full(self, get_std_features):
        """测试整个区间内的标准差差值是否符合手动计算的结果"""
        mkt_index: pd.Series = get_std_features['mkt_index']
        delta_std_full: pd.Series = get_std_features['delta_std_full']

        assert mkt_index.iloc[5:25].std() - mkt_index.iloc[0:20].std(
        ) == pytest.approx(delta_std_full[24])

    def test_get_std(self):
        proda.get_delta_std_features()
        proda.get_delta_std_features()
        proda.get_delta_std_forward_interval()


class Test_ret_sign(object):
    def test_ret_sign(self):
        ret_sign: pd.Series = proda.calculate_ret_sign(targets)
        assert (ret_sign[0:5].to_numpy() == [1.0, 0.0, 1.0, 0.0, 1.0]).all()

        assert (ret_sign.index == targets.index).all()

    '''
    对数据乘dummy 的结果进行测试
    依次测试结果数据的长度 == features 和dummy 中的最大值，
    结果数据的类型与features 数据的类型相同
    所有dummy 中为0 的项（index），与结果数据中index 完全相同
    '''
    def test_feature_mul_dummy_simple_with_muti(self):
        ret_sign: pd.Series = proda.get_ret_sign()
        std_features: pd.Series = proda.get_rolling_std_features()

        # 使用简单的index 的Series 乘mutipleIndex 的dummy 变量
        multipied = proda.features_mul_dummy(std_features, ret_sign)

        assert len(multipied) == max(len(ret_sign), len(std_features))
        assert isinstance(std_features, type(multipied))
        assert all(multipied[multipied == 0.0].index == ret_sign[ret_sign ==
                                                                 0.0].index)

    def test_feature_mul_dummy_simple_df_with_muti(self):
        ret_sign: pd.Series = proda.get_ret_sign()
        delta_std: pd.Series = proda.get_delta_std_features()

        # 使用简单的index 的DataFrame 乘mutipleIndex 的dummy 变量
        multiplied: pd.DataFrame = proda.features_mul_dummy(
            delta_std, ret_sign)

        assert len(multiplied) == max(len(ret_sign), len(delta_std))
        assert isinstance(delta_std, type(multiplied))

        # 这里使用相乘后变量的第一列，去掉Na 值后，与dummy 等于0 且其对应位置不是Na 的量对比Index
        multiplied_first_col = multiplied.iloc[:, 0].dropna()
        assert all(multiplied_first_col[multiplied_first_col == 0].index ==
                   ret_sign[(ret_sign == 0.0)
                            & (~multiplied_first_col.isna())].index)

    def test_feature_mul_dummy_muti_with_muti(self):
        ret_sign: pd.Series = proda.get_ret_sign()

        # 使用简单的index 变量乘mutipleIndex 的dummy 变量
        multipied = proda.features_mul_dummy(targets, ret_sign)

        assert len(multipied) == max(len(ret_sign), len(targets))
        assert isinstance(targets, type(multipied))
        assert all(multipied[multipied == 0.0].index == ret_sign[ret_sign ==
                                                                 0.0].index)

    def test_feature_mul_dummy_simple_with_simple(self):
        std_features: pd.Series = proda.get_rolling_std_features()
        std_sign: pd.Series = proda.calculate_ret_sign(std_features)

        # 使用简单的index 变量乘简单index 的dummy 变量。这里使用std 计算的一个假的sign 做模拟
        multipied = proda.features_mul_dummy(std_features, std_sign)

        assert len(multipied) == max(len(std_sign), len(std_features))
        assert isinstance(targets, type(multipied))
        assert all(multipied[multipied == 0.0].index == std_sign[std_sign ==
                                                                 0.0].index)

    def test_feature_mul_dummy_muti_with_simple(self):
        std_features: pd.Series = proda.get_rolling_std_features()

        # 使用multiIndex 变量乘简单index 的dummy 变量。这里使用std 计算的一个假的sign 做模拟
        multipied = proda.features_mul_dummy(targets, std_features)

        assert len(multipied) == max(len(std_features), len(targets))
        assert isinstance(targets, type(multipied))
        assert all(multipied[multipied == 0.0].index == std_features[
            std_features == 0.0].index)

    def test_feature_mul_dummy_name(self):
        std_features = proda.get_rolling_std_features()
        delta_std_features = proda.get_delta_std_features()
        ret_sign = proda.get_ret_sign()

        std_multiplied = proda.features_mul_dummy(std_features, ret_sign)
        delta_std_multipied: pd.DataFrame = proda.features_mul_dummy(
            delta_std_features, ret_sign)
        assert std_multiplied.name == 'rolling_std_log_ret_sign'
        assert delta_std_multipied.columns[0] == 'delta_std_t_1_ret_sign'
