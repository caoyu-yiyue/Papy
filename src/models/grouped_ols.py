"""分组计算OLS 的对象化接口，定义了分组OLS 的对象类"""
import pandas as pd
from src.features import process_data_api as proda
from src.features.process_data_api import ProcessedType
from statsmodels import api as sm
from enum import Enum
import re
import warnings


class OLSFeatures(Enum):
    """
    用于枚举回归feature 类型的Enum，value 的格式为main/dummy/control。
    main 将与每个dummy 相乘，control 则不会。若dummy 缺省，需要留下两个// 为标记。
    """
    # 原论文中的部分
    market_ret = 'market_ret'
    rolling_std_log = 'rolling_std_log'
    delta_std = 'delta_std'
    delta_std_and_rm = 'delta_std//market_ret'

    # 新增加的部分
    delta_std_full = 'delta_std_full'
    delta_std_full_rm = 'delta_std_full//market_ret'
    amihud_back = 'amihud_back'
    amihud_for = 'amihud_for'
    turnover = 'turnover'

    # 加上sign 的部分
    std_with_sign = 'rolling_std_log/ret_sign'
    amihudBack_with_sign = 'amihud_back/ret_sign'
    turnover_with_sign = 'turnover/ret_sign'
    delta_std_full_sign = 'delta_std_full/ret_sign'
    delta_std_full_sign_rm = 'delta_std_full/ret_sign/market_ret'

    # 同时有vol 和liquid 的部分
    std_amihudBack = 'rolling_std_log&amihud_back'
    std_amihudBack_sign = 'rolling_std_log&amihud_back/ret_sign'
    std_amihudBack_sign_rm = 'rolling_std_log&amihud_back/ret_sign/market_ret'
    std_amihudBack_sign_3f = 'rolling_std_log&amihud_back/ret_sign/three_fac'


class GroupedOLS(object):
    _ols_dframe = None

    @property
    def forward_window(self):
        if self._processed_dir == 'data/processed/':
            self._forward_window = 5
            return self._forward_window
        else:
            self._forward_window = int(
                re.compile(r'(?<=f)\d').search(self._processed_dir).group())
            return self._forward_window

    # @forward_window.setter
    # def forward_window(self, value):
    #     if not isinstance(value, int):
    #         raise TypeError('forward_window must be interger!')
    #     self._forward_window = value

    @property
    def targets(self):
        return self._targets

    # @targets.setter
    # def targets(self, value):
    #     if not isinstance(value, (pd.Series, pd.DataFrame)):
    #         raise TypeError('Targets must be pd.Series or pd.DataFrame!')
    #     self._targets = value

    @property
    def ols_features(self):
        return self._ols_features

    @property
    def ols_dframe(self):
        return self._ols_dframe

    # @ols_features.setter
    # def ols_features(self, value):
    #     if not isinstance(value, (pd.Series, pd.DataFrame)):
    #         raise TypeError('ols_features must be pd.Series or pd.DataFrame')
    #     self._ols_features = value

    # ------------------------ Methods ----------------------------- #
    def __init__(self,
                 processed_dir: str = None,
                 ols_features=None,
                 targets=None,
                 forward_window=None):
        """
        一个用于分组回归的对象。
        支持的两种调用方式：
        兼容保存过的数据：
            GroupedOLS(processed_dir, ols_features:OLSFeatures)
        使用自定义数据：
            GroupedOLS(ols_features:df or Series, targets, forward_window)
        其他方式可能会出现未知的Bug

        Parameters:
        -----------
        processed_dir:
            str
            存储process data 的路径
        ols_features:
            OLSFeatures, pd.DataFrame or pd.Series
            用于指定分组OLS 的features 类
        targets:
            pd.DataFrame or pd.Series
            用于手动指定targets
        forward_window:
            int
            持有期的长度

        Return:
        -------
        A instace of GroupedOLS.
        """
        # 判定参数processed_dir 和forward_window
        if processed_dir is not None and forward_window is None:
            # 如果指定了processed_dir，则赋值给属性
            if not isinstance(processed_dir, str):
                raise TypeError('processed_dir must be str!')
            self._processed_dir = processed_dir
            self.forward_window
        elif forward_window is not None:
            # 一旦制定了forward_window，那么直接给_forward_window 属性赋值
            try:
                self._forward_window = forward_window
            except TypeError as error:
                raise error

        # 判定ols_features 参数
        if isinstance(ols_features, OLSFeatures):
            # 如果是OLSFeatures，则使用select_features 方法
            self.select_features(ols_features)
        else:
            # 如果不是，判定类型然后赋值
            if not isinstance(ols_features,
                              (pd.Series, pd.DataFrame, list, tuple)):
                raise TypeError(
                    'ols_features must be pd.Series or pd.DataFrame!')
            self._ols_features = ols_features

        # 当targets 为空，去数据路径中获取targets。
        if targets is None:
            try:
                self._targets = self._get_proc_data(ProcessedType.targets)
            except AttributeError as er:
                raise er
        else:
            if not isinstance(targets, (pd.DataFrame, pd.Series)):
                raise TypeError('targets must be pd.DataFrame or pd.Series!')
            self._targets = targets

    def _get_proc_data(self, pro_data_type):
        return proda.get_processed(from_dir=self._processed_dir,
                                   which=pro_data_type)

    def select_features(self, features_type: OLSFeatures):
        """
        输入一种features 类型，指定属性_ols_features(为一个list)
        Parameters:
        -----------
        features_type:
            FeatureType
            指定想要返回的features 类型

        Results:
        --------
        将_ols_features 指定为features_type 对应的类型
        """

        feas_str: str = features_type.value
        # 如果feas_str 中'/' 不足两个，则为其增加数量，使其有两个
        if feas_str.count('/') < 2:
            feas_str += '/' * (2 - feas_str.count('/'))

        # 分离出每个属性，大类分为main dummy 和control，每个里面为具体的ProcessedType.name
        try:
            main_list = feas_str.split('/')[0].split('&')
            dummy_list = feas_str.split('/')[1].split('&')
            control_list = feas_str.split('/')[2].split('&')
        except Exception:
            pass

        # 三种features 分别按照str 取出来，
        # list_three_type 中0, 1, 2 位置分别存储main, dummy 和control 三个list
        list_three_type = []
        for fea_str_list in [main_list, dummy_list, control_list]:
            # 对每个存有ProcessedType name 的list 循环，按照其中的每个name 读取相应features
            feas_a_type = list(
                map(
                    lambda x: None
                    if x == '' else proda.get_processed(ProcessedType[x]),
                    fea_str_list))
            list_three_type.append(feas_a_type)

        if list_three_type[1][0] is not None:
            # 对于dummy list，如果不是None（这里以第一个项目为标准即可），则与main 相乘
            main_mul_dummy_list = [
                proda.features_mul_dummy(features=fea, dummy=dum)
                for fea in list_three_type[0] for dum in list_three_type[1]
            ]

            # 按照features 的name 排序，使输出顺序尽量可控。
            # 但只能保证main 和dummy 都是Sereis 的时候有效
            try:
                main_and_mutiplied = sorted(
                    (list_three_type[0] + main_mul_dummy_list),
                    key=lambda x: x.name)

                features = list_three_type[
                    1] + main_and_mutiplied + list_three_type[2]
            except AttributeError:
                # 如果main 或main_mul_dummy 中有DataFrame，
                # 则不执行按name 排序，直接返回如下的顺序
                features = list_three_type[1] + list_three_type[
                    0] + main_mul_dummy_list + list_three_type[2]
        else:
            # 没有dummy 时，直接组合main 和control
            features = list_three_type[0] + list_three_type[2]

        # 去掉features 里的None
        features = list(filter(None.__ne__, features))
        self._ols_features = features

    # 用于单组内的OLS 回归设定，在在每组内apply
    def __each_group_ols_setting(self, targets: pd.DataFrame,
                                 features: pd.DataFrame):
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

    def __each_ols_train(self, model: sm.OLS):
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
        fit = model.fit(cov_type='HAC',
                        cov_kwds={'maxlags': self._forward_window})
        return fit

    # 用于将target 与features 组合后进行分组设定回归模型的函数
    def ols_in_group(self,
                     merge_on: list = None,
                     groupby_col=['cap_group', 'rev_group']):
        """
        为一个target 和一个或一组features 进行分组ols 拟合。结果返回按照groupby_col 为index 的DataFrame

        Parameter:
        ----------
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
        assert (
            merge_on is None or len(merge_on) == len(self._ols_features)
        ), "Parameter 'merge_on' must be None or as long as '_ols_features'"

        # 判定输入的类型，统一转置为DataFrame 然后合并
        if isinstance(self._ols_features, (list, tuple)):
            combine_list = list(self._ols_features)
        elif isinstance(self._ols_features, pd.DataFrame) or isinstance(
                self._ols_features, pd.Series):
            combine_list = [self._ols_features]
        else:
            raise TypeError(
                'features type must be DataFrame, Series or list of them.')
        combine_list.insert(0, self._targets)

        combine_df_list = [
            pd.DataFrame(item) if not isinstance(item, pd.DataFrame) else item
            for item in combine_list
        ]

        # 依次将features 数据框合并进用于ols 回归的总数据框中，用于下一步的回归。
        df_for_ols = combine_df_list[0]
        for i, feature_df in enumerate(combine_df_list[1:]):
            # 如果传入的merge_on 为空值，那么将其设定为features 的index 的name
            if merge_on is None or merge_on[i] is None:
                warnings.warn('Did not specify on which to merge Y and X, \
                    using X\'s index instead.')
                perhap_index_name = [
                    feature_df.index.name, feature_df.index.names
                ]
                merger = [
                    item for item in perhap_index_name if item is not None
                ][0]
            else:
                merger = merge_on[i]

            # 将targets 与features 合并为一个数据框，用于下一步分组OLS
            df_for_ols = df_for_ols.join(feature_df,
                                         how='left',
                                         on=merger,
                                         lsuffix='target')

        # 使用传入的分组参数groupby_col 对组合完的数据框分组，每组内应用前面的函数设定OLS 模型对象
        ols_setted: pd.DataFrame = df_for_ols.groupby(
            groupby_col).apply(lambda df: self.__each_group_ols_setting(
                df.iloc[:, 0], df.iloc[:, 1:]))

        # 对上面设定完的每个OLS 对象进行拟合
        ols_trained: pd.Series = ols_setted.apply(self.__each_ols_train)

        # reindex the Series for the ols results
        ols_frame_reindexed = ols_trained.reindex(
            index=['Small', '2', '3', '4', 'Big'], level=0).unstack()

        self._ols_dframe = ols_frame_reindexed

        return self

    def __star_df(self, pvalue):
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

    def look_up_ols_detail(self, detail, column=None, t_test_str=None):
        """
        返回一个OLSRsults 对象组成的DataFrame 的系数、pvalue、tvalue 等细节
        如果调用它之前没有调用过ols_in_group()，则会使用默认参数调用该方法

        Parameters:
        -----------
        detail:
            str
            需要的结果细节的类型
            可选范围是{'param', 'params_name', pvalue', 'pvalue_star', 't_test',
            't_test_star'}
        column:
            str or int
            需要返回的结果所在的列名，或它在回归结果中的index 数
        t_test_str:
            若需要返回的是t 检验相关的结果，可以指定检验公式，以str 提供如'const = 0'
            在同时提供该参数与column 时，使用t_test_str 进行检验，column 值无效。

        Returns:
        --------
            pd.DataFrame
            存储所需的OLS 回归结果细节的数据框
        """

        # 如果_ols_dframe 为空，则使用默认参数调用ols_in_group() method
        if self._ols_dframe is None:
            self.ols_in_group()

        # 检查detail 在需要的范围内
        if detail not in {
                'param', 'params_name', 'pvalue', 'pvalue_star', 't_test',
                't_test_star'
        }:
            raise ValueError(
                "detail must be one of 'param','params_name', 'pvalue', 'pvalue_star',\
                't_test', 't_test_star'")

        # 检查detail 为t 检验时，column 和t_test_str 都不是None，
        # 下面使用t_test_str 进行检验，同时抛出Warning
        if detail.startswith('t_test') and None not in (column, t_test_str):
            warnings.warn("Both column and t_test_str are provided,"
                          "just using t_test_str for testing.")

        # ---------------------------- main part ---------------------------- #
        # 返回回归系数的名称
        if detail == 'params_name':
            return self._ols_dframe.iloc[0, 0].params.index.tolist()
        # 返回系数
        elif detail == 'param':
            result_df = self._ols_dframe.applymap(
                lambda ols_result: ols_result.params[column].round(4))

        # 判定是否需要计算p_value df
        if detail.startswith('pvalue') or detail.endswith('star'):
            # 如果计算pvalue 或是需要计算star，那么计算pvalue_df
            try:
                # 首先尝试用t_test_str 获取t 检验后的t 值
                pvalue_df = self._ols_dframe.applymap(
                    lambda ols_result: ols_result.t_test(t_test_str
                                                         ).pvalue.item())
            except ValueError:
                try:
                    # 尝试使用column 找到t 值
                    pvalue_df = self._ols_dframe.applymap(
                        lambda ols_result: ols_result.pvalues[column].round(4))
                except KeyError:
                    raise ValueError("Must provide a int for column or a str"
                                     "for t_test_str if is a t test.")

        # 判定是否需要计算tvalue df
        if detail.startswith('t_test'):
            # 如果是t 检验相关
            try:
                # 先尝试使用t_test_str 进行t 检验并返回t 值
                tvalue_df = self._ols_dframe.applymap(
                    lambda ols_result: ols_result.t_test(t_test_str
                                                         ).tvalue.item())
            except ValueError:
                try:
                    # 再尝试使用column 找到t 值
                    tvalue_df = self._ols_dframe.applymap(
                        lambda ols_result: ols_result.tvalues[column].round(4))
                except KeyError:
                    raise ValueError("Must provide a column or a t_test_str "
                                     "for t test.")

        # pvalues
        if detail == 'pvalue':
            # 返回系数p 值本值
            result_df = pvalue_df
        elif detail == 'pvalue_star':
            # 返回带星号的系数p 值
            star_df: pd.DataFrame = pvalue_df.applymap(self.__star_df)
            result_df: pd.DataFrame = pvalue_df.applymap(
                lambda pvalue: format(pvalue, '.4f')) + star_df

        # t_test
        elif detail == 't_test':
            # 返回t 检验的t 值本值
            result_df = tvalue_df
        elif detail == 't_test_star':
            # 返回带星号的t 值
            star_df: pd.DataFrame = pvalue_df.applymap(self.__star_df)
            result_df = tvalue_df.applymap(
                lambda pvalue: format(pvalue, '.4f')) + star_df

        return result_df.rename_axis(index=None, columns=None)
