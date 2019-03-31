import pandas as pd
# import click
from src.data import preparing_data as predata
# from src.features import reverse_exc_ret as rxr


def obtain_feature_index(reverse_ret_dframe: pd.DataFrame):
    """
    返回reverse_ret_dframe 的index 中的Trddt 一列，作为下一步features 使用的index。
    Parameters:
    -----------
    reverse_ret_dframe:
        pd.DataFrame
        保存有反转组合收益的数据框，下一步作为ols 的Target 使用
    --------
    Index:
        一列与输入的数据框的Trddt 相同的index
    """

    # 从reverse_ret_dframe 的index 中，抛去不需要的level，并去重，拿出日期那一列的level
    year_idx = reverse_ret_dframe.index.droplevel('cap_group').unique()
    return year_idx


def calcualte_market_exc_ret():
    """
    计算市场超额收益率序列
    Returns:
        pd.Series
        市场超额收益率序列
    """
    ret_market: pd.Series = predata.read_rm_data()
    rf: pd.Series = predata.read_rf_data()

    rm_exc: pd.Series = ret_market - rf
    return rm_exc


def calculate_stds(std_roll_window: int = 20):
    """
    计算以过去std_roll_window 天滚动得到的市场历史波动率，以及相邻两天波动率的变动量
    Parameters:
    -----------
    std_roll_window:
        int
        滚动计算标准差时，选用的滚动窗口日期
    Returns:
    --------
    tuple:
        返回一个tuple，第0 个值为滚动标准差，第1 个值为标准差的变动值（查分值）
    """

    # 读取市场指数文件
    market_index: pd.Series = predata.read_market_index_data()

    # 计算历史滚动波动率，以及其查分值（变动情况）
    rolling_std: pd.Series = market_index.rolling(window=std_roll_window).std()
    delta_std: pd.Series = rolling_std.diff()

    return (rolling_std, delta_std)


def generate_features(columns: dict, index: pd.Index):
    """
    生成features DataFrame
    Parameters:
    -----------
    columns:
        dict
        features_df 的各个列。输入一个dict 类，key 为列名，value 为列的值。
    index:
        pd.Index
        features_df 的index。主要是用来输入target_df 的Index，使得两个表对齐
    """

    features_df = pd.DataFrame(data=columns, index=index)
    return features_df


def generate_targets(reverse_ret_dframe: pd.DataFrame):
    """
    从反转收益组合数据，生成用于OLS 分组回归的targets 数据框
    Parameters:
    -----------
    reverse_ret_dframe:
        pd.DataFrame
        反转组合收益率的时间序列表格，index 为时间和规模，columns 为不同反转策略（如Lo-Hi)

    Returns:
    --------
    pd.DataFrame
        用于OLS 回归的数据框，index 为时间、规模组、反转策略，columns 为收益率
    """

    # 按照所有的index 分组
    grouped_df = reverse_ret_dframe.groupby(reverse_ret_dframe.index.names)

    # 每组去掉index 后转置
    target_df = grouped_df.apply(lambda df: df.reset_index(drop=True).T)

    # 为target_df 数据框的index 和列命名
    target_df.rename_axis(['Trddt', 'cap_group', 'rev_group'], inplace=True)
    target_df.rename(columns={0: 'rev_ret'}, inplace=True)

    return target_df


def read_features_data(file='data/processed/features.pickle'):
    dframe = pd.read_pickle(file)
    return dframe


def read_targets_data(file='data/processed/targets.pickle'):
    dframe = pd.read_pickle(file)
    return dframe


# @click.command()
# @click.argument('features_path', type=click.Path(writable=True))
# @click.argument('targets_path', type=click.Path(writable=True))
# def main(features_path, targets_path):
#     # 读取使用**超额收益率** 计算的反转组合收益数据框，并取出时间index
#     reverse_ret_dframe: pd.Series = rxr.read_reverse_exc_data()
#     year_index: pd.Series = obtain_feature_index(reverse_ret_dframe)

#     # 计算市场超额收益率、历史波动率、历史波动率的差值
#     market_ret_exc: pd.Series = calcualte_market_exc_ret()
#     rolling_std, delta_std = calculate_stds()

#     # 使用以上三列形成features 数据框
#     features_df = generate_features(
#         columns={
#             'rm_exc': market_ret_exc,
#             'rolling_std': rolling_std,
#             'delta_std': delta_std
#         },
#         index=year_index)
#     features_df.to_pickle(features_path)
#     print('features_df done.')

#     # 使用反转组合收益数据，形成target 数据框
#     target_df = generate_targets(reverse_ret_dframe)
#     target_df.to_pickle(targets_path)
#     print('targets_df done.')
