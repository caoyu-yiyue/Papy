import pandas as pd
import click
from src.features import process_data_api as proda
from src.data import preparing_data as preda


@click.command()
@click.option('--which',
              help='the type of ols features data frame to generate',
              type=click.Choice(choices=[
                  'rm_features', 'std_features', 'turnover_features',
                  'amihud_features', 'ret_sign_features', '3_fac_features'
              ]))
@click.option(
    '--windows',
    help='Backward and forward window length to calculate some features.',
    nargs=2,
    type=int)
@click.argument('input_file', type=click.Path(exists=True, readable=True))
@click.argument('output_file', type=click.Path(writable=True))
def main(which, windows, input_file, output_file):

    # 读取使用**超额收益率** 计算的反转组合收益数据框，并取出时间index
    reverse_ret_dframe: pd.Series = proda.get_targets(input_file)
    year_index: pd.Series = proda.obtain_feature_index(reverse_ret_dframe)

    # 取出过去和未来的窗口长度
    if len(windows) != 0:
        backward, forward = windows

    if which == 'rm_features':
        # 使用市场超额收益率，错位算出未来t+1,...t+forward 期的列，作为一个features 保存
        market_ret_exc: pd.Series = proda.calcualte_market_exc_ret()
        rm_exc_features: pd.DataFrame = proda.shift_leading_gradually(
            market_ret_exc.reindex(year_index),
            col_name_prefix='rm_exc',
            leading_time=forward)
        features_df: pd.DataFrame = rm_exc_features
    elif which == 'std_features':
        # 使用波动率的差值，错位计算出未来t+1,...,t+forward 期的列，同时加上一列波动率本身的值，
        # 一列整个区间上的波动率变动的值，作为features 保存
        rolling_std_log, delta_std, delta_std_forward = proda.calculate_stds(
            forward_window=forward)
        std_features: pd.DataFrame = proda.shift_leading_gradually(
            delta_std.reindex(year_index),
            col_name_prefix='delta_std',
            leading_time=forward)
        std_features['rolling_std_log'] = rolling_std_log
        std_features['delta_std_full'] = delta_std_forward.shift(-forward)
        features_df: pd.DataFrame = std_features
    elif which == 'turnover_features':
        # 计算组合的turnover，依赖向前和向后的窗口长度
        turnover_series: pd.Series = proda.calculate_turnover(
            backward, forward)
        features_df: pd.Series = turnover_series.reindex(year_index,
                                                         level='Trddt')
    elif which == 'amihud_features':
        # 计算组合的Amihud 指标，依赖向前和向后的窗口长度
        amihud_backward, amihud_forward = proda.calculate_amihud(
            backward, forward)
        features_df = pd.concat([amihud_backward, amihud_forward], axis=1)
        features_df: pd.DataFrame = features_df.reindex(
            year_index, level='Trddt').reindex(
                ['Small', '2', '3', '4', 'Big'],
                level=1).reindex(['Lo-Hi', '2-9', '3-8', '4-7', '5-6'],
                                 level=2)

    elif which == 'ret_sign_features':
        # 计算表示组合收益正负的虚拟变量
        features_df: pd.Series = proda.calculate_ret_sign(reverse_ret_dframe)
    elif which == '3_fac_features':
        # 取出三因子
        features_df = preda.read_three_factors().reindex(year_index)
    else:
        print('Wrong type of features data frame uncaught.')

    features_df.to_pickle(output_file)


if __name__ == "__main__":
    main()
