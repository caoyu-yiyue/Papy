import pandas as pd
import click
from src.features import process_data_api as proda


@click.command()
@click.option(
    '--which',
    help='the type of ols features data frame to generate',
    type=click.Choice(choices=[
        'rm_features', 'std_features', 'turnover', 'amihud', 'ret_sign'
    ]))
@click.option(
    '--windows',
    help='Backward and forward window length to calculate some features.',
    nargs=2,
    type=int)
@click.argument('input_file', type=click.Path(exists=True, readable=True))
@click.argument('output_file', type=click.Path(writable=True))
def main(which, windows, input_file, output_file):

    assert which in (
        'rm_features', 'std_features', 'turnover', 'amihud',
        'ret_sign'), 'Invalid type {} of features data frame'.format(which)

    # 读取使用**超额收益率** 计算的反转组合收益数据框，并取出时间index
    reverse_ret_dframe: pd.Series = proda.get_targets(input_file)
    year_index: pd.Series = proda.obtain_feature_index(reverse_ret_dframe)

    if which == 'rm_features':
        # 使用市场超额收益率，错位算出未来t+1,...t+5 期的列，作为一个features 保存
        market_ret_exc: pd.Series = proda.calcualte_market_exc_ret()
        rm_exc_features: pd.DataFrame = proda.shift_leading_gradually(
            market_ret_exc.reindex(year_index), col_name_prefix='rm_exc')
        features_df: pd.DataFrame = rm_exc_features
    elif which == 'std_features':
        # 使用波动率的差值，错位计算出未来t+1,...,t+5 期的列，同时加上一列波动率本身的值，作为features 保存
        rolling_std_log, delta_std = proda.calculate_stds()
        std_features: pd.DataFrame = proda.shift_leading_gradually(
            delta_std.reindex(year_index), col_name_prefix='delta_std')
        std_features['rolling_std_log'] = rolling_std_log
        features_df: pd.DataFrame = std_features
    elif which == 'turnover':
        # 计算组合的turnover，依赖向前和向后的窗口长度
        backward, forward = windows
        turnover_series: pd.Series = proda.calculate_turnover(
            backward, forward)
        features_df: pd.Series = turnover_series.reindex(year_index,
                                                         level='Trddt')
    elif which == 'amihud':
        # 计算组合的Amihud 指标，依赖向前和向后的窗口长度
        backward, forward = windows
        amihud_series: pd.Series = proda.calculate_amihud(backward, forward)
        features_df: pd.Series = amihud_series.reindex(year_index,
                                                       level='Trddt')
    elif which == 'ret_sign':
        # 计算表示组合收益正负的虚拟变量
        features_df: pd.Series = proda.calculate_ret_sign()
    else:
        print('Wrong type of features data frame uncaught.')

    features_df.to_pickle(output_file)


if __name__ == "__main__":
    main()
