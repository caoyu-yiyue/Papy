import pandas as pd
import click
from src.features import reverse_exc_ret as rxr
from src.features import process_data_api as proda


@click.command()
@click.argument('input_file', type=click.Path(exists=True, readable=True))
@click.argument('rmfeatures', type=click.Path(writable=True))
@click.argument('stdfeatures', type=click.Path(writable=True))
def main(input_file, rmfeatures, stdfeatures):
    # 读取使用**超额收益率** 计算的反转组合收益数据框，并取出时间index
    reverse_ret_dframe: pd.Series = rxr.read_reverse_exc_data(input_file)
    year_index: pd.Series = proda.obtain_feature_index(reverse_ret_dframe)

    # 计算市场超额收益率、历史波动率、历史波动率的差值
    market_ret_exc: pd.Series = proda.calcualte_market_exc_ret()
    rolling_std, delta_std = proda.calculate_stds()

    # 使用市场超额收益率，错位算出未来t+1,...t+5 期的列，作为一个features 保存
    rm_exc_features = proda.shift_leading_gradually(
        market_ret_exc.reindex(year_index), col_name_prefix='rm_exc')
    rm_exc_features.to_pickle(rmfeatures)

    # 使用波动率的差值，错位计算出未来t+1,...,t+5 期的列，同时加上一列波动率本身的值，作为features 保存
    std_features = proda.shift_leading_gradually(
        delta_std.reindex(year_index), col_name_prefix='delta_std')
    std_features['rolling_std'] = rolling_std
    std_features.to_pickle(stdfeatures)


if __name__ == "__main__":
    main()
