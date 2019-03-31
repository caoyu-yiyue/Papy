import pandas as pd
import click
from src.features import reverse_exc_ret as rxr
from src.features import process_data_api as proda


@click.command()
@click.argument('input_file', type=click.Path(exists=True, readable=True))
@click.argument('output_file', type=click.Path(writable=True))
def main(input_file, output_file):
    # 读取使用**超额收益率** 计算的反转组合收益数据框，并取出时间index
    reverse_ret_dframe: pd.Series = rxr.read_reverse_exc_data(input_file)
    year_index: pd.Series = proda.obtain_feature_index(reverse_ret_dframe)

    # 计算市场超额收益率、历史波动率、历史波动率的差值
    market_ret_exc: pd.Series = proda.calcualte_market_exc_ret()
    rolling_std, delta_std = proda.calculate_stds()

    # 使用以上三列形成features 数据框
    features_df = proda.generate_features(
        columns={
            'rm_exc': market_ret_exc,
            'rolling_std': rolling_std,
            'delta_std': delta_std
        },
        index=year_index)
    features_df.to_pickle(output_file)
    print('features_df done.')


if __name__ == "__main__":
    main()
