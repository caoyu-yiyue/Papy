import pandas as pd
import click
from src.features import reverse_exc_ret as rxr
from src.features import process_data_api as proda


@click.command()
@click.argument('input_file', type=click.Path(exists=True, readable=True))
@click.argument('output_file', type=click.Path(writable=True))
def main(input_file, output_file):
    reverse_ret_dframe: pd.Series = rxr.read_reverse_exc_data(input_file)

    # 使用反转组合收益数据，形成target 数据框
    target_df = proda.generate_targets(reverse_ret_dframe)
    target_df.to_pickle(output_file)
    print('targets_df done.')


if __name__ == "__main__":
    main()
