"""计算反转组合的换手率情况"""

import pandas as pd
from src.data import preparing_data as preda
from src.features import reverse_port_ret as rpt
import click


def read_reverse_port_turnover_data(
        fname='data/interim/reverse_port_turnover.pickle'):
    dframe = pd.read_pickle(fname)
    return dframe


# %%
@click.command()
@click.argument('output_file', type=click.Path(writable=True))
def main(output_file):
    prepared_data: pd.DataFrame = preda.read_prepared_data()
    turnover_df: pd.DataFrame = preda.read_turnover_data()
    turnover_merged: pd.DataFrame = prepared_data.join(turnover_df,
                                                       on=['Stkcd', 'Trddt'])

    backward_window, forward_window = 60, 5

    rev_port_turnover: pd.Series = rpt.reverse_port_ret_quick(
        turnover_merged,
        backward_window=backward_window,
        forward_window=forward_window,
        backward_method=rpt._normalize_last,
        forward_method=sum,
        col_for_backward_looking='log_ret',
        col_for_forward_looking='turnOver',
        average_in='reverse_group')

    rev_port_turnover.to_pickle(output_file)


if __name__ == "__main__":
    main()
