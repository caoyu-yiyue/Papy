import pandas as pd
from features import reverse_port_ret as rpt
import click


def read_rf_data(file='data/raw/raw_data.h5'):
    """
    read the risk free return seires.
    Parameters:
    -----------
    file:
        str, default 'data/raw/raw_data.h5'
        the file for risk free return pd.DataFrame.
    Returns:
    --------
    pd.Series:
        a series for the risk free return with index for date.

    """

    ret_f: pd.Series = pd.read_hdf(file, key='ret_Rf').loc[:, 'ret_f']
    return ret_f


def add_exc_ret_column(dframe: pd.DataFrame,
                       rf_series: pd.Series,
                       exc_col_name: str = 'exc_ret'):
    """
    Add a excess return column for the dframe.
    Parameters:
    -----------
    dframe:
        pd.DataFrame
        the DataFrame you want to add the excess return column to.
    ret_f_series:
        pd.Series
        the risk free return series used to calculate the excess return.
    exc_col_name:
        str, default 'exc_ret'
        the default column name for the excess return.

    Retursn:
    --------
    pd.DataFrame:
        A dataframe added the risk free column.
    """

    dframe = dframe.copy()
    # exc_ret: substract the rf_series from the stock return seires.
    exc_ret: pd.Series = dframe['Dretwd'] - rf_series
    dframe[exc_col_name] = exc_ret

    return dframe


@click.command()
@click.argument('output_file', type=click.Path(writable=True))
def main(output_file):
    # read the files
    dframe_prepared = rpt.read_prepared_data()
    risk_free_series = read_rf_data()

    # add the excess return column
    dframe_added = add_exc_ret_column(
        dframe=dframe_prepared,
        rf_series=risk_free_series,
        exc_col_name='exc_ret')

    # use the excess return column to calculate reverse portfolie return.
    reverse_ret_use_exc: pd.DataFrame = rpt.reverse_port_ret_quick(
        dframe=dframe_added, col_for_forward_looking='exc_ret')

    reverse_ret_use_exc.to_pickle(output_file)


if __name__ == "__main__":
    main()
