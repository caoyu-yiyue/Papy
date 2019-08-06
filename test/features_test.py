import pandas as pd
from src.features import process_data_api as proda

targets: pd.Series = pd.read_pickle('data/processed/targets.pickle')


class TestProda(object):
    def test_ret_sign(self):
        ret_sign: pd.Series = proda.calculate_ret_sign(targets)
        assert (ret_sign[0:5].to_numpy() == [1.0, 0.0, 1.0, 0.0, 1.0]).all()

        assert (ret_sign.index == targets.index).all()
