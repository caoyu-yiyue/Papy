# reading data frow the raw .csv files

# %%
import glob
import pandas as pd

data_root_folder = 'data/rawData/'
# %% search for the file list for Daily return and read to data frame.
file_list = glob.glob(data_root_folder + 'TRD_Dalyr*.csv')
for index, file in enumerate(file_list):
    temp_df = pd.read_csv(file,
                          sep='\t',
                          encoding='utf-16le',
                          parse_dates=['Trddt'],
                          index_col=['Stkcd', 'Trddt'])
    if index == 0:
        ret_df = temp_df
    else:
        ret_df = ret_df.append(temp_df)

# %% read market return data frame.
ret_market = pd.read_csv(data_root_folder + 'TRD_Dalym.csv',
                         sep='\t',
                         encoding='utf-16le',
                         header=0,
                         names=['Trddt', 'ret_m'],
                         parse_dates=['Trddt'],
                         index_col='Trddt')

# %% read risk free return data frame.
ret_Rf = pd.read_csv(data_root_folder + 'TRD_Nrrate.csv',
                     sep='\t',
                     encoding='utf-16le',
                     header=0,
                     names=['Trddt', 'ret_f'],
                     parse_dates=['Trddt'],
                     index_col='Trddt')

# %% read announce date data frame.
annodt_df = pd.read_csv(data_root_folder + 'IAR_Rept.csv',
                        sep='\t',
                        encoding='utf-16le',
                        usecols=['Stkcd', 'Annodt'],
                        parse_dates=['Annodt'])
annodt_df = annodt_df.drop_duplicates().set_index('Stkcd')

# %% read turn over data frame.
file_list = glob.glob(data_root_folder + 'STK_MKT_Dalyr*.csv')
for index, file in enumerate(file_list):
    temp_df = pd.read_csv(file,
                          sep='\t',
                          encoding='utf-16le',
                          header=0,
                          names=['Trddt', 'Stkcd', 'turnOver'],
                          parse_dates=['Trddt'],
                          index_col=['Stkcd', 'Trddt'])
    if index == 0:
        turnOver_df = temp_df
    else:
        turnOver_df = turnOver_df.append(temp_df)

# %%
# read market index data frame.
market_index = pd.read_csv(data_root_folder + 'TRD_Index.csv',
                           sep='\t',
                           encoding='utf-16le',
                           header=0,
                           usecols=['Trddt', 'Clsindex'],
                           parse_dates=['Trddt'],
                           index_col='Trddt')
market_index.sort_index(inplace=True)

# read three factors
three_factor = pd.read_csv(data_root_folder + 'STK_MKT_ThrfacDay.csv',
                           sep='\t',
                           encoding='utf_16le',
                           header=0,
                           names=['Trddt', 'risk_premium', 'SMB', 'HML'],
                           parse_dates=['Trddt'],
                           index_col='Trddt')

# %% look up data frame shape.
[x.shape for x in [ret_df, ret_market, ret_Rf, annodt_df, turnOver_df]]

# %% store the readed data frames.
store = pd.HDFStore('data/raw/raw_data.h5')
store['ret_df'] = ret_df
store['ret_market'] = ret_market
store['ret_Rf'] = ret_Rf
store['annodt_df'] = annodt_df
store['turnOver_df'] = turnOver_df
store['market_index'] = market_index
store['three_factors'] = three_factor
store.close()
