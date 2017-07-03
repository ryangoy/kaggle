import numpy as np
import pandas as pd
from os.path import join
import sys

DS_DIR = '/home/ryan/cs/datasets/srhm/'

# important macro columns 
# https://www.kaggle.com/robertoruiz/dealing-with-multicollinearity
MACRO_COLUMNS = ['timestamp', 'balance_trade', 'balance_trade_growth', 'eurrub',
                 'average_provision_of_build_contract', 'micex_rgbi_tr',
                 'micex_cbi_tr', 'deposits_rate', 'mortgage_value',
                 'mortgage_rate', 'income_per_cap', 'rent_price_4+room_bus',
                 'museum_visitis_per_100_cap', 'apartment_build']

def import_clean():
    train = pd.read_csv(join(DS_DIR, 'train.csv'), parse_dates=['timestamp'])
    test = pd.read_csv(join(DS_DIR, 'test.csv'), parse_dates=['timestamp'])
    macro = pd.read_csv(join(DS_DIR, 'macro.csv'), parse_dates=['timestamp'])
    # macro = macro.set_index('timestamp')  

    macro = macro[MACRO_COLUMNS]
    error_fixes = pd.read_excel(join(DS_DIR, 'error_fixes.xlsx'))
    error_fixes = error_fixes.drop_duplicates('id').set_index('id')

    train, test = clean_data(train, test)

    train.update(error_fixes)
    test.update(error_fixes)

    train_lat_lon = pd.read_csv(join(DS_DIR, 'train_lat_lon.csv'))
    test_lat_lon = pd.read_csv(join(DS_DIR, 'test_lat_lon.csv'))
    train = train.merge(train_lat_lon, on='id')
    test = test.merge(test_lat_lon, on='id')
    
    train = train.merge(macro, on='timestamp', how='left')
    test = test.merge(macro, on='timestamp', how='left')

    return train, test


def clean_data(train, test):
    bad_index = train[train.life_sq > train.full_sq].index
    train.loc[bad_index, "life_sq"] = np.NaN
    equal_index = [601,1896,2791]
    test.loc[equal_index, "life_sq"] = test.loc[equal_index, "full_sq"]
    bad_index = test[test.life_sq > test.full_sq].index
    test.loc[bad_index, "life_sq"] = np.NaN
    bad_index = train[train.life_sq < 5].index
    train.loc[bad_index, "life_sq"] = np.NaN
    bad_index = test[test.life_sq < 5].index
    test.loc[bad_index, "life_sq"] = np.NaN
    bad_index = train[train.full_sq < 5].index
    train.loc[bad_index, "full_sq"] = np.NaN
    bad_index = test[test.full_sq < 5].index
    test.loc[bad_index, "full_sq"] = np.NaN
    kitch_is_build_year = [13117]
    train.loc[kitch_is_build_year, "build_year"] = train.loc[kitch_is_build_year, "kitch_sq"]
    bad_index = train[train.kitch_sq >= train.life_sq].index
    train.loc[bad_index, "kitch_sq"] = np.NaN
    bad_index = test[test.kitch_sq >= test.life_sq].index
    test.loc[bad_index, "kitch_sq"] = np.NaN
    bad_index = train[(train.kitch_sq == 0).values + (train.kitch_sq == 1).values].index
    train.loc[bad_index, "kitch_sq"] = np.NaN
    bad_index = test[(test.kitch_sq == 0).values + (test.kitch_sq == 1).values].index
    test.loc[bad_index, "kitch_sq"] = np.NaN
    bad_index = train[(train.full_sq > 210) & (train.life_sq / train.full_sq < 0.3)].index
    train.loc[bad_index, "full_sq"] = np.NaN
    bad_index = test[(test.full_sq > 150) & (test.life_sq / test.full_sq < 0.3)].index
    test.loc[bad_index, "full_sq"] = np.NaN
    bad_index = train[train.life_sq > 300].index
    train.loc[bad_index, ["life_sq", "full_sq"]] = np.NaN
    bad_index = test[test.life_sq > 200].index
    test.loc[bad_index, ["life_sq", "full_sq"]] = np.NaN
    train.product_type.value_counts(normalize= True)
    test.product_type.value_counts(normalize= True)
    bad_index = train[train.build_year < 1500].index
    train.loc[bad_index, "build_year"] = np.NaN
    bad_index = test[test.build_year < 1500].index
    test.loc[bad_index, "build_year"] = np.NaN
    bad_index = train[train.num_room == 0].index
    train.loc[bad_index, "num_room"] = np.NaN
    bad_index = test[test.num_room == 0].index
    test.loc[bad_index, "num_room"] = np.NaN
    bad_index = [10076, 11621, 17764, 19390, 24007, 26713, 29172]
    train.loc[bad_index, "num_room"] = np.NaN
    bad_index = [3174, 7313]
    test.loc[bad_index, "num_room"] = np.NaN
    bad_index = train[(train.floor == 0).values * (train.max_floor == 0).values].index
    train.loc[bad_index, ["max_floor", "floor"]] = np.NaN
    bad_index = train[train.floor == 0].index
    train.loc[bad_index, "floor"] = np.NaN
    bad_index = train[train.max_floor == 0].index
    train.loc[bad_index, "max_floor"] = np.NaN
    bad_index = test[test.max_floor == 0].index
    test.loc[bad_index, "max_floor"] = np.NaN
    bad_index = train[train.floor > train.max_floor].index
    train.loc[bad_index, "max_floor"] = np.NaN
    bad_index = test[test.floor > test.max_floor].index
    test.loc[bad_index, "max_floor"] = np.NaN
    train.floor.describe(percentiles= [0.9999])
    bad_index = [23584]
    train.loc[bad_index, "floor"] = np.NaN
    train.material.value_counts()
    test.material.value_counts()
    train.state.value_counts()
    bad_index = train[train.state == 33].index
    train.loc[bad_index, "state"] = np.NaN

    
    # brings error down a lot by removing extreme price per sqm
    train.loc[train.full_sq == 0, 'full_sq'] = 50
    train = train[train.price_doc/train.full_sq <= 600000]
    train = train[train.price_doc/train.full_sq >= 10000]
    
    return train, test
