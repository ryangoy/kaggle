import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.ensemble import IsolationForest

SEED = 0

# TO DO per dataset:
# - write as many feature generation methods as desired
# - call the methods in generate_features

def generate_features(train, test):
    # call desired feature generation methods here
    print 'Encoding categorical features...'
    encode_categorical_features(train, test)
    
    print 'Filling duplicate rows with mean y...'
    free_test_labels = fill_duplicates_with_mean(train, test)
    print 'Generating duplicate counts...'
    generate_duplicate_count(train, test)
    print 'Generating mean y features...'
    generate_mean_y_features(train, test)
    # MUST COME AFTER DUPLICATE OPERATIONS
    print 'Generating decomposition features...'
    generate_decomposition_features(train, test)

    return 0
    #return free_test_labels

def encode_categorical_features(train, test):
    for c in train.columns:
        if train[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train[c].values) + list(test[c].values)) 
            train[c] = lbl.transform(list(train[c].values))
            test[c] = lbl.transform(list(test[c].values))

def generate_decomposition_features(train, test, n_components=12):
    generate_decomposition_feature(train, test, decomposition.PCA, n_components)
    generate_decomposition_feature(train, test, decomposition.FastICA, n_components)
    generate_decomposition_feature(train, test, decomposition.TruncatedSVD, n_components)
    # GRP and/or SRP here

def generate_decomposition_feature(train, test, decomposition=decomposition.PCA, 
                                   n_components=12):
    model = decomposition(n_components=n_components, random_state=SEED)
    decomposed_train = model.fit_transform(train.drop(["y"], axis=1))
    decomposed_test = model.transform(test)
    for i in range(n_components):
        train[decomposition.__name__ + str(i)] = decomposed_train[:, i]
        test[decomposition.__name__ + str(i)] = decomposed_test[:, i]

def generate_mean_y_features(train, test):
    usable_columns = train.columns.drop(['ID','y'])
    for c in usable_columns:
        mean_feature = train[[c, 'y']].groupby([c], as_index=False).mean()
        mean_feature.columns = [c, 'mean_'+c]
        train['mean_'+c] = pd.merge(train, mean_feature, on=c, how='left')['mean_'+c]
        test['mean_'+c] = pd.merge(test, mean_feature, on=c, how='left')['mean_'+c]
        train['mean_'+c].fillna(train['mean_'+c].dropna().mean(), inplace=True)
        test['mean_'+c].fillna(test['mean_'+c].dropna().mean(), inplace=True)

# finds duplicate rows and averages the y values, removes
# outliers with an IsolationTree
def fill_duplicates_with_mean(train, test):
    usable_cols = train.columns.drop(['ID','y'])
    grouped = train.groupby(list(usable_cols))

    iso = IsolationForest(n_estimators=100, contamination=0.1, 
                     max_features=1.0, n_jobs=1, random_state=0,
                     verbose=0)

    count = 0
    mean_values = pd.DataFrame(columns=['ID', 'y'])
    for name, group in grouped:
        if group.shape[0] > 1:
            iso.fit(group.drop('ID', axis=1))
            preds = iso.predict(group.drop('ID', axis=1))
            indices = []
            for i in range(len(preds)):
                if preds[i] == 1:
                    indices.append(i)
            if len(indices) < 1: 
                indices = range(len(preds))
            mean = group.y.values[indices].mean()
            for i in group.ID.values:
                mean_values.loc[count] = [i, mean]
                count += 1

    mean_values.ID = mean_values.ID.astype(int)

    merged_train = train.merge(mean_values, how='left', on='ID', suffixes=('','_mean'))
    merged_train['y'] = np.where(np.isnan(merged_train['y_mean']), 
                        merged_train['y'], merged_train['y_mean'])
    merged_train = merged_train.drop('y_mean', axis=1)
    free_labels = test.merge(merged_train, how='left', on=list(usable_cols), suffixes=('','_extra'))
    free_labels = free_labels[['ID', 'y']]

    train['y'] = merged_train['y']
    return free_labels

def generate_duplicate_count(train, test):
    usable_cols = train.columns.drop(['ID','y'])
    data = pd.concat([train.drop('y', axis=1), test])
    grouped = data.groupby(list(usable_cols))

    count = 0
    num_duplicate_values = pd.DataFrame(columns=['ID', 'num_duplicates'])
    for name, group in grouped:
        if group.shape[0] > 1:
            for i in group.ID.values:
                num_duplicate_values.loc[count] = [i, len(group.ID.values)]
                count += 1
    num_duplicate_values.ID = num_duplicate_values.ID.astype(int)
    num_duplicate_values.num_duplicates = \
                num_duplicate_values.num_duplicates.astype(int)

    merged_train = train.merge(num_duplicate_values, how='left', on='ID', suffixes=('','_dup_ERROR'))
    merged_test = test.merge(num_duplicate_values, how='left', on='ID', suffixes=('','_dup_ERROR'))
    train['num_duplicates'] = merged_train['num_duplicates']
    test['num_duplicates'] = merged_test['num_duplicates']

    train.num_duplicates = train.num_duplicates.fillna(1)
    test.num_duplicates = test.num_duplicates.fillna(1)

def generate_bitwise_features(train, test):
    # create bitwise features then use feature elimination
    
    return

# generic time feature generation from data objects (from SBHM kaggle competition)
def generate_time_features(train, test):   
    # Add month-year
    month_year = (train.timestamp.dt.month + train.timestamp.dt.year * 100)
    month_year_cnt_map = month_year.value_counts().to_dict()
    train['month_year_cnt'] = month_year.map(month_year_cnt_map)

    month_year = (test.timestamp.dt.month + test.timestamp.dt.year * 100)
    month_year_cnt_map = month_year.value_counts().to_dict()
    test['month_year_cnt'] = month_year.map(month_year_cnt_map)

    # Add week-year count
    week_year = (train.timestamp.dt.weekofyear + train.timestamp.dt.year * 100)
    week_year_cnt_map = week_year.value_counts().to_dict()
    train['week_year_cnt'] = week_year.map(week_year_cnt_map)

    week_year = (test.timestamp.dt.weekofyear + test.timestamp.dt.year * 100)
    week_year_cnt_map = week_year.value_counts().to_dict()
    test['week_year_cnt'] = week_year.map(week_year_cnt_map)

    # Add month and day-of-week
    train['month'] = train.timestamp.dt.month
    train['dow'] = train.timestamp.dt.dayofweek

    test['month'] = test.timestamp.dt.month
    test['dow'] = test.timestamp.dt.dayofweek

    train.drop('timestamp', axis=1, inplace=True)
    test.drop('timestamp', axis=1, inplace=True)
