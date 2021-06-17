import pandas as pd
import math
from scipy import special
from scipy.signal import butter, lfilter, filtfilt
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb
from copy import deepcopy

DATASET_PATH = './dailylog2016_dataset/subject1/data/acc_csv'
gpu_dict = {
    'objective': 'reg:squarederror',
    'tree_method': 'gpu_hist'
}


def main():
    data_all = pd.read_csv(DATASET_PATH + '/SensorAccelerometerData_labeled_day1.csv')
    milliseconds_per_instance = get_freq(data_all)
    data_valid = data_all.drop(data_all.loc[data_all['label_valid'] == False].index, axis=0).reset_index(drop=True)
    data_valid.drop(['id', 'attr_time', 'label_valid'], axis=1)
    # data_valid = data_all.drop('label_valid', axis=1)

    d1 = encode_categoricals(data_valid)
    d2 = delete_outliers(d1)
    d3 = impute_missing_values(d2)
    d4 = transform(d3, milliseconds_per_instance)

    d4.to_csv('result1.csv')

    # X = data_valid.drop(
    #     ['id', 'attr_time', 'label_environment', 'label_posture', 'label_deviceposition', 'label_activity'],
    #     axis=1)
    # y = data_valid['label_activity']
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    #
    # # TODO: visualize dataset
    #
    # # TODO: work out outliers (ch3 stuff)
    #
    # # TODO: add some more filters (lowpass, pca, etc.)
    #
    # # TODO: Add aggregation methods
    #
    # # TODO: clusters stuff maybe?
    #
    # model = xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=0)
    # print('Model loaded')
    # train_model = model.fit(X_train, y_train)
    # print('Training done')
    # pred = train_model.predict(X_test)
    #
    # print('Model XGboost Report:')
    # print(classification_report(y_test, pred))


def get_freq(df):
    time0 = pd.to_datetime(df['attr_time'].iloc[0], format='%d.%m.%y %H:%M:%S.%f')
    time1 = pd.to_datetime(df['attr_time'].iloc[1], format='%d.%m.%y %H:%M:%S.%f')
    return (time1 - time0).total_seconds() * 1e3


def encode_categoricals(df, method='one hot'):
    print("Encoding categoricals...")
    prefixes = ['label_env', 'label_post', 'label_dpos']
    for i, col in enumerate(['label_environment', 'label_posture', 'label_deviceposition']):
        if method == 'one hot':
            dummies = pd.get_dummies(df[col], prefix=prefixes[i])
            df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
    return df


def delete_outliers(df, c=2):
    print("Deleting outliers...")
    for col in [c for c in df.columns if not 'label' in c]:
        # Computer the mean and standard deviation.
        mean = df[col].mean()
        std = df[col].std()
        N = len(df.index)
        criterion = 1.0 / (c * N)

        # Consider the deviation for the data points.
        deviation = abs(df[col] - mean) / std

        # Express the upper and lower bounds.
        low = -deviation / math.sqrt(2)
        high = deviation / math.sqrt(2)

        for i in range(0, len(df.index)):
            # Determine the probability of observing the point
            prob = 1.0 - 0.5 * (special.erf(high[i]) - special.erf(low[i]))
            # And mark as an outlier when the probability is below our criterion.
            if prob < criterion:
                df[col].iloc[i] = np.nan
    return df


def impute_missing_values(df):
    print("Imputing missing values...")
    for col in [c for c in df.columns if not 'label' in c]:
        df[col] = df[col].interpolate()
        df[col] = df[col].fillna(method='bfill')
    return df


def transform(df, fs):
    print("Applying lowpass filter...")
    cutoff = 1.5
    for col in [c for c in df.columns if not 'label' in c]:
        nyq = 0.5 * fs
        cut = cutoff / nyq

        b, a = butter(10, cut, btype='low', output='ba', analog=False)
        df[col] = filtfilt(b, a, df[col])
    return df


def extract_features(df, cols=None, window=5):
    if cols is None:
        cols = ['attr_azimuth', 'attr_pitch', 'attr_roll']
    group_dfs = []
    for n, group in df.groupby('label_activity_enc'):
        windowed_std = df[cols].rolling(window=window).std()
        windowed_mean = df[cols].rolling(window=window).mean()
        windowed_median = df[cols].rolling(window=window).median()
        windowed_min = df[cols].rolling(window=window).min()
        windowed_max = df[cols].rolling(window=window).max()
        group_df = pd.concat([group.attr_time, windowed_std, windowed_mean, windowed_median, windowed_min, windowed_max,
                              group.label_activity_enc])
        group_dfs.append(group_df)
    return group_dfs


if __name__ == '__main__':
    main()
