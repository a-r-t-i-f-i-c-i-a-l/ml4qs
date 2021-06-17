import pandas as pd
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

data_all = pd.read_csv(DATASET_PATH + '/SensorAccelerometerData_labeled_day1.csv')
data_valid = data_all.loc[data_all['label_valid'] == True].drop('label_valid', axis=1)
X = data_valid.drop(['id', 'attr_time', 'label_environment', 'label_posture', 'label_deviceposition', 'label_activity'],
                    axis=1)
y = data_valid['label_activity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# TODO: edit granularity df = df.iloc[::5]

# TODO: work out outliers (ch3 stuff)

# TODO: Add aggregation methods

# TODO: the ch5 stuff maybe?

model = xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=0)
print('Model loaded')
train_model = model.fit(X_train, y_train)
print('Training done')
pred = train_model.predict(X_test)

print('Model XGboost Report:')
print(classification_report(y_test, pred))

