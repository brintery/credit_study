# Copyright (c) 2019 Mindon Gao
#
# -*- coding:utf-8 -*-
# @Script: test_rf.py
# @Author: Mindon Gao
# @Email: brintery@gmail.com
# @Create At: 2019-06-04 22:47:30
# @Last Modified By: Mindon Gao
# @Last Modified At: 2019-06-09 22:43:12
# @Description: This is description.

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# read data
data = pd.read_csv("./data/rf_train.csv")
data.info()

test_data = pd.read_csv("./data/cs-test-abnormal.csv")
test_data.info()

del test_data['Unnamed: 0']
del data['NumberOfPastDue']

# train data use rfr
X_feature = data.drop('SeriousDlqin2yrs', axis=1)
y = data['SeriousDlqin2yrs']
X_feature.info()
y.head()

X_test = test_data.drop('SeriousDlqin2yrs', axis=1)
X_test.fillna(0, inplace=True)
X_test.info()

# modeling with rfr
rfr = RandomForestRegressor(
    random_state=0, n_estimators=100, max_depth=3, n_jobs=-1)
rfr.fit(X_feature, y)
test_data['SeriousDlqin2yrs'] = rfr.predict(X_test)

test_data['SeriousDlqin2yrs'].plot(kind='hist', bins=100)
test_data['SeriousDlqin2yrs'].value_counts()

# Test Result
# 1. when use one-zero label, model will give value in [0,1] despite it is RF or XGB.
# 2. when label not one-zero, model will give the predict value which not in [0, 1].
# in situation 1, it not sure the output of the RF is probability