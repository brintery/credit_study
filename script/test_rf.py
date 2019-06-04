# Copyright (c) 2019 Mindon Gao
#
# -*- coding:utf-8 -*-
# @Script: test_rf.py
# @Author: Mindon Gao
# @Email: brintery@gmail.com
# @Create At: 2019-06-04 22:47:30
# @Last Modified By: Mindon Gao
# @Last Modified At: 2019-06-04 23:04:58
# @Description: This is description.

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# read data
data = pd.read_csv("./data/cs-training.csv")
data.head()

