# Copyright (c) 2019 Your Name
#
# -*- coding:utf-8 -*-
# @Script: woe.py
# @Author: Your Name
# @Email: someone@gmail.com
# @Create At: 2019-03-28 22:23:34
# @Last Modified By: Your Name
# @Last Modified At: 2019-03-28 22:35:31
# @Description: This is description.

import os
import numpy as np
import pandas as pd
import copy
from sklearn.externals import joblib
from sklearn.model_selection import KFold

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_width', 1000)

__all__ = ['WoeFeatureProcess']


