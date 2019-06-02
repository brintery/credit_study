#!/usr/bin/python
# -*- coding:utf-8 -*-
 
import pandas as pd
import numpy as np
from woe import WoeFeatureProcess
from conf import PathConf
import datetime
from sklearn.externals import joblib
 
pd.options.mode.chained_assignment = None
pd.set_option('precision', 4)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
 
 
if __name__ == '__main__':
    date = str(datetime.date.today() - datetime.timedelta(days=1))
 
    """
    basic = datetime.datetime.strptime(date, "%Y-%m-%d").date() - datetime.timedelta(days=105)
    train_start = str(basic - datetime.timedelta(days=120))
    train_end = str(basic - datetime.timedelta(days=31))
    test_start = str(basic - datetime.timedelta(days=30))
    test_end = str(basic - datetime.timedelta(days=0))
    """
 
    train_start = '2017-07-01'
    train_end = '2018-06-15'
    test_start = '2018-06-16'
    test_end = '2018-07-15'
 
    path = PathConf(date=date, f_conf='b_card_config.conf', source="b_card_features_df_head.csv")
    logger = logger(log_name='logs', log_level=logging.INFO, log_dir='logs', file_name='log_python.log').getlog()
 
    logger.info('starting...')
    logger.info('start loading data...')
    print('start loading data'.center(80, '='))
    woe = WoeFeatureProcess(train_start=train_start, train_end=train_end,
                            test_start=test_start, test_end=test_end, alpha=0.05)
    woe.load_file(path.config_path, path.data_path)
 
    logger.info('start training woe rule...')
    print('start training woe rule'.center(80, '='))
    dataset_all, dataset_train, dataset_test, model_var_list, identify_var_list = woe.fit(path.woed_train_path,
                                                                                          path.woed_test_path,
                                                                                          path.feature_detail_path,
                                                                                          path.rule_pkl_path)
    print('model features: %s' % len(model_var_list))