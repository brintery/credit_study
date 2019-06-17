#!/usr/bin/python
# -*- coding:utf-8 -*-
 
import os
import datetime
 
class PathConf(object):
    """
    有些配置没什么用，自己舍去
    """
    def __init__(self, date=str(datetime.date.today()), f_conf='test.conf', source='test.csv'):
        self.conf = os.path.join(os.getcwd(), 'conf')
        self.source = os.path.join(os.getcwd(), 'source')
        self.result = os.path.join(os.getcwd(), 'result')
        self.tmp = os.path.join(os.getcwd(), 'tmp')
        self.rec_date = date
 
        self.config_path = os.path.join(self.conf, f_conf)
        self.data_path = os.path.join(self.source, source)
        self.woed_train_path = os.path.join(self.tmp, 'woed_train.csv')
        self.woed_test_path = os.path.join(self.tmp, 'woed_test.csv')
        self.feature_detail_path = os.path.join(self.result, 'detail.csv')
        self.rule_pkl_path = os.path.join(self.result, 'woe_rule.pkl')
        self.model_pkl_path = os.path.join(self.result, 'model.pkl')
        self.user_score_path = os.path.join(self.result, 'score_%s.csv' % self.rec_date)
        self.user_score_nohead_path = os.path.join(self.result, 'score_%s_nohead.csv' % self.rec_date)
        self.user_score_stat_path = os.path.join(self.result, 'score_stat_%s.csv' % self.rec_date)
        self.report_name = '%s_report.html' % self.rec_date
        self.report_path = os.path.join(self.result, '%s_report.html' % self.rec_date)