# Copyright (c) 2019 Your Name
#
# -*- coding:utf-8 -*-
# @Script: credit.py
# @Author: Your Name
# @Email: someone@gmail.com
# @Create At: 2019-02-14 21:32:09
# @Last Modified By: Mindon Gao
# @Last Modified At: 2019-09-24 21:19:39
# @Description: This is description.

# %%
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from reportgen.utils.preprocessing import chimerge

# %%
# read data
data = pd.read_csv("./data/cs-training.csv")

# %%
# delete useless column
del data['Unnamed: 0']

# %%
# see the info
data.info()

# %%
"""
process of null value
"""
# full the null value use random forest
random_df = data.loc[:,
                     "SeriousDlqin2yrs":"NumberOfTime60-89DaysPastDueNotWorse"]
random_df.head()

# %%
# create X and Y to full null value use random forest in MonthlyIncome
# use known value to create X and y, train the random forest,
# and use the unknow column to predict MonthlyIncome
# create notnull and isnull data
mth_known = random_df[random_df['MonthlyIncome'].notnull()]
mth_known.info()

mth_unknown = random_df[random_df['MonthlyIncome'].isnull()]
mth_unknown.info()

# drop MonthlyIncome column in random_known
X_known_mth = mth_known.drop('MonthlyIncome', axis=1)
X_known_mth.info()

# filter MonthlyIncome column in random_known
y_known_mth = mth_known['MonthlyIncome']
y_known_mth.head()


# drop MonthlyIncome column in random_unknown
X_unknown_mth = mth_unknown.drop('MonthlyIncome', axis=1)
X_unknown_mth.info()

# use random forest regression to fill value
rfr = RandomForestRegressor(
    random_state=0, n_estimators=100, max_depth=3, n_jobs=-1)
rfr.fit(X_known_mth, y_known_mth)
data.loc[random_df['MonthlyIncome'].isnull(
), 'MonthlyIncome'] = rfr.predict(X_unknown_mth).round(0)
data.info()

# %%
# use random forest to fill null value of the column of NumberOfDependents
random_copy = data.copy()
random_copy.info()

dpd_unknown = random_copy[random_copy['NumberOfDependents'].isnull()]
dpd_unknown.info()
dpd_known = random_copy[random_copy['NumberOfDependents'].notnull()]
dpd_known.info()

X_known_dpd = dpd_known.drop('NumberOfDependents', axis=1)
X_known_dpd.info()

y_known_dpd = dpd_known['NumberOfDependents']
y_known_dpd

X_unknown_dpd = dpd_unknown.drop('NumberOfDependents', axis=1)
X_unknown_dpd.info()

rfr2 = RandomForestRegressor(
    random_state=0, n_estimators=100, max_depth=3, n_jobs=-1)
rfr2.fit(X_known_dpd, y_known_dpd)
data.loc[random_copy['NumberOfDependents'].isnull(
), 'NumberOfDependents'] = rfr2.predict(X_unknown_dpd).round(0)
data.info()

# %%
"""
process of abnormal value of variables
"""
# data backup
prs_data = data.copy()
prs_data.info()

# %%
# abnormal value detection
# analysis of personal information
prs_data[['age', 'NumberOfDependents']].describe()
# prs_data['age'].plot(kind='box')
# prs_data['NumberOfDependents'].plot(kind='box')

prs_data = prs_data[(prs_data['age'] >= 21) & (prs_data['age'] <= 100)]

# %%
# analysis of personal credit information
prs_data[['RevolvingUtilizationOfUnsecuredLines',
          'NumberOfOpenCreditLinesAndLoans',
          'NumberRealEstateLoansOrLines']].describe()
# prs_data[['NumberOfOpenCreditLinesAndLoans',
#           'NumberRealEstateLoansOrLines']].plot(kind='box')

prs_data = prs_data[(prs_data['RevolvingUtilizationOfUnsecuredLines'] >= 0) & (
    prs_data['RevolvingUtilizationOfUnsecuredLines'] <= 1)]

# %%
# analysis of personal income and liability information variables
prs_data[['DebtRatio', 'MonthlyIncome']].describe()
prs_data.loc[prs_data['DebtRatio'] > 5000, 'DebtRatio'].count()

prs_data = prs_data[prs_data['DebtRatio'] <= 5000]
prs_data = prs_data[prs_data['MonthlyIncome'] <= 100000]

# %%
# analysis of the number of overdue times of borrowers in the past two years
for column in ['NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfTimes90DaysLate']:
    prs_data[prs_data[column] > 90] = 0

# delete the duplicate value
prs_data.drop_duplicates(inplace=True)
prs_data.info()

# copy process data
ft_data = prs_data.copy()
ft_data.info()

# %%
# create variable
ft_data['Debt'] = ft_data['DebtRatio']*ft_data['MonthlyIncome']
ft_data['NumberOfPastDue'] = ft_data['NumberOfTime30-59DaysPastDueNotWorse'] + \
    ft_data['NumberOfTimes90DaysLate'] + \
    ft_data['NumberOfTime60-89DaysPastDueNotWorse']

# %%
"""
feature engineering
0. pre feature engineering
1. bins for continuous and discrete variables
2. calculate iv information for each variables
3. calculate corrence for each variables
4. select variable
5. woe value for each values

continuous variables:
age
RevolvingUtilizationOfUnsecuredLines
Debt
DebtRatio
MonthlyIncome

discrete variables:
NumberOfDependents
NumberOfOpenCreditLinesAndLoans
NumberRealEstateLoansOrLines
NumberOfTime30-59DaysPastDueNotWorse
NumberOfTime60-89DaysPastDueNotWorse
NumberOfTimes90DaysLate
NumberOfPastDue
"""


def optimal_binning_boundary(x, y, nan=float(-999.)):
    '''
    利用决策树获得最优分箱的边界值列表

    Args:
        x: pd.Series
        y: pd.Series
        nan: float

    Returns:
        boundary: list
    '''
    boundary = []  # 待return的分箱边界值列表

    x = x.fillna(nan).values  # 填充缺失值
    y = y.values

    clf = DecisionTreeClassifier(criterion='entropy',  # “信息熵”最小化准则划分
                                 max_leaf_nodes=6,       # 最大叶子节点数
                                 min_samples_leaf=0.05)  # 叶子节点样本数量最小占比

    clf.fit(x.reshape(-1, 1), y)  # 训练决策树

    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    threshold = clf.tree_.threshold

    for i in range(n_nodes):
        if children_left[i] != children_right[i]:  # 获得决策树节点上的划分边界值
            boundary.append(threshold[i])

    boundary.sort()

    min_x = x.min()
    max_x = x.max() + 0.1  # +0.1是为了考虑后续groupby操作时，能包含特征最大值的样本
    boundary = [min_x] + boundary + [max_x]

    return boundary


def feature_woe_iv(x, y, method='chimerge', **kwargs):
    '''
    计算变量各个分箱的WOE、IV值，返回一个DataFrame

    Args:
        x: pd.Series
        y: pd.Series
        nan: float
        method: the method of segment data
            chimerge: chi square method
            bestiv: best iv method
            bestks: best ks method
        **kwargs: another args of chimerge function
            max_intervals:
            threshold:
            bins:
            boundary:

    Returns:
        result_df: pd.Dataframe
    '''
    # x = x.fillna(nan)
    boundary = []

    if method == 'chimerge':
        boundary = chimerge(x, y,
                            max_intervals=kwargs['max_intervals'],
                            threshold=kwargs['threshold'])
    elif method == 'bestiv':
        boundary = optimal_binning_boundary(x, y)
    elif method == 'eqfreq':
        boundary = list(pd.qcut(x, q=kwargs['bins'], retbins=True)[1])
        boundary = boundary[0:-1] + [boundary[-1] + 0.1]
    elif method == 'eqdist':
        boundary = list(pd.cut(x, bins=kwargs['bins'], retbins=True)[1])
        boundary = boundary[0:-1] + [boundary[-1] + 0.1]
    elif method == 'custom':
        boundary = kwargs['boundary']

    df = pd.concat([x, y], axis=1)
    df.columns = ['x', 'y']
    df['bins'] = pd.cut(x=x, bins=boundary, right=False)

    # process NA value to a single bins
    if df['bins'].isnull().any():
        df['bins'].cat.add_categories('NA', inplace=True)
        df['bins'].fillna('NA', inplace=True)

    grouped = df.groupby('bins')['y']
    result_df = grouped.agg([('good', lambda y: (y == 0).sum()),
                             ('bad', lambda y: (y == 1).sum()),
                             ('total', 'count')])

    result_df['good_pct'] = result_df['good'] / \
        result_df['good'].sum()
    result_df['bad_pct'] = result_df['bad'] / \
        result_df['bad'].sum()
    result_df['total_pct'] = result_df['total'] / \
        result_df['total'].sum()

    result_df['bad_rate'] = result_df['bad'] / \
        result_df['total']

    result_df['woe'] = np.log(
        result_df['good_pct'] / result_df['bad_pct'])
    result_df['iv'] = (result_df['good_pct'] -
                       result_df['bad_pct']) * result_df['woe']

    print(f"the variable's IV = {result_df['iv'].sum()}")

    return result_df


# %%
# calculate continues variable's iv
# init iv result
iv_result = {}

continues_var = ft_data[['age',
                         'RevolvingUtilizationOfUnsecuredLines',
                         'Debt',
                         'DebtRatio',
                         'MonthlyIncome']].copy()
continues_rst = {}
for key in list(continues_var.keys()):
    continues_rst[key] = feature_woe_iv(
        continues_var[key], ft_data['SeriousDlqin2yrs'], method='bestiv')
    iv_result[key] = continues_rst[key]['iv'].sum()

# %%
# select discrete variable
discrete_var = ft_data[['NumberOfDependents',
                        'NumberOfOpenCreditLinesAndLoans',
                        'NumberRealEstateLoansOrLines',
                        'NumberOfTime30-59DaysPastDueNotWorse',
                        'NumberOfTime60-89DaysPastDueNotWorse',
                        'NumberOfTimes90DaysLate',
                        'NumberOfPastDue']].copy()
discrete_rst = {}
for key in list(discrete_var.keys()):
    discrete_rst[key] = feature_woe_iv(
        discrete_var[key], ft_data['SeriousDlqin2yrs'], method='chimerge',
        max_intervals=5, threshold=5)
    iv_result[key] = discrete_rst[key]['iv'].sum()

# %%
iv_result_df = pd.DataFrame(iv_result, index=['iv']).T.sort_index(by='iv')

# %%
# correlation analysis
corr_data = ft_data.loc[:,
                        'RevolvingUtilizationOfUnsecuredLines':
                        'NumberOfPastDue'].corr(method='pearson')

for row in range(1, len(corr_data)+1):
    col = row - 1
    corr_data.iloc[0:row, col] = None

corr_data = corr_data[(corr_data >= 0.8) | (
    corr_data <= -0.8)].dropna(how='all').stack().reset_index()
corr_data.rename(columns={'level_0': 'var1',
                          'level_1': 'var2', 0: 'corr'}, inplace=True)


# %%
"""
pre process data
"""
woe_data = ft_data.drop(['Debt', 'NumberOfTime30-59DaysPastDueNotWorse'],
                        axis=1).copy()

var_list = ['RevolvingUtilizationOfUnsecuredLines',
            'age',
            'DebtRatio',
            'MonthlyIncome',
            'NumberOfOpenCreditLinesAndLoans',
            'NumberOfTimes90DaysLate',
            'NumberRealEstateLoansOrLines',
            'NumberOfTime60-89DaysPastDueNotWorse',
            'NumberOfDependents',
            'NumberOfPastDue']

bin_dict = {'RevolvingUtilizationOfUnsecuredLines': [0.0, 0.115, 0.495, 0.773, 1.1],
            'age': [0.0, 36.5, 52.5, 56.5, 63.5, 67.5, 99.1],
            'DebtRatio': [0.0, 0.0192, 0.654, 5000.1],
            'MonthlyIncome': [0.0, 2726.5, 4833.5, 100000.1],
            'NumberOfOpenCreditLinesAndLoans': [-0.01, 1.0, 4.0, 6.0, 58.58],
            'NumberOfTimes90DaysLate': [-0.01, 1.0, 2.0, 3.0, 4.0, 17.17],
            'NumberRealEstateLoansOrLines': [-0.01, 3.0, 4.0, 5.0, 8.0, 54.54],
            'NumberOfTime60-89DaysPastDueNotWorse': [-0.01, 1.0, 2.0, 3.0, 6.0, 11.11],
            'NumberOfDependents': [-0.01, 1.0, 2.0, 3.0, 6.0, 20.2],
            'NumberOfPastDue': [-0.01, 1.0, 2.0, 3.0, 19.19]}

woe_dict = {'RevolvingUtilizationOfUnsecuredLines': [1.22, 0.35, -0.6, -1.27],
            'age': [-0.51, -0.24, -0.01, 0.36, 0.71, 1.07],
            'DebtRatio': [0.38, 0.02, -0.12],
            'MonthlyIncome': [-0.22, -0.11, 0.18],
            'NumberOfOpenCreditLinesAndLoans': [-1.54, -0.39, 0.11, 0.11],
            'NumberOfTimes90DaysLate': [0.35, -2.0, -2.68, -3.01, -3.29],
            'NumberRealEstateLoansOrLines': [0.03, -0.08, -0.38, -0.9, -1.52],
            'NumberOfTime60-89DaysPastDueNotWorse': [0.25, -1.85, -2.65, -2.95, -3.21],
            'NumberOfDependents': [0.16, -0.1, -0.2, -0.35, -0.74],
            'NumberOfPastDue': [0.85, -0.71, -1.54, -2.42]}


#%%
