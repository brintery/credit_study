# Copyright (c) 2019 Your Name
#
# -*- coding:utf-8 -*-
# @Script: credit.py
# @Author: Your Name
# @Email: someone@gmail.com
# @Create At: 2019-02-14 21:32:09
# @Last Modified By: Mindon Gao
# @Last Modified At: 2019-06-04 22:58:14
# @Description: This is description.


import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# read data
data = pd.read_csv("./data/cs-training.csv")
data.head()


# delete useless column
del data['Unnamed: 0']

# see the info
data.info()

# full the null value use random forest
random_df = data.loc[:,
                     "SeriousDlqin2yrs":"NumberOfTime60-89DaysPastDueNotWorse"]
random_df.head()

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

# abnormal value detection
# analysis of personal information
prs_data = data.copy()
prs_data.info()

prs_data[['age', 'NumberOfDependents']].describe()
prs_data['age'].plot(kind='box')
prs_data['NumberOfDependents'].plot(kind='box')

prs_data = prs_data[(prs_data['age'] >= 21) & (prs_data['age'] <= 100)]

# analysis of personal credit information
prs_data[['RevolvingUtilizationOfUnsecuredLines',
          'NumberOfOpenCreditLinesAndLoans', 'NumberRealEstateLoansOrLines']].describe()
prs_data[['NumberOfOpenCreditLinesAndLoans',
          'NumberRealEstateLoansOrLines']].plot(kind='box')

prs_data = prs_data[(prs_data['RevolvingUtilizationOfUnsecuredLines'] >= 0) & (
    prs_data['RevolvingUtilizationOfUnsecuredLines'] <= 1)]

# analysis of personal income and liability information variables
prs_data[['DebtRatio', 'MonthlyIncome']].describe()
prs_data.loc[prs_data['DebtRatio'] > 5000, 'DebtRatio'].count()

prs_data = prs_data[prs_data['DebtRatio'] <= 5000]
prs_data = prs_data[prs_data['MonthlyIncome'] <= 100000]

# analysis of the number of overdue times of borrowers in the past two years
for column in ['NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfTimes90DaysLate']:
    prs_data[prs_data[column] > 90] = 0

# delete the duplicate value
prs_data.drop_duplicates(inplace=True)
prs_data.info()

# copy process data
ft_data = prs_data.copy()
ft_data.info()

# create variable
ft_data['Debt'] = ft_data['DebtRatio']*ft_data['MonthlyIncome']
ft_data['NumberOfPastDue'] = ft_data['NumberOfTime30-59DaysPastDueNotWorse'] + \
    ft_data['NumberOfTimes90DaysLate'] + \
    ft_data['NumberOfTime60-89DaysPastDueNotWorse']
