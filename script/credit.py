# Copyright (c) 2019 Your Name
#
# -*- coding:utf-8 -*-
# @Script: credit.py
# @Author: Your Name
# @Email: someone@gmail.com
# @Create At: 2019-02-14 21:32:09
# @Last Modified By: Your Name
# @Last Modified At: 2019-03-18 22:39:17
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
random_known = random_df[random_df['MonthlyIncome'].notnull()]
random_known.info()

random_unknown = random_df[random_df['MonthlyIncome'].isnull()]
random_unknown.info()

# drop MonthlyIncome column in random_known
X_known = random_known.drop('MonthlyIncome', axis=1)
X_known.head()

# filter MonthlyIncome column in random_known
y_known = random_known['MonthlyIncome']
y_known.head()

# drop MonthlyIncome column in random_unknown
X_unknown = random_unknown.drop('MonthlyIncome', axis=1)
X_unknown.head()

# use random forest regression to fill value
rfr = RandomForestRegressor(
    random_state=0, n_estimators=100, max_depth=3, n_jobs=-1)
rfr.fit(X_known, y_known)
data.loc[random_df['MonthlyIncome'].isnull(
), 'MonthlyIncome'] = rfr.predict(X_unknown).round(0)
data.info()

# use random forest to fill null value of the column of NumberOfDependents
random_copy = data.copy()
random_copy.info()

copy_unknown = random_copy[random_copy['NumberOfDependents'].isnull()]
copy_unknown.info()
copy_known = random_copy[random_copy['NumberOfDependents'].notnull()]
copy_known.info()

X_known2 = copy_known.drop('NumberOfDependents', axis=1)
X_known2.info()

y_known2 = copy_known['NumberOfDependents']
y_known2

X_unknown2 = copy_unknown.drop('NumberOfDependents', axis=1)
X_unknown2.info()

rfr2 = RandomForestRegressor(
    random_state=0, n_estimators=100, max_depth=3, n_jobs=-1)
rfr2.fit(X_known2, y_known2)
data.loc[random_copy['NumberOfDependents'].isnull(
), 'NumberOfDependents'] = rfr2.predict(X_unknown2).round(0)
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

