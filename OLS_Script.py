import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
from statsmodels.tools.eval_measures import mse, rmse
import seaborn as sns
pd.options.display.float_format = '{:.5f}'.format
import warnings
import math
import scipy.stats as stats
import scipy
from sklearn.preprocessing import scale
warnings.filterwarnings('ignore')
import pickle

df = pd.read_csv('us_bank_wages/us_bank_wages.txt', sep="\t")

# create dummies
jobc_dummies = pd.get_dummies(df['JOBCAT'], prefix='jobc', drop_first=True)
gend_dummies = pd.get_dummies(df['GENDER'], prefix='gend', drop_first=True)
mino_dummies = pd.get_dummies(df['MINORITY'], prefix='mino', drop_first=True)
educ_dummies = pd.get_dummies(df['EDUC'], prefix='educ', drop_first=True)
# drop columns
df_d = df.drop(['JOBCAT','GENDER','MINORITY','EDUC'], axis=1)
# combine datasets
df_d = pd.concat([df, jobc_dummies, gend_dummies, mino_dummies, educ_dummies], axis=1)
df_d.head()

y = df_d['SALARY']
X = df_d[['SALBEGIN', 'jobc_2', 'gend_1', 'mino_1', 'educ_12', 'educ_14', 'educ_15', 'educ_16', 'educ_17', 'educ_18', 'educ_19', 'educ_20', 'educ_21']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = sm.add_constant(X_train)
results = sm.OLS(y_train, X_train).fit()
X_test = sm.add_constant(X_test)
y_preds = results.predict(X_test)

filename = 'OLS_model.sav'
pickle.dump(results, open(filename, 'wb'))

print(results.summary())
