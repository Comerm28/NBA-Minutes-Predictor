from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

df = pd.read_csv('/Users/mitch/Desktop/VSCode/NBAStatPredictor/DataSet.csv')
# print(df)

x = df.drop(columns = ['id', 'full_name', 'first_name', 'last_name', 'is_active', 'minAve'])
# print(x)

y = df['minAve']
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)

reg = LinearRegression()

reg.fit(x_train, y_train)

y_pred_test = reg.predict(x_test)

print(r2_score(y_test, y_pred_test))
print(mean_absolute_error(y_test, y_pred_test))