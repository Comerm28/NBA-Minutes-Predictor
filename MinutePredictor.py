from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

pd.set_option('display.max_columns', None)

def createModel():
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

    print('R^2:', r2_score(y_test, y_pred_test))
    print('Mean absolute error:', mean_absolute_error(y_test, y_pred_test))

    joblib.dump(reg, 'currentmodel.pkl')

# createModel()

reg = joblib.load('currentmodel.pkl')

ex = pd.read_csv('/Users/mitch/Desktop/VSCode/NBAStatPredictor/DataDemonstration.csv')
xDemonstrate = ex.drop(columns = ['minAve'])

print('Jaden McDaniels, Naz Reid, and Nickeil Alexander-Walker\'s MPG:', reg.predict(xDemonstrate))
print('Actual: 27.588028169014084, 19.23926380368098, 19.62541806020067')