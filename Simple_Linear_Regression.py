import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

import plotly.graph_objects as go
from my_func import Rsquared


# Preprocessing data
df = pd.read_csv('D:\\Programming\\Udemy - 2022 Python for ML & DS\\DATA\\Advertising.csv')
df['total_spend'] = df['TV'] + df['radio'] + df['newspaper']
cols = df.columns.tolist()
cols = cols[:3] + cols[-1:] 
cols.append('sales')

# Reassign df
df = df[cols]

# Assign X and y
X = df['total_spend']
y = df['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# Plot Figure
fig = go.Figure()
fig.add_scatter(x= X_train, y= y_train, mode= 'markers', name= 'train')
fig.add_scatter(x= X_test, y= y_test, mode= 'markers', name= 'test')

fig_train = go.Figure()
fig_train.add_scatter(x= X_train, y= y_train, mode= 'markers')

fig_test = go.Figure()
fig_test.add_scatter(x= X_test, y= y_test, mode= 'markers')

# Train model
X_train_new = np.array(X_train).reshape(-1,1)
X_test_new = np.array(X_test).reshape(-1,1)
model = LinearRegression()
final_model = model.fit(X_train_new,y_train)

# Prediction
y_pred = final_model.predict(X_test_new)

# Plot best fit line
fig_fit = fig_train
fig_fit.add_scatter(x=X_test, y= y_pred, mode='lines')

# Evaluation Metrics
mean_sq_er = mean_squared_error(y_test,y_pred)
mean_ab_er = mean_absolute_error(y_test,y_pred)
rmse = np.sqrt(mean_sq_er)
rsquare = Rsquared(y_test,y_pred)