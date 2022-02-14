import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px

import plotly.graph_objects as go
from my_func import Rsquared

df = pd.read_csv('D:\\Programming\\Udemy - 2022 Python for ML & DS\\DATA\\Advertising.csv')

fig, axes = plt.subplots(nrows=1,ncols=3, figsize = (16,6))

axes[0].plot(df['TV'], df['sales'],'o')
axes[0].set_ylabel("Sales")
axes[0].set_title("TV Spend")

axes[1].plot(df['radio'], df['sales'],'o')
axes[1].set_ylabel("Sales")
axes[1].set_title("Radio Spend")

axes[2].plot(df['newspaper'], df['sales'],'o')
axes[2].set_ylabel("Sales")
axes[2].set_title("Newspaper Spend")

plt.tight_layout()

# Histogram

TV_hist = px.histogram(df,x='TV',nbins=50)
radio_hist = px.histogram(df,x='radio',nbins=50)
news_hist = px.histogram(df,x='newspaper',nbins=50)
sales_hist = px.histogram(df,x='sales',nbins=50)

X = df.drop('sales', axis=1)
y = df['sales']


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

#Create test set dataframe
df_test_set = pd.merge(X_test, y_test, how ='left', left_index=True, right_index=True)

# Train model
model = LinearRegression()
final_model = model.fit(X_train,y_train)

y_pred = model.predict(X_test)

# Evaluate metrics
mae = mean_absolute_error(y_test,y_pred)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
rsquare = Rsquared(y_test,y_pred)

# Residual Plot
residuals = y_test - y_pred

residual_plot = go.Figure()

residual_plot.add_scatter(x= y_test, y= residuals, mode='markers')
residual_plot.add_hline(0)

