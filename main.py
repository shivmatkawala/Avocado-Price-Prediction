# import numpy as np
import pandas as pd
# import random
import matplotlib.pyplot as plt
from prophet import Prophet
import seaborn as sns
avocado_df = pd.read_csv("avocado.csv")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 20)

# Data Representation..

print(avocado_df)
print(avocado_df.head())
print(avocado_df.tail())
print(avocado_df.head(20))
print(avocado_df.tail(20))


# Data Manipulation

avocado_df = avocado_df.sort_values('Date')
print(avocado_df)

# Data Analyze..

plt.figure(figsize=(10, 10))
plt.plot(avocado_df['Date'], avocado_df['AveragePrice'])
plt.xticks(rotation=90, size=3.25)
plt.show()

plt.figure(figsize=(24, 12))
sns.countplot(x='region', data=avocado_df)
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(24, 12))
sns.countplot(x='year', data=avocado_df)
plt.xticks(rotation=45)
plt.show()


# Data Manipulation

avocado_prophet_df = avocado_df[['Date', 'AveragePrice']]
print(avocado_prophet_df)

avocado_prophet_df = avocado_prophet_df.rename(columns={'Date': 'ds', 'AveragePrice': 'y'})
print(avocado_prophet_df)


# Model Making...

model = Prophet()
model.fit(avocado_prophet_df)

future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)
print(forecast)

figure = model.plot(forecast, xlabel='Date', ylabel='Price')
figure1 = model.plot_components(forecast)
plt.show()

