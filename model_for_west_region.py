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

# Data sorting..

avocado_region_west_df = avocado_df[avocado_df['region'] == 'West']
print(avocado_region_west_df)
avocado_region_west_df = avocado_region_west_df.sort_values('Date')
print(avocado_region_west_df)

# Data Analyze...

plt.figure(figsize=(10, 10))
plt.plot(avocado_region_west_df['Date'], avocado_region_west_df['AveragePrice'])
plt.xticks(rotation=90, size=3.25)
plt.show()

plt.figure(figsize=(24, 12))
sns.countplot(x='year', data=avocado_region_west_df)
plt.xticks(rotation=45)
plt.show()


# Data Manipulation...

avocado_region_west_prophet_df = avocado_region_west_df[['Date', 'AveragePrice']]
print(avocado_region_west_prophet_df)

avocado_region_west_prophet_df = avocado_region_west_prophet_df.rename(columns={'Date': 'ds', 'AveragePrice': 'y'})
print(avocado_region_west_prophet_df)

# Model making of west region...

model1 = Prophet()
model1.fit(avocado_region_west_prophet_df)

future1 = model1.make_future_dataframe(periods=365)
forecast_west = model1.predict(future1)
print(forecast_west)

figure2 = model1.plot(forecast_west, xlabel='Date', ylabel='Price')
figure3 = model1.plot_components(forecast_west)
plt.show()
