# Sales Forecasting using Time Series
# CSV columns expected: Date, Product_ID, Store_ID, Units_Sold, Revenue, Promotion, Holiday

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA

df=pd.read_csv("sales_data.csv")
df["Date"]=pd.to_datetime(df["Date"])
df=df.sort_values("Date")

# Aggregate to daily sales
daily=df.groupby("Date")["Units_Sold"].sum().asfreq("D").fillna(0)
daily.plot(figsize=(12,5),title="Daily Sales")
plt.ylabel("Units Sold");plt.show()

# Time-based split
split=int(len(daily)*0.80)
train,test=daily.iloc[:split],daily.iloc[split:]

# ARIMA baseline
model=ARIMA(train,order=(1,1,1))
fit=model.fit()
forecast=fit.forecast(steps=len(test))

mae=mean_absolute_error(test,forecast)
rmse=np.sqrt(mean_squared_error(test,forecast))
print("ARIMA MAE:",mae)
print("ARIMA RMSE:",rmse)

plt.figure(figsize=(12,5))
plt.plot(train,label="Train")
plt.plot(test,label="Actual")
plt.plot(forecast,label="Forecast")
plt.legend();plt.title("Sales Forecast");plt.show()

# Refit on all historical data and forecast next 30 days
final_fit=ARIMA(daily,order=(1,1,1)).fit()
future=final_fit.forecast(steps=30)
print("Next 30 days forecast:")
print(future)

future.to_csv("next_30_day_forecast.csv")
