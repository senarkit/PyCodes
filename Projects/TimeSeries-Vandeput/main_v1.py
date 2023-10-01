import MA
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

###################### CONFIG ####################
st_date = "2015-01-01"
en_date = "2016-12-01"

###################### CODE #####################
## Get Data
data = pd.read_csv("Projects\TimeSeries-Vandeput\data\monthly_train.csv")
data["Date"] = pd.to_datetime(data.Date).apply(lambda x: x.date())
data = data[(data.Date >= pd.to_datetime(st_date).date()) & 
            (data.Date <= pd.to_datetime(en_date).date())].reset_index(drop=True)
data = data.set_index('Date')

## Movign Average
df = MA.moving_average(data.Demand, extra_periods=6, n=3)

## Exponential Smoothing


#### OOT Period
pred_data = df[df.index > pd.to_datetime(en_date).date()]
oot_data = pd.read_csv("Projects\TimeSeries-Vandeput\data\monthly_test.csv")
oot_data.Date = pd.to_datetime(oot_data['Date']).apply(lambda x: x.date())

pred_data.index.name = 'Date'
pred_data = pred_data.reset_index()[['Date','Forecast']].merge(oot_data, on='Date', how='left')
pred_data['Error'] = round((pred_data['Forecast'] - pred_data['Demand'])/ pred_data['Demand'] *100, 2)

print("MAE :", {mean_absolute_error(pred_data["Demand"], pred_data["Forecast"])})
print("MSE :", {mean_squared_error(pred_data["Demand"], pred_data["Forecast"])})