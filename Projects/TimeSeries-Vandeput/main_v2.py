import MA
import pandas as pd
import datetime
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

###################### CONFIG ####################
st_date = "2023-09-01"
en_date = "2023-09-10"
iter_name = "MA"

###################### CODE #####################
## Get Data
data = pd.read_csv("Projects\TimeSeries-Vandeput\data\sample_train.csv")
data["Date"] = pd.to_datetime(data.Date).apply(lambda x: x.date())
data = data[(data.Date >= pd.to_datetime(st_date).date()) & 
            (data.Date <= pd.to_datetime(en_date).date())].reset_index(drop=True)
data = data.set_index('Date')

## Movign Average
df = MA.moving_average(data.Demand, extra_periods=6, n=3)

## Exponential Smoothing

print(df.tail(8))

#### OOT Period
pred_data = df[df.index > pd.to_datetime(en_date).date()]
oot_data = pd.read_csv("Projects\TimeSeries-Vandeput\data\sample_test.csv")
oot_data.Date = oot_data.Date.apply(lambda x: datetime.datetime.strptime(x, "%d-%m-%Y").date())

pred_data.index.name = 'Date'
pred_data = pred_data.reset_index()[['Date','Forecast']].merge(oot_data, on='Date', how='left')
# pred_data['Error'] = round((pred_data['Forecast'] - pred_data['Demand'])/ pred_data['Demand'] *100, 2)

## Save plot
ax = plt.gca()
pred_data.plot(x="Date", y="Demand", color="black", ax=ax)
pred_data.plot(x="Date", y="Forecast", color="red", ax=ax)
plt.title(f"Actuals vs Forecast ({iter_name})")

if os.path.exists("./Projects/TimeSeries-Vandeput/output/"):
    plt.savefig(f"./Projects/TimeSeries-Vandeput/output/forecast_{iter_name}.png")
else:
    os.makedirs("./Projects/TimeSeries-Vandeput/output/")
    plt.savefig(f"./Projects/TimeSeries-Vandeput/output/forecast_{iter_name}.png")
plt.close()

## Terminal Display
print("MAE :", {mean_absolute_error(pred_data["Demand"], pred_data["Forecast"])})
print("MSE :", {mean_squared_error(pred_data["Demand"], pred_data["Forecast"])})