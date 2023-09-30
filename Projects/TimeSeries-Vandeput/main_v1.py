import MA
import pandas as pd

###################### CONFIG ####################
st_date = "2015-01-01"
en_date = "2017-12-01"

###################### CODE #####################
## Get Data
data = pd.read_csv("Projects\TimeSeries-Vandeput\data\monthly_train.csv")
data["Date"] = pd.to_datetime(data.Date).apply(lambda x: x.date())
data = data[(data.Date >= pd.to_datetime(st_date).date()) & 
            (data.Date <= pd.to_datetime(en_date).date())].reset_index(drop=True)
data = data.set_index('Date')

## Movign Average
df = MA.moving_average(data.Demand, extra_periods=5, n=3)

print(df)