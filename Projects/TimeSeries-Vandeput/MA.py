import numpy as np
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta


def moving_average(d, extra_periods=1, n=3):
    """
    to calculate moving average
    Parameters:
        d : a time series that contains the historical demand
        n : (int) lag window
        extra_periods : (int) no of period intend to predict
    Returns:
        dataframe
    """
    # transform input into numpy array
    cpy = d.copy()
    d = np.array(d)
    # input length
    cols = len(d)
    # input np.nan for future extra periods required
    d = np.append(d, [np.nan] * extra_periods)
    # define forecast array
    f = np.full(cols + extra_periods, np.nan)

    # create all t+1 forecasts
    for t in range(n, cols + extra_periods):
        f[t] = round(np.mean(d[t-n: t]), 2)
        if np.isnan(d[t]):
            d[t] = round(np.mean(d[t-n: t]), 2)

    # forecast for all extra periods
    d[cols:] = np.nan

    df_curr = pd.DataFrame.from_dict({"Demand": d[:cols], "Forecast": f[:cols], "Error": d[:cols] - f[:cols]})
    df_curr.index = cpy.index
    df_future = pd.DataFrame.from_dict({"Demand": d[cols:], "Forecast": f[cols:], "Error": d[cols:] - f[cols:]})
    df_future.index = pd.date_range(start=max(df_curr.index)+relativedelta(months=1, day=1),
                                    periods=extra_periods, freq='MS')
    df_future.index = df_future.index.date

    df = pd.concat([df_curr, df_future])

    return df
