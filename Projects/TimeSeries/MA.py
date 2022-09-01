import numpy as np
import pandas as pd


def moving_average(d, extra_periods=1, n=3):
    """
    to calculate moving average
    Parameters:
        d : a time series that contains the historical demand
    Returns:
        dataframe
    """
    # transform input into numpy array
    d = np.array(d)
    # input length
    cols = len(d)
    # input np.nan for future extra periods required
    d = np.append(d, [np.nan] * extra_periods)
    # define forecast array
    f = np.full(cols + extra_periods, np.nan)

    # create all t+1 forecasts
    for t in range(n, cols + 1):
        f[t] = round(np.mean(d[t - n : t]), 2)

    # forecast for all extra periods
    f[cols + 1 :] = f[t]
    df = pd.DataFrame.from_dict({"Demand": d, "Forecast": f, "Error": d - f})

    return df
