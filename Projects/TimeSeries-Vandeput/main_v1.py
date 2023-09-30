import MA
import pandas as pd

d = [28, 19, 18, 13, 19, 16, 19, 18, 13, 16, 16, 11, 18, 15, 13, 15, 13, 15, 13, 11, 13, 10, 12]
d = [28, 19, 18, 13, 19, 16]
df = MA.moving_average(d, extra_periods=5, n=3)
print(df])