import numpy as np
import pandas as pd

df = pd.read_csv("weather_classification_data.csv")

# (‑60…60 °C).
fahrenheit_mask = df["Temperature"] > 80
df.loc[fahrenheit_mask, "Temperature"] = (
    (df.loc[fahrenheit_mask, "Temperature"] - 32) * 5 / 9
)

out_of_range_temp = (df["Temperature"] < -60) | (df["Temperature"] > 60)
df.loc[out_of_range_temp, "Temperature"] = np.nan

# 0–100 %
df.loc[(df["Humidity"] < 0) | (df["Humidity"] > 100), "Humidity"] = np.nan

#0–100 %.
df["Precipitation (%)"] = df["Precipitation (%)"].clip(0, 100)

#официально 0–11+. 0–15 на всякий случай.
df["UV Index"] = df["UV Index"].clip(0, 15)

df = df.dropna().reset_index(drop=True)
