#!/usr/bin/env python3

import csv
import os
from collections import defaultdict
from datetime import date
from pprint import pprint

import pandas as pd
import plotly.express as px

COLUMNS = ['Time', 'Distance', 'Avg Speed', 'Avg HR', 'Max HR',
           'Elev Gain', 'Elev Loss', 'Calories', 'Avg Temperature', 'Max Speed',
           'Moving Time', 'Avg Moving Speed']

dts = {
    22.84: date(2020, 3, 23),
    75.77: date(2020, 3, 22),
    62.54: date(2020, 3, 21),
    70.43: date(2020, 3, 20),
    79.39: date(2020, 3, 19),
    95.15: date(2020, 3, 18),
    68.16: date(2020, 3, 17),
    34.12: date(2020, 3, 15),
    32.52: date(2020, 3, 14),
    16.72: date(2020, 3, 14),
    32.94: date(2020, 3, 14),
    28.82: date(2020, 3, 13),
    75.28: date(2020, 3, 12),
    20.46: date(2020, 3, 11),
    39.90: date(2020, 3, 11),
    31.23: date(2020, 3, 10),
    8.19: date(2020, 3, 10),
    55.81: date(2020, 3, 9),
    87.31: date(2020, 3, 8),
    49.56: date(2020, 3, 7),
    69.58: date(2020, 3, 6),
    26.18: date(2020, 3, 5),
    36.53: date(2020, 3, 5),
    73.10: date(2020, 3, 4),
    58.80: date(2020, 3, 3),
    61.46: date(2020, 3, 2),
    73.32: date(2020, 3, 1),
    69.02: date(2020, 2, 29),
    14.02: date(2020, 2, 28),
    43.32: date(2020, 2, 28),
    42.24: date(2020, 2, 27),
    72.57: date(2020, 2, 26),
    45.37: date(2020, 2, 24),
    53.13: date(2020, 2, 23),
    61.50: date(2020, 2, 22),
}

dts2 = {k: (v - date(2020, 2, 21)).days for k, v in dts.items()}


def timeToSecs(inp):
    parts = list(map(int, inp.split(':')))
    if len(parts) > 2:
        return parts[2] + parts[1] * 60 + parts[0] * 60 * 60
    return parts[1] + parts[0] * 60


def solidInt(inp):
    if inp is None:
        return 0
    return int(inp.replace(',', ''))


keyFuncs = {
    'Time': timeToSecs,
    'Distance': float,
    'Avg Speed': float,
    'Avg HR': solidInt,
    'Max HR': solidInt,
    'Elev Gain': solidInt,
    'Elev Loss': solidInt,
    'Calories': solidInt,
    'Avg Temperature': float,
    'Max Speed': float,
    'Moving Time': timeToSecs,
    'Avg Moving Speed': float,
    }


def read_data():
    dataByType = defaultdict(list)

    for filename in os.listdir("raw_daily"):
        with open("raw_daily/" + filename) as csvfile:
            rows = list(csv.DictReader(csvfile))
            for k in COLUMNS:
                dataByType[k].append(keyFuncs[k](rows[-1].get(k)))

    for dist in dataByType["Distance"]:
        dataByType['Date'].append(dts[dist])
        dataByType['Day'].append(dts2[dist])

    return dataByType


def write_csv(dataByType):
    with open("daily_points.csv", 'w') as f:
        fieldnames = ['Date'] + keys
        writer = csv.writer(f)
        writer.writerow(fieldnames)

        for i in range(len(dataByType['Date'])):
            row = []
            for k in fieldnames:
                row.append(dataByType[k][i])
            writer.writerow(row)


def displayData(dataByType):
    df = pd.DataFrame.from_dict(dataByType)
    df = df.groupby(['Date', 'Day'], as_index=False).agg({'Distance': 'sum', 'Elev Gain': 'sum', "Avg HR": "mean"})

    fig = px.bar(df, x='Day', y='Distance', labels={'Distance':'Distance (km)'})
    fig.update_layout(title='Distance')
    fig.update_traces(marker={'color': '#156fa9', 'line_width': 2})
    fig.write_image('rendered/daily_distance.png')

    fig = px.bar(df, x='Day', y='Elev Gain', labels={'Elev Gain':'Elevation gain (meters)'})
    fig.update_layout(title='Elevation gain')
    fig.update_traces(marker={'color': '#156fa9', 'line_width': 2})
    fig.write_image('rendered/daily_elevation.png')

    df = df[df['Avg HR'] > 100]
    fig = px.scatter(df, x='Day', y='Avg HR', trendline="ols", labels={'Avg HR':'Average heart rate (bpm)'})
    fig.update_layout(title='Average heart rate')
    fig.update_traces(marker={'color': '#156fa9', 'line_width': 2, 'size': 20})
    fig.write_image('rendered/daily_heartrate.png')


if __name__ == "__main__":
    dataByType = read_data()
    displayData(dataByType)

