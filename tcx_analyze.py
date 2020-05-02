#!/usr/bin/env python3

from collections import defaultdict, namedtuple
from datetime import datetime, date, timedelta
import csv
import io
import os

import numpy as np
import matplotlib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytz
import xmltodict
from PIL import Image, ImageDraw
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit


POINTS_CSV_FILENAME = "trackpoints.csv"
INTERVALS_CSV_FILENAME = "intervals.csv"

PNT_COLUMNS = ['time', 'altitude', 'distance', 'heartrate']
AUCKLAND_TIME = pytz.timezone('Pacific/Auckland')
START_DATETIME = datetime(2020, 2, 21, 11, 0, 0)
START_DATETIME = START_DATETIME.replace(tzinfo=AUCKLAND_TIME)

_Pnt = namedtuple('Pnt', PNT_COLUMNS)

matplotlib.rcParams['font.family'] = 'Open Sans'


class HeartRateException(Exception):
    pass


class Pnt(_Pnt):
    def to_row(self):
        return (self.time.isoformat(), self.altitude, self.distance, self.heartrate)

    @staticmethod
    def from_row(row):
        return Pnt(time=datetime.fromisoformat(row[0]),
                   altitude=float(row[1]),
                   distance=float(row[2]),
                   heartrate=int(row[3]))

    @staticmethod
    def from_raw_xml(raw_pnt):
        parseable_tm = raw_pnt['Time'].rsplit(".", 1)[0]
        if 'HeartRateBpm' not in raw_pnt:
            raise HeartRateException
        naive_time = datetime.fromisoformat(parseable_tm)
        aware_time = naive_time.replace(tzinfo=pytz.UTC).astimezone(AUCKLAND_TIME)
        return Pnt(time=aware_time,
                   altitude=float(raw_pnt['AltitudeMeters']),
                   distance=float(raw_pnt['DistanceMeters']),
                   heartrate=int(raw_pnt['HeartRateBpm']['Value']))


class Interval(object):
    def __init__(self, time, seconds, kms_per_hour, meters_per_second, heartrate):
        self.time = time
        self.seconds = float(seconds)
        self.kms_per_hour = float(kms_per_hour)
        self.meters_per_second = float(meters_per_second)
        self.heartrate = float(heartrate)

    def to_row(self):
        return (self.time.isoformat(), self.seconds, self.kms_per_hour, self.meters_per_second, self.heartrate)

    @staticmethod
    def from_row(row):
        return Interval(time=datetime.fromisoformat(row[0]),
                        seconds=row[1],
                        kms_per_hour=row[2],
                        meters_per_second=row[3],
                        heartrate=row[4])

    def __repr__(self):
        return "<Inteval %ds %.1fdist %.1fel %dhr %r>" % (self.seconds, self.kms_per_hour, self.meters_per_second, self.heartrate, self.time)

    @property
    def grade(self):
        return self.meters_per_second * (60 * 60 / 1000) / self.kms_per_hour

    @property
    def day(self):
        return (self.time.date() - START_DATETIME.date()).days

    @property
    def hour(self):
        td = self.time - START_DATETIME
        return td.days * 24 + int(td.seconds / 3600)

    @staticmethod
    def from_two_pnts(pnt, comp_pnt):
        seconds = (pnt.time - comp_pnt.time).seconds
        kms_distance_delta = (pnt.distance - comp_pnt.distance) / 1000.0
        ms_altitude_delta = (pnt.altitude - comp_pnt.altitude)
        kms_per_hour = kms_distance_delta / (seconds / 60.0 / 60.0)
        meters_per_second = ms_altitude_delta / seconds
        return Interval(pnt.time, seconds, kms_per_hour, meters_per_second, pnt.heartrate)

    @staticmethod
    def from_pnt(pnt, i, points, min_time):
        j = i
        while True:
            j-=1

            try:
                comp_pnt = points[j]
            except IndexError:
                return

            if (pnt.time - comp_pnt.time).seconds < min_time:
                # keep iterating until we get to TIME_INTERVAL seconds
                continue
            if (pnt.time - comp_pnt.time).seconds > min_time * 2:
                # throw out more than 2x TIME_INTERVAL. This is a break or prev day
                return

            if pnt.altitude - comp_pnt.altitude < 0:
                # throw out downhill
                return

            interval = Interval.from_two_pnts(pnt, comp_pnt)

            if interval.kms_per_hour < 5:
                return
            if interval.kms_per_hour > 25:
                return
            if interval.meters_per_second > 1:
                return

            return interval


def process_trackpoints(raw_pnts):
    ret = []
    errors = 0
    for raw_pnt in raw_pnts:
        try:
            ret.append(Pnt.from_raw_xml(raw_pnt))
        except HeartRateException:
            errors += 1
    return ret, errors


def get_points_from_raw_xml():
    all_points = []
    for fn in os.listdir("raw_tcx"):
        if not fn.endswith(".tcx"):
            continue
        fn = "raw_tcx/" + fn
        print("processing %s" % fn)
        with open(fn) as fl:
            doc = xmltodict.parse(fl.read())
            tracks = doc['TrainingCenterDatabase']['Activities']['Activity']['Lap']
            if isinstance(tracks, list):
                for track in tracks:
                    raw_pnts = track['Track']['Trackpoint']
                    points, errors = process_trackpoints(raw_pnts)
                    all_points += points
                    print("processed lap with %d/%d errors" % (errors, len(raw_pnts)))
            else:
                raw_pnts = tracks['Track']['Trackpoint']
                points, errors = process_trackpoints(raw_pnts)
                all_points += points
                print("processed with %d/%d errors" % (errors, len(raw_pnts)))
    return all_points


def write_points_to_csv(points):
    with open(POINTS_CSV_FILENAME, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(PNT_COLUMNS)
        for pnt in points:
            writer.writerow(pnt.to_row())
        print("Wrote points to CSV")


def get_points_from_csv():
    points = []
    with open(POINTS_CSV_FILENAME) as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if not i:
                continue
            points.append(Pnt.from_row(row))
    print("read %d points" % len(points))
    return points


def getDf(intervals):
    ret = []
    for interval in intervals:
        ret.append((interval.kms_per_hour, interval.meters_per_second, interval.grade, interval.day, interval.hour, interval.time, interval.heartrate))
    return pd.DataFrame(ret, columns = ['horiz_speed', 'vert_speed', 'grade', 'day', 'hour', 'time', 'heartrate'])


def make_intervals(points, min_time):
    intervals = []
    for i, pnt in enumerate(points):
        intervals.append(Interval.from_pnt(pnt, i, points, min_time))

    intervals = list(filter(None, intervals))
    print("Generated %d intervals" % len(intervals))
    return intervals


def curve_fit(X, a, b, c, d, e):
    return X[0]*a + X[1]*b + X[2]*c + X[3]*d + e


def write_intervals_to_csv(intervals):
    with open(INTERVALS_CSV_FILENAME, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(PNT_COLUMNS)
        for interval in intervals:
            writer.writerow(interval.to_row())
        print("Wrote intervals to CSV")


def get_intervals_from_csv():
    intervals = []
    with open(INTERVALS_CSV_FILENAME) as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if not i:
                continue
            intervals.append(Interval.from_row(row))
    print("read %d intervals" % len(intervals))
    return intervals


def clean_data():
    write_points_to_csv(get_points_from_raw_xml())
    write_intervals_to_csv(make_intervals(get_points_from_csv(), 90))


def animate_intervals():
    intervals = get_intervals_from_csv()
    df = getDf(intervals)
    df = df[df['grade'] >= .04][df['grade'] <= .06][df['horiz_speed'] < 12][df['horiz_speed'] > 8]

    print("Processing %d points" % len(df))
    marker_dict = {
        'color': '#156fa9',
        'line_width': .5,
        'opacity': .25,
        }

    images = []

    last_len = 0
    num_days = 30
    days_visible = 5
    for hour in range(0, num_days*24):
        af = df[df.hour < hour]
        print(hour, len(af))
        if len(af) == 0:
            continue
        if len(af) != last_len:
            last_len = len(af)
            fig = px.scatter(af, x='time', y='heartrate', trendline="ols")
            fig.update_traces(marker=marker_dict, selector=dict(mode='markers'))
            fig.update_traces(marker={'color': '#a92415', 'line_width': 2}, selector=dict(mode='lines'))
            fig.update_layout(title='Heart rate over a 30 day cycling tour (3-5% grade at 8-12 km/hr)')
            fig['layout']['yaxis'].update(range=[100, 200], autorange=False)
        start_range = max(START_DATETIME, START_DATETIME + timedelta(hours=hour-days_visible*24))
        end_range = max(START_DATETIME + timedelta(hours=hour), START_DATETIME + timedelta(hours=24*days_visible))
        fig['layout']['xaxis'].update(range=[start_range, end_range], autorange=False, title="")
        img_bytes = fig.to_image(format="png")
        images.append(Image.open(io.BytesIO(img_bytes)))

    for hour in range(0, (num_days-days_visible)*24, 8):
        start_range = START_DATETIME + timedelta(hours=(num_days-days_visible)*24 - hour)
        end_range = START_DATETIME + timedelta(days=num_days)
        fig['layout']['xaxis'].update(range=[start_range, end_range], autorange=False, title="")
        img_bytes = fig.to_image(format="png")
        images.append(Image.open(io.BytesIO(img_bytes)))

    for _ in range(75):
        images.append(images[-1])

    images[0].save('graph.gif',
                    save_all=True, append_images=images[1:], optimize=False, duration=30, loop=0)


def analyze_intervals():
    intervals = get_intervals_from_csv()
    df = getDf(intervals)
    print("Total intervals %d" % len(df))

    thirdColor = '#a96e15'

    marker_dict = {
        'color': '#156fa9',
        'line_width': .5,
        'opacity': .25,
        }

    fig = px.scatter(df, x='horiz_speed', y='heartrate', trendline="ols")
    fig.update_traces(marker=marker_dict, selector=dict(mode='markers'))
    fig.update_traces(marker={'color': '#a92415', 'line_width': 2}, selector=dict(mode='lines'))
    fig.update_layout(title='Speed vs. heartrate')
    fig.write_image('static/speed_vs_heartrate.png')

    fig = px.scatter(df, x='grade', y='heartrate', trendline="ols")
    fig.update_traces(marker=marker_dict, selector=dict(mode='markers'))
    fig.update_traces(marker={'color': '#a92415', 'line_width': 2}, selector=dict(mode='lines'))
    fig.update_layout(title='Grade vs. heartrate', xaxis_tickformat = '%')
    fig.write_image('static/grade_vs_heartrate.png')

    fig = px.scatter(df, x='horiz_speed', y='grade')
    fig.update_traces(marker=marker_dict, selector=dict(mode='markers'))
    fig.update_traces(marker={'color': '#a92415', 'line_width': 2}, selector=dict(mode='lines'))
    fig.update_layout(title='Speed vs. grade', yaxis_tickformat = '%')
    fig.write_image('static/speed_vs_grade.png')

    fig = px.scatter(df, x='horiz_speed', y='grade', color='heartrate')
    fig.update_traces(marker={'color': '#a92415', 'line_width': 2}, selector=dict(mode='lines'))
    fig.update_layout(title='Heart rate by speed and grade', yaxis_tickformat = '%')
    fig.write_image('static/speed_vs_grade_w_heartrate.png')

    X_all = df[['horiz_speed', 'grade', 'day']]
    y_all = df['heartrate']

    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.1)

    model = LinearRegression()
    model.fit(X_train, y_train)

    print("Score %.3f" % model.score(X_test, y_test))
    print("Coefs", model.coef_)
    print("Intercept", model.intercept_)


if __name__ == "__main__":
    clean_data()
    analyze_intervals()
    animate_intervals()
