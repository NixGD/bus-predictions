import pandas as pd

from utils import *

def get_trip_data():
    """Returns a pandas dataframe with one row for every inbound 1-trip in
    the dataset. Does some minmal processing: eg getting datatypes and filtering
    out some bad data."""
    date_cols = ["service_date"] + stop_list
    trip_data = pd.read_csv("../data/one_trips.csv",
                       index_col="half_trip_id",
                       parse_dates = date_cols)

    # ensure every stop is in the correct
    # currently removes about 1 in 10000 trips
    subsequent_validation = pd.DataFrame({
        "to_" + stop_list[i]:
            (trip_data[stop_list[i]] > trip_data[stop_list[i-1]])
        for i in range(1, len(stop_list))
    })
    valid_row = subsequent_validation.all(axis='columns')
    trip_data = trip_data[valid_row]

    return trip_data


def get_trip_and_history_tables(current_stop_index=6, ONE_HOT_WEEKDAY=True):
    trip_data = get_trip_data()

    prev_stops = stop_list[:current_stop_index]
    current_stop = stop_list[current_stop_index]
    target_stop = "Dudly"

    trip_features = pd.DataFrame({
        "current_time": trip_data[current_stop],
        "to_target": secs_between(trip_data, current_stop, target_stop),
        'service_date': trip_data.service_date
    })

    leg_history = pd.DataFrame({
        "to_"+ prev_stops[i]: secs_between(trip_data, prev_stops[i-1], prev_stops[i])
        for i in range(1, len(prev_stops))
    })

    trip_features['headway'] = get_headways(trip_features, 'current_time')
    trip_features['current_time'] = time_to_secs(trip_features.current_time)

    if ONE_HOT_WEEKDAY:
        weekdays = ['monday', 'tuesday', 'wednesday', 'thursday','friday', 'saturday','sunday']
        for i, day in enumerate(weekdays):
            trip_features[day] = (trip_features.service_date.dt.weekday==i).astype('float32')
    else:
        trip_features['weekday'] = (trip_features.service_date.dt.weekday < 5).astype('float32')
    del trip_features['service_date']

    row_is_na = trip_features.isna().any(axis=1) | leg_history.isna().any(axis=1)

    trip_features = trip_features[~row_is_na]
    leg_history = leg_history[~row_is_na]

    trip_features = trip_features[["current_time", "headway"]+weekdays+["to_target"]]

    return trip_features, leg_history
