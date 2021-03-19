import pandas as pd
import os

stop_list = ["hhgat", "maput", "cntsq", "mit", "hynes", "masta", "Wasma", "Melwa", "Dudly"]

def time_to_secs(time_col):
    return (time_col - pd.to_datetime("1900-01-01")).dt.seconds

def secs_between(df, start, end):
    return (df[end] - df[start]).dt.seconds

def get_headways(df, stop):
    df_sorted = df.sort_values(['service_date', stop])
    return df_sorted.current_time.diff().dt.seconds

def create_output_folder(folder):
    if not os.path.exists("output"):
        os.mkdir("output")
    if not os.path.exists("output/"+folder):
        os.mkdir("output/"+folder)
