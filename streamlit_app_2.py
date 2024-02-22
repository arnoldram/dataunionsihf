import streamlit as st
import pandas as pd
import utils
import datetime
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

def plot_shifts_with_intensity(df: pd.DataFrame,
                               block_config: dict,
                               time_window_start: int = 0,
                               time_window_duration: int = 5,
                               intensity_indicator: str = "Skating Intensity"
                               ):
    """
    Creates plot with shifts of all players together with the intensity of all individual shifts.

    :param df: dataframe with shift data
    :param time_window_start: start of time window to be analysed (in-game minute). Default = 0
    :param time_window_duration: duration of time window to be analysed (in minutes). Default = 5
    :param intensity_indicator:  Which column should be used as intensity indicator? Default = Skating Intensity
    :param block_config:  Configuration how to find players per shift. If None, then intensities for individual player are plotted.
    :return: the figure object that was plotted
    """
    # TODO: convert timestamp to in-game time
    # TODO: find period breaks automatically

    # prepare necessary dataframe columns
    df_plot = pd.DataFrame()
    df_plot["timestamp"] = pd.to_datetime(df["Timestamp (ms)"], unit="ms")
    df_plot["time"] = pd.to_timedelta(df["Duration (s)"], unit="sec")
    df_plot["Timestamp (ms)"] = df["Timestamp (ms)"]  # Necessary for plots
    df_plot["Duration (s)"] = df["Duration (s)"]  # Necessary for plots
    df_plot["Player ID"] = df["Player ID"].astype(str)  # Necessary for plots
    df_plot[intensity_indicator] = df[intensity_indicator]

    # filter time frame
    df_plot = df_plot[df_plot["timestamp"] < df_plot["timestamp"].min() + datetime.timedelta(
        minutes=time_window_start) + datetime.timedelta(minutes=time_window_duration)]
    df_plot = df_plot[
        df_plot["timestamp"] >= df_plot["timestamp"].min() + datetime.timedelta(minutes=time_window_start)]

    # Calculate relativ intensities by block
    nof_shifts, data = utils.find_optimal_amount_of_shifts(df_plot,
                                                     block_config["naive_number_of_shifts"],
                                                     block_config["verbose"])

    # Find actual shifts using the calculated number of shifts
    kmeans = KMeans(n_clusters=nof_shifts)
    kmeans.fit(data)

    # add labels to dataframe
    df_plot["Shift_Label"] = kmeans.labels_

    if block_config["verbose"]:
        print("Summary of all Shifts. Points are the individual players. Colors are their blocks.")
        plt.scatter(df_plot["Timestamp (ms)"], df_plot["Duration (s)"], c=kmeans.labels_)
        plt.xlabel("Start of shift")
        plt.ylabel("Duration of shift")
        plt.title("Shifts of all players")
        plt.show()

    # calculate block intensity
    df_shift_intensities = df_plot.groupby("Shift_Label")[intensity_indicator].mean().reset_index().set_index(
        "Shift_Label")
    df_plot['block_intensity'] = df_plot['Shift_Label'].apply(
        lambda x: df_shift_intensities[intensity_indicator][x])
    df_plot["block_intensity"] /= 100.0
    df_plot = utils.order_block_labels(df_plot)

    # plot the shifts and return the figure object
    fig = utils.plot_shifts(df_plot,
                      time_window_start,
                      time_window_duration,
                      block_config)

    return fig

def load_data():
    df = utils.read_file("data/Events-Match_Test__TEAM_A_vs__TEAM_B-Period_Period_1_Period_Period_2_Period_Period_3.csv", "Shifts")
    df_no_keepers = df[~df["Name"].str.contains("Goalkeeper")]
    return df_no_keepers

def main():
    st.title("Shifts and Intensity Plot")

    # Load data
    df_no_keepers = load_data()

    # Time window slider
    time_window_range = st.slider(
        "Select Time Window Range (minutes):",
        min_value=0,
        max_value=130,
        value=(0, 60),  # start and end
        step=1
    )
    start_time, end_time = time_window_range

    # Generate block config - adjust according to your utils' requirements
    block_config = utils.generate_block_config(naive=True,
                                               verbose=False,
                                               team_name="Guest",
                                               file_name_raw_data="data/Events-Match_Test__TEAM_A_vs__TEAM_B-Period_Period_1_Period_Period_2_Period_Period_3.csv",
                                               file_name_save_plot="testi_plot.png")

    # Update plot based on slider inputs
    _ = plot_shifts_with_intensity(df_no_keepers, time_window_start=start_time, time_window_duration=end_time,
                                       block_config=block_config)

if __name__ == "__main__":
    main()
