import streamlit as st
import datetime
import json
from io import StringIO

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import requests
from kneed import KneeLocator
from requests.auth import HTTPBasicAuth
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.pyplot import text

import seaborn as sns

BLOCK_CONFIG_NOF_SHIFTS_DESCRIPTOR = "naive_number_of_shifts"
BLOCK_CONFIG_VERBOSE_DESCRIPTOR = "verbose"
BLOCK_CONFIG_TEAM_NAME_DESCRIPTOR = "team_name"
BLOCK_CONFIG_CSV_FILE_DESCRIPTOR = "CSV_FILE"
BLOCK_CONFIG_SAVE_PLOT_DESCRIPTOR = "PNG_FILE"

file_name = 'data/Events-Match_Test__TEAM_A_vs__TEAM_B-Period_Period_1_Period_Period_2_Period_Period_3.csv'
event_type = 'Shifts'


def read_file(file_name: str, event_type: str) -> pd.DataFrame:
    """
    Create a dataframe from a "Events export"-file downloaded from kinexon.

    For best results, select all periods and only the metrics "Timestamp in local format" and desired event_type.

    :param file_name: path to the file
    :param event_type: event type, e.g. "Shifts", "Changes of Direction", ...
    :return: pandas dataframe
    """

    with open(file_name, "r", encoding="utf-8") as f:
        # read first line
        columns = f.readline()

        # get rid of weird starting characters and remove new line
        columns = columns.strip()

        # find descriptor of element type
        event_columns = None
        while True:
            next_line = f.readline()

            # all event-descriptors start with ;;;;. If it does not start with this, then its data
            if not next_line.startswith(";;;;"):
                break

            # we find event type-descriptor
            if event_type in next_line:
                event_columns = next_line

        if not event_columns:
            raise ValueError(f"CSV does not have a descriptor for {event_type}")

        # delete delete prefix and descriptor, i.e. ";;;;Shifts;"
        event_columns = event_columns[4 + len(event_type):]

        # concat columns
        columns += event_columns

        # read all rows
        file_str = columns
        while next_line:
            file_str += next_line
            next_line = f.readline()

        # replace comma separated first columns with semicolon
        file_str = file_str.replace(", ", ";")

        # save data to df
        df = pd.read_csv(StringIO(file_str), sep=";", index_col=False)

    df = df[df["Event type"].str.contains(event_type[:-1])]

    return df

def generate_block_config(naive: bool,
                          verbose: bool,
                          team_name: str,
                          file_name_raw_data: str,
                          file_name_save_plot: str = None) -> dict:
    """
    Generates a block configuration for the plot_shifts_with_intensity function

    :param naive: If True: Calculates number of shifts by dividing row-numbers by 5 (which is the usual number of players per shift). Otherwise, elbow-method is used. The naive approach tends to deliver better results.
    :param verbose: Show plots and explanations leading to selection of number of shifts
    :param team_name: Name of the team for plot description
    :param file_name_raw_data: path to file for plot description
    :param file_name_save_plot: If given, plot is saved to this file. Ending should be ".png".
    :return: block configuration
    """

    block_config = {
        BLOCK_CONFIG_NOF_SHIFTS_DESCRIPTOR: naive,
        BLOCK_CONFIG_VERBOSE_DESCRIPTOR: verbose,
        BLOCK_CONFIG_TEAM_NAME_DESCRIPTOR: team_name,
        BLOCK_CONFIG_CSV_FILE_DESCRIPTOR: file_name_raw_data,
        BLOCK_CONFIG_SAVE_PLOT_DESCRIPTOR: file_name_save_plot
    }

    return block_config

def get_colour(intensity: float, min_intensity: float, max_intensity: float) -> str:
    """
    Creates a colour between green and red according to intensity, where green is less intense and red is more intense.

    :param intensity: The absolute intensity value.
    :param min_intensity: The minimum intensity value for normalization.
    :param max_intensity: The maximum intensity value for normalization.
    :return: A hex color string, linearly interpolated between green (less intense) and red (more intense).
    """
    normalized_intensity = (intensity - min_intensity) / (max_intensity - min_intensity)
    green = int((1 - normalized_intensity) * 255)
    red = int(normalized_intensity * 255)
    return f"#{red:02x}{green:02x}00"

def plot_shifts(df: pd.DataFrame,
                starting_minute: int,
                time_window: int,
                block_config: dict):
    """
    Creates plot with shifts of all players together with the intensity of all individual shifts

    df requires 3 columns:
        timestamp: datetime -> start of a shift of a player
        time: timedelta -> duration of a shift of a player in seconds
        block_intensity: float -> a relative intensity of a shift for a player between 0 and 1

    :param df: dataframe with shift data
    :param starting_minute:  What minute (in-game-time) does the plot start?
    :param time_window:  how much time should be plotted? (in minutes)
    :param block_config:  initial block config for labelling the plot
    :return: None
    """

    # create plot
    fig, ax = plt.subplots()

    # create correct time formats
    start_time = df['timestamp'].min()
    df['Time Since Start'] = (df['timestamp'] - start_time).dt.total_seconds() / 60

    # plot bars
    for i in df.index:
        start_minute = df['Time Since Start'][i]
        end_minute = start_minute + df['time'][i].total_seconds() / 60

        ax.plot([start_minute, end_minute],
                [df['Player ID'][i], df['Player ID'][i]],
                linewidth=10,
                c=get_colour(df["block_intensity"][i],
                             df["block_intensity"].min(),
                             df["block_intensity"].max()),
                solid_capstyle="butt")

        text(start_minute, df['Player ID'][i], df['Shift_Label'][i], fontsize=9, ha='left', va='center', color='black')

    # format date on x axis
    plt.xlim(starting_minute, starting_minute + time_window)

    # some configurations for background
    ax.grid(axis="y", color="r")
    ax.set(frame_on=False)

    # label axes
    if BLOCK_CONFIG_TEAM_NAME_DESCRIPTOR in block_config:
        plt.title(
            f"Intensity of Shifts of Ice Hockey Players from team: {block_config[BLOCK_CONFIG_TEAM_NAME_DESCRIPTOR]}")
    else:
        plt.title("Intensity of Shifts of Ice Hockey Players ")

    if BLOCK_CONFIG_CSV_FILE_DESCRIPTOR in block_config:
        plt.xlabel(f"In-Game Minute\n\nPlot generated using file:\n {block_config[BLOCK_CONFIG_CSV_FILE_DESCRIPTOR]})")
    else:
        plt.xlabel("In-Game Minute")

    plt.ylabel("Player ID")

    # Add legend for intensity
    norm = mcolors.Normalize(vmin=df["block_intensity"].min(), vmax=df["block_intensity"].max())
    sm = cm.ScalarMappable(norm=norm, cmap=cm.RdYlGn_r)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Skating Intensity')

    # Save plot if file_name_save_plot is given
    if BLOCK_CONFIG_SAVE_PLOT_DESCRIPTOR in block_config and block_config[BLOCK_CONFIG_SAVE_PLOT_DESCRIPTOR]:
        plt.savefig(block_config[BLOCK_CONFIG_SAVE_PLOT_DESCRIPTOR], dpi=300, bbox_inches="tight")

    st.pyplot()

def plot_SIS(df: pd.DataFrame,
             starting_minute: int,
             time_window: int,
             team_name: str = None,
             file_name_raw_data: str = None,
             file_name_save_plot: str = None):
    """
    Creates plot with shifts of all players together with the intensity of all individual shifts

    df requires 3 columns:
        timestamp: datetime -> start of a shift of a player
        time: timedelta -> duration of a shift of a player in seconds
        SIS: float -> a relative intensity of a shift for a player between 0 and 1

    :param df: dataframe with shift data
    :param starting_minute:  What minute (in-game-time) does the plot start?
    :param time_window:  how much time should be plotted? (in minutes)
    :param team_name: team name for plot labelling
    :param file_name_raw_data: path to file for plot description
    :param file_name_save_plot: If given, plot is saved to this file. Ending should be ".png".
    :return:
    """

    # create plot
    fig, ax = plt.subplots()

    # create correct time formats
    df["timestamp"] = pd.to_datetime(df["Timestamp (ms)"], unit="ms")
    df["time"] = pd.to_timedelta(df["Duration (s)"], unit="sec")
    start_time = df['timestamp'].min()
    df['Time Since Start'] = (df['timestamp'] - start_time).dt.total_seconds() / 60

    # plot bars
    for i in df.index:
        start_minute = df['Time Since Start'][i]
        end_minute = start_minute + df['time'][i].total_seconds() / 60

        if end_minute > starting_minute + time_window:
            continue

        ax.plot([start_minute, end_minute],
                [str(df['Player ID'][i]), str(df['Player ID'][i])],
                linewidth=10,
                c=get_colour(df["SIS"][i],
                             df["SIS"].min(),
                             df["SIS"].max()),
                solid_capstyle="butt")

        text(start_minute, str(order_block_labels(df)['Player ID'][i]), df['Shift_Label'][i], fontsize=9, ha='left', va='center', color='black')

    # format date on x axis
    plt.xlim(starting_minute, starting_minute + time_window)

    # some configurations for background
    ax.grid(axis="y", color="r")
    ax.set(frame_on=False)

    # label axes
    if team_name:
        plt.title(
            f"SIS of Ice Hockey Players from team: {team_name}")
    else:
        plt.title("SIS of Ice Hockey Players ")

    if file_name_raw_data:
        plt.xlabel(f"In-Game Minute\n\nPlot generated using file:\n {file_name_raw_data})")
    else:
        plt.xlabel("In-Game Minute")

    plt.ylabel("Player ID")

    # Add legend for intensity
    norm = mcolors.Normalize(vmin=df["SIS"].min(), vmax=df["SIS"].max())
    sm = cm.ScalarMappable(norm=norm, cmap=cm.RdYlGn_r)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('SIS')

    # Save plot if file_name_save_plot is given
    if file_name_save_plot:
        plt.savefig(file_name_save_plot, dpi=300, bbox_inches="tight")

    st.pyplot()

def order_block_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orders the block labels in the DataFrame such that smallest label number is the first shift to happen.

    :param df: DataFrame containing shift data with a 'Shift_Label' column.
    :return: The original DataFrame with the 'Shift_Label' column ordered by the average intensity of each block.
    """

    new_label = 1
    old_label = df['Shift_Label'].iloc[0]
    label_mapping = {
        old_label: new_label
    }

    for label in df["Shift_Label"]:
        if label == old_label:
            continue
        else:
            new_label += 1
            old_label = label
            label_mapping[old_label] = new_label

    # Map the old block labels to the new block labels
    df['Shift_Label'] = df['Shift_Label'].map(label_mapping)

    return df

def find_optimal_amount_of_shifts(df: pd.DataFrame, simple: bool, verbose: bool) -> (int, pd.DataFrame):
    """
    Finds optimal number of shifts to use in kmeans.

    :param df: dataframe
    :param simple: True -> use simple method. Otherwise, use more complex method.
    :param verbose: print intermediate results
    :return: number of shifts to be used and data for kmeans
    """
    nof_players_per_shift = 5
    nof_shifts = int(df.shape[0] / nof_players_per_shift) + 1
    data = list(zip(df["Timestamp (ms)"],
                    df["Duration (s)"]))  # conversion necessary because kmeans cant work with datetime
    if simple:
        # print reasoning
        if verbose:
            print("NAIVE METHOD")
            print(f"Number of rows in dataset = {df.shape[0]}")
            print(f"Probable number of shifts = {nof_shifts}")
    else:
        search_radius = 0.4
        lower_bound = int(nof_shifts - search_radius * nof_shifts)
        upper_bound = int(nof_shifts + search_radius * nof_shifts)
        inertias = []

        for i in range(lower_bound, upper_bound):
            kmeans = KMeans(n_clusters=i)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)

        k1 = KneeLocator(range(lower_bound, upper_bound), inertias, curve="convex", direction="decreasing")
        nof_shifts = k1.knee

        if verbose:
            # show plot
            print("ELBOW METHOD")
            plt.plot(range(lower_bound, upper_bound), inertias, marker='o')
            plt.title('Elbow method')
            plt.xlabel('Number of clusters')
            plt.ylabel('Inertia')
            st.pyplot()

            print("Knee, i.e. calculated number of shifts, is at:", nof_shifts)

    return nof_shifts, data


def add_sis_column(df):
    """
    Adds a 'SIS' (Shift Intensity Score) column to the DataFrame for guest players excluding goalkeepers.
    This function filters the DataFrame for guest players, calculates the optimal number of shifts using KMeans clustering,
    computes the average shift intensity, and calculates the SIS for each player based on the average intensity
    of all their shifts relative to the overall average shift intensity of all players.

    Parameters:
    - df: pandas.DataFrame containing player data including columns for 'Name', 'Timestamp (ms)', 'Duration (s)', and 'Skating Intensity'.

    Returns:
    - pandas.DataFrame: The original DataFrame with an added 'SIS' column for each player.
    """
    # Filter DataFrame for guest players excluding goalkeepers
    df_filtered = df[df["Name"].str.contains("Guest") & ~df["Name"].str.contains("Goalkeeper")].copy()

    # Finding the optimal number of shifts
    optimal_shifts, shift_labels = find_optimal_amount_of_shifts(df_filtered, True, False)

    # Preparing data for clustering
    data_for_clustering = df_filtered[["Timestamp (ms)", "Duration (s)"]]

    # Clustering with KMeans
    kmeans = KMeans(n_clusters=optimal_shifts)
    kmeans.fit(data_for_clustering)

    # Adding cluster labels
    df_filtered["Shift_Label"] = kmeans.labels_

    # Calculating average intensity for each shift
    df_filtered['Average_Shift_Intensity'] = df_filtered.groupby('Shift_Label')['Skating Intensity'].transform('mean')

    # Average intensity of all shifts for each player
    player_shift_means = df_filtered.groupby(['Name', 'Shift_Label'])['Skating Intensity'].mean().reset_index()
    player_average_intensity = player_shift_means.groupby('Name')['Skating Intensity'].mean()

    # Average value of intensities for all shifts of all players
    overall_average_shift_intensity = player_shift_means['Skating Intensity'].mean()

    # Calculating SIS for each player
    player_sis = player_average_intensity / overall_average_shift_intensity

    # Adding the SIS to df_filtered
    df_filtered['SIS'] = df_filtered['Name'].map(player_sis)

    return df_filtered

def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("Shifts and Intensity Plot")

    # Time window slider
    time_window_range = st.slider(
        "Select Time Window Range (minutes):",
        min_value=0,
        max_value=130,
        value=(0, 10),  # start and end
        step=5
    )
    start_time, end_time = time_window_range

    # Load data
    file_name = "data/Events-Match_Test__TEAM_A_vs__TEAM_B-Period_Period_1_Period_Period_2_Period_Period_3.csv"
    event_type = "Shifts"

    df = read_file(file_name, event_type)
    df_no_keepers = df[~df["Name"].str.contains("Goalkeeper")]

    BLOCK_CONFIG_NOF_SHIFTS_DESCRIPTOR = "naive_number_of_shifts"
    BLOCK_CONFIG_VERBOSE_DESCRIPTOR = "verbose"
    BLOCK_CONFIG_TEAM_NAME_DESCRIPTOR = "team_name"
    BLOCK_CONFIG_CSV_FILE_DESCRIPTOR = "CSV_FILE"
    BLOCK_CONFIG_SAVE_PLOT_DESCRIPTOR = "PNG_FILE"

    # Generieren der Block-Konfiguration für die Visualisierung
    block_config = generate_block_config(
        naive=True,  # Oder False, je nach Analyse
        verbose=False,
        team_name="Guest",
        file_name_raw_data=file_name,
        file_name_save_plot=None  # Optional
    )

    df_with_sis = add_sis_column(df)

    # Plot SIS
    plot_SIS(df_with_sis,
             0,
             end_time,
             team_name="Guest",
             file_name_raw_data=file_name
             )

if __name__ == "__main__":
    main()
