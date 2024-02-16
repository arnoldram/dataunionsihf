import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from datetime import datetime
from io import StringIO
from kneed import KneeLocator


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


def plot_shifts_with_intensity(df: pd.DataFrame, time_window: int):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    start_time = df['timestamp'].min()
    df['Time Since Start'] = (df['timestamp'] - start_time).dt.total_seconds() / 60

    # Filter data to include only shifts within the specified time window
    df_within_time_window = df[df['Time Since Start'] <= time_window]

    # Calculate the average intensity for each Shift_Label block
    df_avg_intensities = df_within_time_window.groupby('Shift_Label')['Skating Intensity'].mean().reset_index()

    min_intensity = df_avg_intensities["Skating Intensity"].min()
    max_intensity = df_avg_intensities["Skating Intensity"].max()

    fig, ax = plt.subplots()

    # Plot each shift within the time window
    for shift_label in df_within_time_window['Shift_Label'].unique():
        block_data = df_within_time_window[df_within_time_window['Shift_Label'] == shift_label]
        if not block_data.empty:
            # Get the average intensity for the current block
            block_average_intensity = \
            df_avg_intensities[df_avg_intensities['Shift_Label'] == shift_label]['Skating Intensity'].values[0]
            # Get color based on the block's average intensity
            color = get_colour(block_average_intensity, min_intensity, max_intensity)

            # Plot each shift in the block with the block's color
            for _, row in block_data.iterrows():
                start_minute = row['Time Since Start']
                end_minute = start_minute + row['time'].total_seconds() / 60

                ax.plot([start_minute, end_minute],
                        [row['Name'], row['Name']],
                        linewidth=10,
                        color=color,
                        solid_capstyle="butt")

    plt.xlabel("Minutes Since Start")
    plt.ylabel("Player Name")
    plt.title("Intensity of Shifts of Ice Hockey Players")
    plt.xlim(0, time_window)

    norm = mcolors.Normalize(vmin=min_intensity, vmax=max_intensity)
    sm = cm.ScalarMappable(norm=norm, cmap=cm.RdYlGn_r)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Skating Intensity')

    # Display the plot in Streamlit
    st.pyplot(fig)


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
            plt.show()

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

# Streamlit app starts here
st.title('Ice Hockey Shifts and Intensities Visualization')

# Slider for selecting time window
time_window = st.slider('Select a time window in minutes:', min_value=1, max_value=120, value=50)

# Predefined file path and event type
FILE = "data/Events-Match_Test__TEAM_A_vs__TEAM_B-Period_Period_1_Period_Period_2_Period_Period_3.csv"
EVENT_TYPE = "Shifts"

try:
    df = read_file(FILE, EVENT_TYPE)

    # Assuming 'Timestamp (ms)' and 'Duration (s)' are columns in df for timestamp and duration
    df['timestamp'] = pd.to_datetime(df['Timestamp (ms)'], unit='ms')
    df['time'] = pd.to_timedelta(df['Duration (s)'], unit='s')

    # Add SIS column
    df_filtered = add_sis_column(df)

    # Prepare data for visualization
    df_avg_intensities = df_filtered.groupby('Shift_Label')['Skating Intensity'].mean().reset_index()

    # Visualization
    plot_shifts_with_intensity(df_filtered, time_window)

except ValueError as e:
    st.error(f"Error processing file: {e}")