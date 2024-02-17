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

import seaborn as sns


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


def read_from_web(credential_file: str) -> pd.DataFrame:
    """
    Reads shift data from web

    :param credential_file: path to credentials file for download
    :return: data frame
    """

    # load credentials
    with open(credential_file, "r") as f:
        credentials = json.load(f)

    # configure parameters
    headers = {'Accept': 'application/json',
               "Authorization": credentials["API_KEY"]}
    auth = HTTPBasicAuth(credentials["USER"], credentials["PASSWORD"])
    url = "https://elverum-api-test.access.kinexon.com/public/v1/events/shift/player/in-entity/session/latest?apiKey=" + credentials["API_KEY"]

    # download data
    req = requests.get(url,
                       headers,
                       auth=auth)

    # extract json from request
    js = json.loads(req.content)

    return pd.json_normalize(js)


def get_colour(intensity: int) -> str:
    """
    creates a colour between green and red  according to intensity

    :param intensity: hockey intensity within [0,1]
    :return: linear interpolation between green = easy and red = heavy in hex-format, e.g. #FF0000
    """

    # LERP between green and red
    green = int((1 - intensity) * 255)
    red = int(intensity * 255)

    colour = "#%02x%02x%02x" % (red, green, 0)  # blue value is 0
    return colour


def plot_shifts(df: pd.DataFrame, time_window: int):
    """
    Plots shifts of all players together with the intensity of all individual shifts

    df requires 3 columns:
        timestamp: datetime -> start of a shift of a player
        time: timedelta -> duration of a shift of a player in seconds
        relative_intensity: float -> a relative intensity of a shift for a player between 0 and 1

    :param df: dataframe with shift data
    :param time_window:  how much time should be plotted? (in minutes)
    :return: None
    """

    # create plot
    fig, ax = plt.subplots()

    # plot bars
    for i in df.index:
        ax.plot([df['timestamp'][i], df['timestamp'][i] + df['time'][i]],
                [df['Name'][i], df['Name'][i]],
                linewidth=10,
                c=get_colour(df["relative_intensity"][i]),
                solid_capstyle="butt")

    # format date on x axis
    myFmt = mdates.DateFormatter('%H:%M:%S')
    ax.xaxis.set_major_formatter(myFmt)
    plt.xticks(rotation=90)
    plt.xlim(df["timestamp"].min(), df["timestamp"].min() + datetime.timedelta(minutes=time_window))

    # some configurations for background
    ax.grid(axis="y", color="r")
    ax.set(frame_on=False)

    # label axes
    plt.xlabel("Time")
    plt.ylabel("Player Name")
    plt.title("Intensity of Shifts of Ice Hockey Players ")

    # display plot
    plt.show()


def plot_shifts_with_intensity(df: pd.DataFrame,
                               time_window_start: int = 0,
                               time_window_duration: int = 5,
                               intensity_indicator: str = "Skating Intensity",
                               block_config: dict = None):
    """
    Plots shifts of all players together with the intensity of all individual shifts.

    ============================================

    Hint: It is best to only supply data from 1 team, not both!
    Hint: For now, time_window_start has to be manually calculated concerning all kinds of breaks
    Hint: Best results, if Goalkeeper is not in the dataset

    ============================================

    Explanation block_config:

    dict {
        "naive_number_of_shifts": True,
        "verbose": False
    }

    naive_number_of_shifts -> If True: Calculates number of shifts by dividing row-numbers by 5 (which is the usual number of players per shift). Otherwise, elbow-method is used. The naive approach tends to deliver better results.
    verbose -> Show plots and explanations leading to selection of number of shifts

    ============================================

    :param df: dataframe with shift data
    :param time_window_start: start of time window to be analysed (in-game minute). Default = 0
    :param time_window_duration: duration of time window to be analysed (in minutes). Default = 5
    :param intensity_indicator:  Which column should be used as intensity indicator? Default = Skating Intensity
    :param block_config:  Configuration how to find players per shift. If None, then intensities for individual player are plotted.
    :return: final dataset that was plotted
    """
    # TODO: convert timestamp to in-game time
    # TODO: find period breaks automatically

    # prepare necessary dataframe columns
    df_plot = pd.DataFrame()
    df_plot["timestamp"] = pd.to_datetime(df["Timestamp (ms)"], unit="ms")
    df_plot["time"] = pd.to_timedelta(df["Duration (s)"], unit="sec")
    df_plot["Timestamp (ms)"] = df["Timestamp (ms)"]  # Necessary for plots
    df_plot["Duration (s)"] = df["Duration (s)"]  # Necessary for plots
    df_plot["Name"] = df["Name"]
    df_plot[intensity_indicator] = df[intensity_indicator]

    # filter time frame
    df_plot = df_plot[df_plot["timestamp"] < df_plot["timestamp"].min() + datetime.timedelta(
        minutes=time_window_start) + datetime.timedelta(minutes=time_window_duration)]
    df_plot = df_plot[
        df_plot["timestamp"] >= df_plot["timestamp"].min() + datetime.timedelta(minutes=time_window_start)]

    # calculate relative intensities
    if block_config:
        # Calculate relativ intensities by block
        nof_shifts, data = find_optimal_amount_of_shifts(df_plot,
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

        # calculate relative intensities per block
        df_shift_intensities = df_plot.groupby("Shift_Label")[intensity_indicator].mean().reset_index().set_index(
            "Shift_Label")
        df_plot['relative_intensity'] = df_plot['Shift_Label'].apply(
            lambda x: df_shift_intensities[intensity_indicator][x])
        df_plot["relative_intensity"] -= df_plot["relative_intensity"].min()
        df_plot["relative_intensity"] /= df_plot["relative_intensity"].max()

    else:
        # calculate relative intensities by player
        df_plot["relative_intensity"] = df_plot[intensity_indicator] - df_plot[intensity_indicator].min()
        df_plot["relative_intensity"] /= df_plot["relative_intensity"].max()

    plot_shifts(df_plot, time_window_duration)

    return df_plot


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

def plot_skating_intensity(df: pd.DataFrame, time_window: int, player_sis: pd.Series):  # Note: The time_window can be implemented but may not be useful
    """
    Plots skating intensity over time for "Guest" players, excluding goalkeepers.

    :param df: DataFrame containing player data including 'Timestamp (ms)', 'Name', and 'Skating Intensity'.
    :param time_window: How much time should be plotted, in minutes, from the start. Not jet implemented.
    :param player_sis: Series containing the Shift Intensity Score for each player.
    Example:
    plot_skating_intensity(add_sis_column(df), 5, player_sis)
    """
    # Convert timestamps to a readable format and calculate the duration since the start
    df['Readable Timestamp'] = pd.to_datetime(df['Timestamp (ms)'], unit='ms')
    df['End Timestamp'] = df['Readable Timestamp'] + pd.to_timedelta(df['Duration (s)'], unit='s')
    df['Duration Since Start'] = (df['Readable Timestamp'] - df['Readable Timestamp'].min()).dt.total_seconds() / 60

    # Filter to include only "Guest" players and exclude goalkeepers
    df_filtered = df[df["Name"].str.contains("Guest") & ~df["Name"].str.contains("Goalkeeper")].copy()

    # Extract the numeric part from the names for sorting
    df_filtered['Name Number'] = df_filtered['Name'].str.extract('(\d+)').astype(int)

    # Sort by the extracted numbers
    df_filtered = df_filtered.sort_values('Name Number')

    # Create the visualization
    g = sns.FacetGrid(df_filtered, col="Name", col_wrap=4, height=4, aspect=1.5)

    def draw_duration_lines(data, **kwargs):
        ax = plt.gca()
        for _, row in data.iterrows():
            start_time = row['Duration Since Start']
            duration = (row['End Timestamp'] - row['Readable Timestamp']).total_seconds() / 60
            end_time = start_time + duration
            # Draw lines indicating the duration of each event (start = green, end = red)
            ax.vlines(x=start_time, ymin=0, ymax=row['Skating Intensity'], color='green', linestyle='-', alpha=0.7)
            ax.vlines(x=end_time, ymin=0, ymax=row['Skating Intensity'], color='red', linestyle='-', alpha=0.7)

    g.map_dataframe(draw_duration_lines)
    g.map_dataframe(sns.scatterplot, 'Duration Since Start', 'Skating Intensity', alpha=0.7)
    g.set_axis_labels('Minutes Since Start', 'Skating Intensity')

    # Add SIS values to each facet
    for ax, name in zip(g.axes.flatten(), df_filtered['Name'].unique()):
        sis_value = player_sis[name]
        ax.text(0.95, 0.90, f'SIS: {sis_value:.2f}', transform=ax.transAxes,
                horizontalalignment='right', verticalalignment='top',
                fontsize=20, bbox=dict(facecolor='white', alpha=0.5))

    g.set_titles("{col_name}")
    plt.show()