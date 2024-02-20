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
    :return: plot
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
        plt.title(f"Intensity of Shifts of Ice Hockey Players from team: {block_config[BLOCK_CONFIG_TEAM_NAME_DESCRIPTOR]}")
    else:
        plt.title("Intensity of Shifts of Ice Hockey Players ")

    if BLOCK_CONFIG_CSV_FILE_DESCRIPTOR in block_config:
        plt.xlabel(f"In-Game Minute\n\nPlot generated using file:\n {block_config[BLOCK_CONFIG_CSV_FILE_DESCRIPTOR]})")
    else:
        plt.xlabel("In-Game Minute")

    plt.ylabel("Player Name")

    # Add legend for intensity
    norm = mcolors.Normalize(vmin=df["block_intensity"].min(), vmax=df["block_intensity"].max())
    sm = cm.ScalarMappable(norm=norm, cmap=cm.RdYlGn_r)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Skating Intensity')

    # Save plot if file_name_save_plot is given
    if BLOCK_CONFIG_SAVE_PLOT_DESCRIPTOR in block_config and block_config[BLOCK_CONFIG_SAVE_PLOT_DESCRIPTOR]:
        plt.savefig(block_config[BLOCK_CONFIG_SAVE_PLOT_DESCRIPTOR],dpi=300, bbox_inches = "tight")

    plt.show()


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


def plot_shifts_with_intensity(df: pd.DataFrame,
                               block_config: dict,
                               time_window_start: int = 0,
                               time_window_duration: int = 5,
                               intensity_indicator: str = "Skating Intensity"
                               ):
    """
    Creates plot with shifts of all players together with the intensity of all individual shifts.

    ============================================

    Hint: It is best to only supply data from 1 team, not both!
    Hint: For now, time_window_start has to be manually calculated concerning all kinds of breaks
    Hint: Best results, if Goalkeeper is not in the dataset

    ============================================

    Explanation block_config:
        Use generate_block_config() to create a block_config.

    ============================================

    :param df: dataframe with shift data
    :param time_window_start: start of time window to be analysed (in-game minute). Default = 0
    :param time_window_duration: duration of time window to be analysed (in minutes). Default = 5
    :param intensity_indicator:  Which column should be used as intensity indicator? Default = Skating Intensity
    :param block_config:  Configuration how to find players per shift. If None, then intensities for individual player are plotted.
    :return: the final dataset that was plotted
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

    # calculate block intensity
    df_shift_intensities = df_plot.groupby("Shift_Label")[intensity_indicator].mean().reset_index().set_index(
        "Shift_Label")
    df_plot['block_intensity'] = df_plot['Shift_Label'].apply(
        lambda x: df_shift_intensities[intensity_indicator][x])
    df_plot["block_intensity"] /= 100.0
    df_plot = order_block_labels(df_plot)

    fig = plot_shifts(df_plot,
                      time_window_start,
                      time_window_duration,
                      block_config)


    return df_plot, fig


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

def add_duration_since_start(df):
    """
    Adds a 'Duration Since Start' column to the dataframe based on 'Timestamp (ms)'.
    """
    df['Readable Timestamp'] = pd.to_datetime(df['Timestamp (ms)'], unit='ms')
    df['End Timestamp'] = df['Readable Timestamp'] + pd.to_timedelta(df['Duration (s)'], unit='s')
    df['Duration Since Start'] = (df['Readable Timestamp'] - df['Readable Timestamp'].min()).dt.total_seconds() / 60
    return df

def plot_skating_intensity(df, selected_players, start_time, end_time):
    """
    Plots skating intensity over time for selected players, excluding goalkeepers, within a specified time window.

    :param df: DataFrame containing player data. Expected columns include 'Timestamp (ms)', 'Name', 'Skating Intensity',
               'End Timestamp', and 'Readable Timestamp'. The 'Timestamp (ms)' is used to calculate the duration since
               the start of the observation period.
    :param selected_players: List of player names to include in the plot. Goalkeepers are automatically excluded.
    :param start_time: The start time for the period of interest, measured as minutes since the observation period began.
    :param end_time: The end time for the period of interest, measured as minutes since the observation period began.

    Example:
    plot_skating_intensity(df_with_sis, selected_players, 0, 130)
    """
    # Preprocess DataFrame
    df = add_duration_since_start(df)
    df_filtered = df[df["Name"].isin(selected_players) & ~df["Name"].str.contains("Goalkeeper")].copy()

    # Extracting numeric part from names for sorting (if applicable)
    df_filtered['Name Number'] = df_filtered['Name'].str.extract('(\d+)').astype(int, errors='ignore')

    # Sorting by extracted numbers (if applicable)
    df_filtered = df_filtered.sort_values('Name Number', ignore_index=True)

    # Filtering for the specified time range
    df_filtered = df_filtered[(df_filtered['Duration Since Start'] >= start_time) &
                              (df_filtered['Duration Since Start'] <= end_time)]
    # Creating the visualization
    g = sns.FacetGrid(df_filtered, col="Name", col_wrap=4, height=4, aspect=1.5)
    plt.figtext(0, -0.05, "Shift Intensity Score (SIS) Calculation and Interpretation:", ha='left', fontsize=14,
                fontweight='bold')
    plt.figtext(0, -0.1, """
        - Average Shift Intensity: Calculated by dividing the sum of intensity values of the shift by the number of measurements in the shift.
        - Average Player Intensity: Calculated by dividing the sum of the player's average shift intensities by the number of shifts of the player.
        - Overall Average Shift Intensity: Calculated by dividing the sum of the average shift intensities of all players by the number of all shifts of all players.
        - SIS: The SIS for each player is determined by dividing the Average Player Intensity by the Overall Average Shift Intensity.""", ha='left', fontsize=14)
    plt.figtext(0, -0.125, "Interpretation:", ha='left', fontsize=14,
                fontweight='bold')
    plt.figtext(0, -0.175, """
        - SIS value of 1.00: Player's average shift intensity aligns with the overall team's average.
        - SIS > 1.00: Player's shifts were more intense than the team's average. Higher values indicate greater intensity.
        - SIS < 1.00: Player's shifts were less intense than the team's average. Lower values indicate reduced intensity.
        """, ha='left', fontsize=14)

    def draw_duration_lines(data, **kwargs):
        ax = plt.gca()
        for _, row in data.iterrows():
            start_time_of_shift = row['Duration Since Start']
            duration = (row['End Timestamp'] - row['Readable Timestamp']).total_seconds() / 60  # Conversion to minutes
            end_time_of_shift = start_time_of_shift + duration
            # Ensure the shift time is within the time range
            if start_time_of_shift >= start_time and end_time_of_shift <= end_time:
                ax.vlines(x=start_time_of_shift, ymin=0, ymax=row['Skating Intensity'], color='green', linestyle='-', alpha=0.7)
                ax.vlines(x=end_time_of_shift, ymin=0, ymax=row['Skating Intensity'], color='red', linestyle='-', alpha=0.7)


    g.map_dataframe(draw_duration_lines)
    g.map_dataframe(sns.scatterplot, 'Duration Since Start', 'Skating Intensity', alpha=0.7)
    g.set_axis_labels('Minutes Since Start', 'Skating Intensity')

    # Add SIS values to each facet
    for ax, name in zip(g.axes.flatten(), df_filtered['Name'].unique()):
        if name in df_filtered['Name'].values:
            sis_value = df_filtered[df_filtered['Name'] == name]['SIS'].iloc[0]
            ax.text(0.95, 0.90, f'SIS: {sis_value:.2f}', transform=ax.transAxes,
                    horizontalalignment='right', verticalalignment='top',
                    fontsize=20, bbox=dict(facecolor='white', alpha=0.5))

    g.set_titles("{col_name}")
    plt.show()