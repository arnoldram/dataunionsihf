# test
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

def plot_skating_intensity(df, selected_players, time_window):
    """
    Plots skating intensity over time for selected players, excluding goalkeepers, within a specified time window.
    """
    # Preprocess DataFrame
    df = add_duration_since_start(df)
    df_filtered = df[df["Name"].isin(selected_players) & ~df["Name"].str.contains("Goalkeeper")].copy()

    # Extracting numeric part from names for sorting (if applicable)
    df_filtered['Name Number'] = df_filtered['Name'].str.extract('(\d+)').astype(int, errors='ignore')

    # Sorting by extracted numbers (if applicable)
    df_filtered = df_filtered.sort_values('Name Number', ignore_index=True)

    # Filtering for the specified time window
    df_filtered = df_filtered[df_filtered['Duration Since Start'] <= time_window]

    # Creating the visualization
    g = sns.FacetGrid(df_filtered, col="Name", col_wrap=4, height=4, aspect=1.5)

    def draw_duration_lines(data, **kwargs):
        ax = plt.gca()
        for _, row in data.iterrows():
            start_time = row['Duration Since Start']
            duration = (row['End Timestamp'] - row['Readable Timestamp']).total_seconds() / 60  # Conversion to minutes
            end_time = start_time + duration
            # Ensure end time is within the time window
            if start_time <= time_window:
                ax.vlines(x=start_time, ymin=0, ymax=row['Skating Intensity'], color='green', linestyle='-', alpha=0.7)
                if end_time <= time_window:
                    ax.vlines(x=end_time, ymin=0, ymax=row['Skating Intensity'], color='red', linestyle='-', alpha=0.7)

    g.map_dataframe(draw_duration_lines)
    g.map_dataframe(sns.scatterplot, 'Duration Since Start', 'Skating Intensity', alpha=0.7)
    g.set_axis_labels('Minutes Since Start', 'Skating Intensity')

    # Add SIS values to each facet
    for ax, name in zip(g.axes.flatten(), df_filtered['Name'].unique()):
        if name in df_filtered['Name'].values:
            sis_value = df_filtered[df_filtered['Name'] == name]['SIS'].iloc[0]
            ax.text(0.95, 0.90, f'SIS: {sis_value:.2f}', transform=ax.transAxes,
                    horizontalalignment='right', verticalalignment='top',
                    fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    g.set_titles("{col_name}")
    st.pyplot(g.fig)

def main():
    st.title("Shift Intensity Index Visualization (based on skating intensity)")
    # Using markdown to format text and include LaTeX for formulas
    st.markdown(r"""
        ### Calculate the average intensity for each shift of a player:

        $$
        \text{Average Shift Intensity} = \frac{\sum (\text{Intensity values of the shift})}{\text{Number of measurements in the shift}}
        $$

        ### Calculate the average intensity of all shifts for a player:

        $$
        \text{Average Player Intensity} = \frac{\sum (\text{Average shift intensities of the player})}{\text{Number of shifts of the player}}
        $$

        ### Calculate the average value of intensities for all shifts of all players:

        $$
        \text{Overall Average Shift Intensity} = \frac{\sum (\text{Average shift intensities of all players})}{\text{Number of all shifts of all players}}
        $$

        ### Determine the Shift Intensity Score (SIS) for each player:

        $$
        \text{SIS} = \frac{\text{Average Player Intensity}}{\text{Overall Average Shift Intensity}}
        $$

        Note: The calculations provide a relative value of shift intensity (based on skating intensity) for each player, representing their performance compared to the average of other players. The SIS (Shift Intensity Score) can then be used to analyze which players have particularly intense or less intense shifts. The values can be visualized to more easily identify trends or patterns.
        """, unsafe_allow_html=True)

    # Load or preprocess your data here
    FILE = "data/Events-Match_Test__TEAM_A_vs__TEAM_B-Period_Period_1_Period_Period_2_Period_Period_3.csv"
    EVENT_TYPE = "Shifts"

    df = read_file(FILE, EVENT_TYPE)  # Implement your own read_file function

    df_with_sis = add_sis_column(df)  # Implement your own add_sis_column function

    # Player selection
    player_options = df_with_sis[df_with_sis['Name'].str.contains("Guest")]['Name'].unique().tolist()
    selected_players = st.multiselect("Select Players:", options=player_options, default=player_options[:4])

    # Time window slider
    time_window = st.slider("Select Time Window (minutes):", min_value=0, max_value=130, value=60, step=1)

    # Display the visualization
    plot_skating_intensity(df_with_sis, selected_players, time_window)

    st.markdown("""
    
    ## Shift Intensity Score (SIS) Interpretation

    The SIS provides insights into each player's performance relative to the team average. It can be interpreted as follows:
    
    - **SIS value of 1.00**: Indicates that the player's average shift intensity aligns precisely with the overall team's average.
    
    - **SIS value greater than 1.00**: Signifies that the player's shifts were more intense than the team's average intensity. A higher value suggests a greater intensity level.
    
    - **SIS value less than 1.00**: Implies that the player's shifts were less intense than the team's average intensity. A lower value indicates a reduced level of intensity.
    
    ### Examples:
    
    - A player with an **SIS of 0.88** had shifts that were less intense compared to the team's average.
    - A player with an **SIS of 1.24** experienced shifts substantially more intense than the team's average.
    
    ### Utilization:
    
    These metrics are particularly valuable for:
    
    - **Assessing Performance**: Gauging players' activity levels and contribution during games.
    - **Strategizing Recovery**: Identifying players who may require additional rest.
    - **Training Adjustments**: Determining who could potentially handle increased workloads.
    
    By analyzing SIS values, coaches and trainers can tailor strategies to optimize both individual and team performance.
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

