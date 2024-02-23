import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from algorithms.utils import read_file, add_sis_column, add_duration_since_start, plot_skating_intensity


def plot_skating_intensity(df, selected_players, start_time, end_time):
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

    # Filtering for the specified time range
    df_filtered = df_filtered[(df_filtered['Duration Since Start'] >= start_time) &
                              (df_filtered['Duration Since Start'] <= end_time)]
    # Creating the visualization
    g = sns.FacetGrid(df_filtered, col="Name", col_wrap=4, height=4, aspect=1.5)

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
    time_window_range = st.slider(
        "Select Time Window Range (minutes):",
        min_value=0,
        max_value=130,
        value=(0, 60),  # Start- und Endwert als Tuple
        step=1
    )
    start_time, end_time = time_window_range

    # Display the visualization
    plot_skating_intensity(df_with_sis, selected_players, start_time, end_time)

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

