# dataunionsihf

We tried to keep our code as user-friendly as possible. As an example on how to use it and and understand the data, we wrote a jupyter notebook `sample_analysis.iypnb`.

All the more complex functions are in the `utils.py` file.

## Installation

We recommend using PyCharm or VS Code to work with the code.

We also assume that you have some version of Python installed. We developed using Python 3.10, but other versions should work as well.

### Quickstart

1. Clone the repository to your local machine.
2. Open the repository in your IDE.
3. Create and activate a virtual environment (optional but we strongly recommend it)
    - Creation: Run `python -m venv venv` in the terminal
    - Activation: 
      - Windows: Run `.\venv\Scripts\activate` in the terminal
      - Unix: Run `source ./venv/bin/activate` in the terminal (we think, not actually tested)
    - (Deactivation, if necessary: Run `deactivate` in the terminal)
4. Install the required packages by running `pip install -r requirements.txt` in the terminal.
5. You are ready to go.

## Reading data

There are two ways to load data. 

### Manual Download

Download event-data from kinexon. In the process, select all columns you want to analyse. This data will have the following format:

```csv
"Timestamp (ms)";"Timestamp in local format";"Player ID";Name;"Event type"
;;;;"Changes of Direction";"Magnitude (°)";"Skating Deceleration (Max.) (m/s²)";"Skating Acceleration (Max.) (m/s²)";"Direction (left/right)"
;;;;"Down on Pads";"Impact (g)";"Time (ms)"
;;;;Hits;"Magnitude (g)";"Speed (km/h)"
;;;;"Hockey Exertions";"Duration (s)";"Hockey Load / min";"Hockey Load (max.)";"Distance (m)";"Speed (max.) (km/h)";"Hockey Exertion"
;;;;Shifts;"Duration (s)";Distance;"Distance (speed | Very low)";"Distance (speed | Low)";"Distance (speed | Medium)";"Distance (speed | High)";"Distance (speed | Very high)";"Distance (speed | Sprint)";"Distance (speed | 0 - 0 km/h)";"Metabolic Power (Ø)";"Speed (max.)";"Skating Load";"Skating Intensity"
;;;;"Skating Accelerations";"Duration (s)";"Distance (m)";"Speed (max.) (km/h)";"Skating Acceleration (Max.) (m/s²)";"Skating Acceleration (Ø) (m/s²)";"Speed Change (km/h)";"Acceleration Category"
;;;;"Skating Bursts";"Duration (s)";"Distance (m)";"Speed (max.)";"Skating Acceleration (Max.) (m/s²)";"Skating Bursts"
;;;;"Skating Decelerations";"Duration (s)";"Distance (m)";"Speed (max.) (km/h)";"Skating Deceleration (Max.) (m/s²)";"Skating Deceleration (Ø) (m/s²)";"Speed Change (km/h)";"Deceleration Category"
;;;;"Skating Transitions";"Body Rotation (°)";"Previous Speed (km/h)";"Following Speed (km/h)";"Direction (left/right)"
;;;;Sprints;"Duration (s)";"Distance (m)";"Speed (max.) (km/h)";"Speed (Ø) (km/h)";"Sprint category"
;;;;Turns;"Duration (s)";"Radius (m)";"Angle (°)";"Direction (left/right)";"End Speed (km/h)";"Start Speed (km/h)";"Turn category"
[... actual data ...]
```

Our function `utils.read_file(file_name, event_type)` will import this CSV file and return a dataframe.
In our case, `event_type` is "Shifts". The function will dismiss all other lines starting with `;;;;` and read the correct columns for the chosen event type.

Alternatively, the CSV file can be edited manually to have following format:

````csv
"Timestamp (ms)";"Timestamp in local format";"Player ID";Name;"Event type";"Duration (s)";Distance;"Distance (speed | Very low)";"Distance (speed | Low)";"Distance (speed | Medium)";"Distance (speed | High)";"Distance (speed | Very high)";"Distance (speed | Sprint)";"Distance (speed | 0 - 0 km/h)";"Metabolic Power (Ø)";"Speed (max.)";"Skating Load";"Skating Intensity"
````

This enables you to read the data with a simple `pandas.read_csv(file_name)`.

### Automatic download from API

Create a JSON file with following content. Then use `utils.read_file_web()`, supplying the path to the file, to download the data as dataframe.

````json
{
    "API_KEY" : "insert_some_api_key",
    "USER" : "insert_username_to_access_page",
    "PASSWORD" : "insert_passwort_to_accesss_page"
}
````
- API_KEY = api key to access the database
- USER = username to access the website kinexon.com. **NOT** the username to login to kinexon.com
- PASSWORD = password to access the webstie kinexon.com. **NOT** the password to login to kinexon.com.

*Hint:* Does not yet work very well.

## Plotting

### Create Shifts and Plot them

To create shifts and plot them, use `utils.plot_shifts_with_intensity()`.

This function requires the dataframe to plot and a block_config. The block_config is an algorithm configuration, which specifies how to search for shifts, whether you want verbose information, a team name, file name for labelling the plot and an option to save the plots to PNG. 

To simplify the creation of the block_config, we provide a function `utils.generate_block_config()`.

See the `sample_analysis.ipynb` for an example on how to use these functions.

### Shift intensity score (SIS)

TODO: Andri



