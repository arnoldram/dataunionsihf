import pandas as pd
from io import StringIO
from requests.auth import HTTPBasicAuth
import requests
import json


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
