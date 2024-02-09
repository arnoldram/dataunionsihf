import pandas as pd
from io import StringIO


def read_file(file_name: str) -> pd.DataFrame:
    """
    Create a dataframe from a "Events export"-file downloaded from kinexon.

    For best results, select all periods and only the metrics "Timestamp in local format" and "Shifts".

    :param file_name: path to the file
    :return: pandas dataframe
    """

    with open(file_name, "r", encoding="utf-8") as f:
        # read first line
        columns = f.readline()

        # get rid of weird starting characters and remove new line
        columns = columns.strip()
        # find "Shifts"-descriptor
        shift_columns = None
        while True:
            next_line = f.readline()

            # all event-descriptors start with ;;;;. If it does not start with this, then its data
            if not next_line.startswith(";;;;"):
                break

            # we find Shifts-descriptor or
            if "Shifts" in next_line:
                shift_columns = next_line

        if not shift_columns:
            raise ValueError("CSV does not have a descriptor for Shifts")

        # delete delete prefix and descriptor, i.e. ";;;;Shifts;"
        shift_columns = shift_columns[9:]

        # concat columns
        columns += shift_columns
        print(columns)
        #        columns = columns.replace("Metabolic Power (Ã˜)", "")
        # read all rows
        file_str = columns
        while next_line:
            file_str += next_line
            next_line = f.readline()

        # replace comma separated first columns with semicolon
        file_str = file_str.replace(", ", ";")

        # save data to df
        df = pd.read_csv(StringIO(file_str), sep=";", index_col=False)

        return df
