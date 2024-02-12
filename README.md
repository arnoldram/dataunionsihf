# dataunionsihf

## Reading data

There are two ways to load data. 

### Manual Download

Download event-data from kinexon. In the process, select all columns you want to analyse.
Then use `utils.read_file()` to import the csv as dataframe.

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