import json
import selenium
import sys
from selenium import webdriver
from time import sleep


BROWSER_LOAD_DELAY = 6


conn = None
def get_connection():
    """Returns a StatsNbaComApi connection
    """
    global conn
    if conn is None:
        conn = StatsNbaComApi()
    return conn


class StatsNbaComApi:
    """
    A stats.nba.com API using selenium.
    """
    def __init__(self):
        self.browser = None

    def __del__(self):
        if self.browser:
            self.browser.close()

    def connect(self):
        """Connects to the stats.nba.com API through a browser window.
        Does not seem to work with headless browsers.
        """
        if self.browser:
            return
        options = webdriver.ChromeOptions()
        self.browser = webdriver.Chrome(options=options)
        self.browser.get('https://stats.nba.com')
        sleep(BROWSER_LOAD_DELAY)

    def get(self, uri):
        """Sends an API request to stats.nba.com
        """
        if self.browser is None:
            self.connect()
        return json.loads(self.browser.execute_script(f'''
            var req = new XMLHttpRequest();
            req.open("GET", "{uri}", false);
            req.send(null);
            return req.responseText;
        '''))


if __name__ == '__main__':
    """When run as a main, the program fetches and prints a stats.nba.com uri.
    Program takes one argument: the url to lead
    """

    ## Exit if the program was run without a single arg: the URL
    if len(sys.argv) != 2:
        print('Program takes one arg: the uri to fetch')
        sys.exit(1)
    
    ## Create an API instance
    api = StatsNbaComApi()

    ## Fetch the data
    uri = sys.argv[1]
    data = api.get(uri)
    print(data)
