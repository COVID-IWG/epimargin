from .commons import download_data

URL      = "https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/"
filename = "OxCGRT_latest.csv"

def download_latest_stringency(dest):
    download_data(dest, filename, base_url = URL)