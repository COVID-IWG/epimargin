from pathlib import Path
import requests


def download_data(data_path: Path, filename: str, base_url: str = 'https://api.covid19india.org/csv/latest/'):
    """ download a file with filename from the base_url to the directory at data_path  """
    url = base_url + filename
    response = requests.get(url)
    with (data_path/filename).open('wb') as dst:
        dst.write(response.content)
